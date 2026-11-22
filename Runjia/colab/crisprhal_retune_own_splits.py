# ============================================================================
# Re-tune crisprHAL 2 on ITS OWN splits, and score it once on ITS OWN test files.
#   -- run in Google Colab on a GPU runtime (Runtime -> Change runtime type -> T4)
#   -- paste the whole file into one cell; set ENZYME below; run SMOKE=1 first (3 minutes)
#
# WHY. Our lead over crisprHAL 2 on its authors' own splits (+0.0286 rho on eSpCas9, +0.0226 on WT-SpCas9) is a lead over
# their RELEASED WEIGHTS. Those were never re-tuned on these splits by us, while our side has had a 200-draw joint search.
# The honest form of the claim today is "a tuned boosted tree beats the released model", not "beats the architecture". This
# script is the missing arm: it gives their architecture a search of its own, chosen without ever touching their test file.
#
# WHAT IT DOES (every step resumes from the CSVs it writes, so a dropped session loses one fit, not the run)
#   stage 1  N_DRAWS random configurations of THEIR OWN hyperparameters (the 11 arguments of models.__init__ plus epochs and
#            batch size), each trained on 80% of THEIR TRAINING FILE and scored on the other 20% (split by protospacer, so a
#            repeated guide cannot sit on both sides). Draw 0 is their published configuration, so the search is anchored and
#            can only match or beat it on the inner split. Epochs are found by early stopping on validation rho (patience 8).
#   stage 2  the TOP_K draws plus draw 0 are re-scored on two FRESH inner partitions at their chosen epoch count, which is what
#            stops the winner being simply the luckiest draw.
#   stage 3  the winner and the published configuration are each retrained on the FULL training file with SEEDS seeds, and
#            scored ONCE on the test file (deduplicated by protospacer exactly as final_eval.py / vs_crisprhal_wt_hal.py do),
#            beside their shipped weights. Per-seed rho and the seed-averaged ensemble are both reported, because our own
#            comparison arm is an average of five.
#
# POSITIVE CONTROLS. (1) their shipped weights must score ~0.6945 (WT) / ~0.7821 (eSp) on the deduplicated test file, or the
# environment or the data handling is wrong. (2) the published configuration retrained here must land within ~0.01 of those; if
# it does not, the training harness is wrong and nothing else here can be trusted.
#
# INTERPRETATION, FIXED IN ADVANCE so it cannot be read after the fact. SLICER (tuned) reaches 0.7172 / 0.7193 (single / average
# of five) on the WT test file and 0.8107 / 0.8104 on the eSp test file. Compare the tuned 5-seed ensemble to the published
# 5-seed ensemble (what tuning bought) and to those figures (what is left of the lead):
#   * gain over the published ensemble <= ~0.005 (about the seed-to-seed spread) -> tuning buys them nothing; the lead stands
#     and the sentence becomes "a tuned boosted tree beats their architecture, not only their release".
#   * tuned ensemble lands between the published ensemble and SLICER                 -> the lead narrows by the gain; quote it.
#   * tuned ensemble reaches SLICER's figure                                         -> the lead was a tuning-effort artefact
#     and the honest result on their splits is parity. Say it plainly.
# A paired bootstrap on guides is needed for any of those (the per-guide predictions are saved for it).
#
# THINGS THAT STAY UNEQUAL, DISCLOSED. (a) The search is 40 draws by default against our 200; more draws only help them, so a
# null here is a lower bound on what a bigger search would find, not proof of none. (b) Inner validation is a random split by
# protospacer, matching how THEIR test file was made; ours was judged on whole held-out genes, which is stricter. (c) SLICER's
# settings were chosen on genes that include their WT test guides (the 1,204 CONFIRM genes, untouched by any search, are the clean
# comparison; see the report's Part 6). (d) GPU kernels are not bit-reproducible, so a re-run moves rho by ~0.002.
#
# COST, from measurement: their published configuration fits one WT fold in 5.5 minutes on a Colab T4. With early stopping a
# draw averages ~8 minutes, so WT is ~8-10 T4-hours in total (40 draws, 6 finalists x 2 partitions, 10 final fits): 2-3 free
# sessions with the resume below. eSpCas9 has 1.8x the rows and a longer input: about 3x that. Run WT first.
#
# WHAT TO BRING BACK (from OUTDIR): retune_<enzyme>_search.csv, _stage2.csv, _final.csv, _summary.json and the three
# preds_final_<wt|espcas9>_crisprhal_{shipped,published5,retuned5}.csv files (columns protospacer,y,pred), which have the same
# format as the stored shipped-weights predictions, so the paired bootstrap against SLICER is one command at home.
# ============================================================================
# %% ---- settings -------------------------------------------------------------------------------------------------------
import os, sys, json, time, subprocess

ENZYME = os.environ.get("ENZYME", "WT")        # "WT" (their SpCas9 split, 26,994 train rows) or "ESP" (eSpCas9, 47,591)
SMOKE = bool(int(os.environ.get("SMOKE", "0")))  # 1 = a few minutes end to end on tiny data, to check the harness
N_DRAWS = int(os.environ.get("N_DRAWS", "40"))   # random draws after draw 0 (their published configuration)
TOP_K = 5                                        # finalists re-scored on fresh partitions
SEEDS = 5                                        # final retrains per arm (our comparison arm is an average of five)
PATIENCE = 8                                     # epochs without a new best validation rho before a draw stops
STOP_AFTER_MIN = None                            # e.g. 220: stop LAUNCHING fits after this many minutes, so a session ends cleanly
SEED = 41
HAL = os.environ.get("HAL", "/content/crisprHAL")
OUTDIR = os.environ.get("OUTDIR", "/content/crisprhal_retune")
if SMOKE:
    N_DRAWS, TOP_K, SEEDS, PATIENCE = 2, 1, 2, 2

try:                                              # Drive survives a disconnect; /content does not
    if not SMOKE and not os.path.isdir("/content/drive/MyDrive"):
        from google.colab import drive; drive.mount("/content/drive")
    if os.path.isdir("/content/drive/MyDrive") and "OUTDIR" not in os.environ:
        OUTDIR = "/content/drive/MyDrive/crisprhal_retune"
except Exception as e:
    print("Drive not mounted (", type(e).__name__, "); results go to", OUTDIR, "and are lost if the runtime resets")
os.makedirs(OUTDIR, exist_ok=True)
if not os.path.isdir(HAL):
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/tbrowne5/crisprHAL.git", HAL], check=True)

# %% ---- environment ----------------------------------------------------------------------------------------------------
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import tensorflow as tf
tf.get_logger().setLevel("ERROR")                # a fresh model per fit makes Keras warn about retracing; it is harmless here
print("tensorflow", tf.__version__, "| outputs ->", OUTDIR)
if not SMOKE:
    assert tf.config.list_physical_devices("GPU"), "no GPU: Runtime -> Change runtime type -> T4 (their GRU is many times slower on CPU)"

sys.path.insert(0, HAL); os.chdir(HAL)
from models import models, modelVersionInputLength, modelVersionDefaultEpochs
from processing import processing

SPEC = {"WT": dict(model="WT-SPCAS9", tag="wt", train="data/WT-SpCas9_training_data.csv", test="data/WT-SpCas9_testing_data.csv",
                    shipped="models/WT-SPCAS9.keras", published_test=0.6945, slicer=(0.7172, 0.7193)),
        "ESP": dict(model="ESPCAS9", tag="espcas9", train="data/eSpCas9_training_data.csv", test="data/eSpCas9_testing_data.csv",
                    shipped="models/ESPCAS9.keras", published_test=0.7821, slicer=(0.8107, 0.8104))}[ENZYME]
MODEL = SPEC["model"]
LEN, UP = modelVersionInputLength[MODEL][0], modelVersionInputLength[MODEL][1]   # context length; nt upstream of the protospacer
DEFAULT_EPOCHS = modelVersionDefaultEpochs[MODEL]
EPOCH_CAP = 2 * DEFAULT_EPOCHS if not SMOKE else 3

DEFAULTS = dict(learning_rate=0.0005, drop_rate=0.3, CNN_filters=128, window_size=3, CNN_drop=0.3, CNN_dense1=128, CNN_dense2=64,
                RNN_size=128, RNN_dense1=128, RNN_dense2=64, CNN_RNN_drop=0.3)     # models.__init__'s own defaults (their published configuration)
NAMES = list(DEFAULTS)

# %% ---- data: exactly their files, deduplicated for scoring exactly as in final_eval.py -------------------------------------
def read(path):
    d = pd.read_csv(path, header=None, names=["context", "score"]).dropna(subset=["context"])
    d["context"] = d["context"].str.upper()
    d = d[d["context"].str.len() == LEN].reset_index(drop=True)
    d["protospacer"] = d["context"].str[UP:UP + 20]
    return d

LUT = np.zeros(256, np.int8) - 1
for i, c in enumerate("ACGT"): LUT[ord(c)] = i
def onehot(contexts):
    """Same encoding as processing.onehotencode (A,C,G,T -> unit vectors; anything else -> zeros), vectorised."""
    a = np.frombuffer("".join(contexts).encode(), np.uint8).reshape(len(contexts), LEN)
    idx = LUT[a]; out = np.zeros((len(contexts), LEN, 4), np.float32)
    r, c = np.nonzero(idx >= 0); out[r, c, idx[r, c]] = 1.0
    return out

tr, te_raw = read(SPEC["train"]), read(SPEC["test"])
if SMOKE:
    tr, te_raw = tr.sample(2000, random_state=SEED).reset_index(drop=True), te_raw.sample(600, random_state=SEED).reset_index(drop=True)
chk = processing().onehotencode(tr["context"].iloc[:50].tolist())
assert np.array_equal(chk, onehot(tr["context"].iloc[:50].tolist())), "encoder differs from theirs"
te = te_raw.groupby("protospacer", as_index=False).agg(context=("context", "first"), score=("score", "mean"))   # one row per protospacer
Xtr, ytr = onehot(tr["context"].tolist()), tr["score"].to_numpy(float)
Xte, yte = onehot(te["context"].tolist()), te["score"].to_numpy(float)
print(f"{ENZYME}: {len(tr):,} training rows ({tr.protospacer.nunique():,} protospacers), {len(te_raw):,} raw test rows -> "
      f"{len(te):,} scored protospacers; input {LEN} nt, default {DEFAULT_EPOCHS} epochs")

def inner_split(seed, frac=0.2):
    """Validation = a random 20% of protospacers (so a repeated guide never straddles the split)."""
    u = tr["protospacer"].unique(); rng = np.random.default_rng(seed); va_p = set(rng.permutation(u)[:int(round(frac * len(u)))])
    m = tr["protospacer"].isin(va_p).to_numpy()
    return np.where(~m)[0], np.where(m)[0]

# %% ---- one fit --------------------------------------------------------------------------------------------------------
class ValRho(tf.keras.callbacks.Callback):
    """Validation Spearman after every epoch; stops after PATIENCE epochs without a new best. Does NOT restore weights: the
    chosen epoch count is used to retrain, so the validation rows never become part of what is scored."""
    def __init__(self, Xv, yv, patience):
        super().__init__(); self.Xv, self.yv, self.patience = Xv, yv, patience; self.hist, self.best, self.best_ep, self.bad = [], -np.inf, 0, 0
    def on_epoch_end(self, epoch, logs=None):
        p = np.asarray(self.model.predict(self.Xv, batch_size=2048, verbose=0)).ravel()
        r = float(spearmanr(self.yv, p).statistic) if np.isfinite(p).all() and p.std() > 0 else -1.0
        self.hist.append(r)
        if r > self.best: self.best, self.best_ep, self.bad = r, epoch + 1, 0
        else:
            self.bad += 1
            if self.bad >= self.patience: self.model.stop_training = True

def build(cfg, seed):
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
    m = models(MODEL, summary=False, **{k: cfg[k] for k in NAMES})
    return m.model            # call Keras directly: their train()/predict() wrappers swallow exceptions and return None on failure

def fit(cfg, epochs, X, y, seed, callbacks=None, validation=None):
    """fit with a batch-size fallback (an out-of-memory error halves the batch and retries); returns (model, batch_used)."""
    bs = int(cfg["batch_size"])
    while True:
        model = build(cfg, seed)
        try:
            model.fit(X, y, epochs=epochs, batch_size=bs, verbose=0, callbacks=callbacks or [])
            return model, bs
        except Exception as e:
            if ("RESOURCE_EXHAUSTED" in str(e) or "OOM" in str(e) or "Resource exhausted" in str(e)) and bs > 64:
                bs //= 2; print(f"    out of memory, retrying at batch {bs}", flush=True); continue
            raise

def predict(model, X):
    p = np.asarray(model.predict(X, batch_size=2048, verbose=0)).ravel()
    assert np.isfinite(p).all() and p.std() > 0, "degenerate predictions (an untrained or diverged network)"
    return p

# %% ---- the search space and the draws ---------------------------------------------------------------------------------
def draw(i):
    """Draw 0 is their published configuration. Draw i>0 depends only on (SEED, i), so raising N_DRAWS later never changes earlier draws."""
    if i == 0: return dict(DEFAULTS, batch_size=1024)
    r = np.random.default_rng([SEED, i]); pick = lambda xs: xs[int(r.integers(len(xs)))]
    return dict(learning_rate=float(np.exp(r.uniform(np.log(1e-4), np.log(2e-3)))), drop_rate=pick([0.1, 0.2, 0.3, 0.4, 0.5]),
                CNN_filters=pick([64, 96, 128, 192, 256]), window_size=pick([3, 5, 7]), CNN_drop=pick([0.1, 0.2, 0.3, 0.4, 0.5]),
                CNN_dense1=pick([64, 128, 256, 512]), CNN_dense2=pick([32, 64, 128, 256]), RNN_size=pick([32, 64, 128, 192, 256]),
                RNN_dense1=pick([64, 128, 256, 512]), RNN_dense2=pick([32, 64, 128, 256]), CNN_RNN_drop=pick([0.1, 0.2, 0.3, 0.4, 0.5]),
                batch_size=pick([256, 512, 1024, 2048]))

def load(name, cols):
    f = f"{OUTDIR}/retune_{SPEC['tag']}_{name}.csv"
    return (pd.read_csv(f) if os.path.exists(f) else pd.DataFrame(columns=cols)), f

T0 = time.time()
def out_of_time(): return STOP_AFTER_MIN is not None and (time.time() - T0) / 60 > STOP_AFTER_MIN

# %% ---- stage 1: the search --------------------------------------------------------------------------------------------
S1, f1 = load("search", ["draw"])
itr, iva = inner_split(SEED)
for i in range(N_DRAWS + 1):
    if i in set(S1["draw"]): continue
    if out_of_time(): print("time budget reached; re-run this cell to resume"); break
    cfg = draw(i); t0 = time.time()
    cb = ValRho(Xtr[iva], ytr[iva], PATIENCE)
    try:
        _, bs = fit(cfg, EPOCH_CAP, Xtr[itr], ytr[itr], SEED, callbacks=[cb])
        ok = True
    except Exception as e:
        print(f"  draw {i}: failed ({type(e).__name__}: {str(e)[:90]})"); ok, bs = False, cfg["batch_size"]
    row = dict(draw=i, val_rho=cb.best if ok and cb.hist else -1.0, best_epoch=cb.best_ep, epochs_run=len(cb.hist),
               rho_at_default_epochs=(cb.hist[DEFAULT_EPOCHS - 1] if len(cb.hist) >= DEFAULT_EPOCHS else np.nan),
               batch_used=bs, minutes=round((time.time() - t0) / 60, 1), is_published=(i == 0),
               config=json.dumps({**cfg, "batch_size": bs}), curve=";".join(f"{x:.4f}" for x in cb.hist))
    S1 = pd.concat([S1, pd.DataFrame([row])], ignore_index=True); S1.to_csv(f1, index=False)
    print(f"  draw {i:3d}{' (published)' if i == 0 else ''}: val rho {row['val_rho']:.4f} at epoch {row['best_epoch']:>3d} "
          f"({row['epochs_run']} run, {row['minutes']} min, batch {bs})", flush=True)
done1 = S1[S1["draw"] <= N_DRAWS]
print(f"\nstage 1: {len(done1)}/{N_DRAWS + 1} draws; published configuration val rho "
      f"{float(S1.loc[S1.draw == 0, 'val_rho'].iloc[0]):.4f}; best draw {int(done1.sort_values('val_rho').iloc[-1]['draw'])} at {done1['val_rho'].max():.4f}")

# %% ---- stage 2: re-score the finalists on fresh partitions ------------------------------------------------------------
cols2 = ["draw", "partition", "rho"]
S2, f2 = load("stage2", cols2)
if len(done1) == N_DRAWS + 1:
    ranked = done1[done1["draw"] > 0].sort_values("val_rho", ascending=False)["draw"].tolist()[:TOP_K]
    finalists = [0] + ranked
    for d in finalists:
        r1 = S1[S1["draw"] == d].iloc[0]; cfg = json.loads(r1["config"]); ep = max(int(r1["best_epoch"]), 1)
        if d == 0: ep = DEFAULT_EPOCHS if not SMOKE else EPOCH_CAP            # the published configuration is re-scored at ITS epochs
        for part in (1, 2):
            if ((S2["draw"] == d) & (S2["partition"] == part)).any(): continue
            if out_of_time(): print("time budget reached; re-run this cell to resume"); break
            a, b = inner_split(SEED + part)
            m, _ = fit(cfg, ep, Xtr[a], ytr[a], SEED + part)
            rho = float(spearmanr(ytr[b], predict(m, Xtr[b])).statistic)
            S2 = pd.concat([S2, pd.DataFrame([dict(draw=d, partition=part, rho=rho, epochs=ep)])], ignore_index=True); S2.to_csv(f2, index=False)
            print(f"  finalist {d:3d} partition {part}: rho {rho:.4f} at {ep} epochs", flush=True)
    s2 = S2.groupby("draw")["rho"].agg(["mean", "count"]); s2 = s2[s2["count"] == 2]
    if set(finalists) <= set(s2.index):
        winner = int(s2["mean"].idxmax())
        print(f"\nstage 2: winner is draw {winner} (fresh-partition rho {s2.loc[winner, 'mean']:.4f}; published configuration {s2.loc[0, 'mean']:.4f})")
    else:
        winner = None; print("stage 2 incomplete; re-run this cell")
else:
    winner = None; print("stage 1 incomplete; re-run the previous cell")

# %% ---- stage 3: retrain on the full training file, score the test file ONCE -----------------------------------------------
if winner is not None:
    F, f3 = load("final", ["arm", "seed"])
    arms = {"published": (draw(0), DEFAULT_EPOCHS if not SMOKE else EPOCH_CAP)}
    w = S1[S1["draw"] == winner].iloc[0]; arms["retuned"] = (json.loads(w["config"]), max(int(w["best_epoch"]), 1) if winner != 0 else arms["published"][1])
    print("winner configuration:", arms["retuned"][0], "at", arms["retuned"][1], "epochs")
    P = {}
    for arm, (cfg, ep) in arms.items():
        for s in range(SEEDS):
            if arm == "retuned" and winner == 0 and (("published", s) in P):      # the published configuration won: no second set of fits
                P[(arm, s)] = P[("published", s)]; continue
            f_p = f"{OUTDIR}/_pred_{SPEC['tag']}_{arm}_{s}.npy"
            if os.path.exists(f_p): P[(arm, s)] = np.load(f_p); continue
            if out_of_time(): print("time budget reached; re-run this cell to resume"); break
            t0 = time.time(); m, _ = fit(cfg, ep, Xtr, ytr, SEED + 100 + s); p = predict(m, Xte); np.save(f_p, p); P[(arm, s)] = p
            print(f"  {arm:9s} seed {s}: test rho {spearmanr(yte, p).statistic:.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    # their shipped weights, the positive control
    ship = None
    try:
        import keras
        ship = np.asarray(keras.models.load_model(SPEC["shipped"], compile=False).predict(Xte, batch_size=2048, verbose=0)).ravel()
    except Exception as e:
        print("could not load their shipped weights (", type(e).__name__, str(e)[:100], ") -- positive control 1 unavailable")
    rho = lambda p: float(spearmanr(yte, p).statistic)
    rows = []
    for arm in arms:
        ps = [P[(arm, s)] for s in range(SEEDS) if (arm, s) in P]
        if len(ps) < SEEDS: continue
        r_each = [rho(p) for p in ps]; ens = np.mean(ps, axis=0)
        rows += [dict(arm=arm, seed=s, test_rho=r) for s, r in enumerate(r_each)] + [dict(arm=arm, seed="mean of seeds", test_rho=float(np.mean(r_each))),
                                                                                       dict(arm=arm, seed="ensemble", test_rho=rho(ens))]
        pd.DataFrame({"protospacer": te["protospacer"], "y": yte, "pred": ens}).to_csv(f"{OUTDIR}/preds_final_{SPEC['tag']}_crisprhal_{arm}5.csv", index=False)
    if ship is not None:
        rows.append(dict(arm="shipped weights", seed="-", test_rho=rho(ship)))
        pd.DataFrame({"protospacer": te["protospacer"], "y": yte, "pred": ship}).to_csv(f"{OUTDIR}/preds_final_{SPEC['tag']}_crisprhal_shipped.csv", index=False)
    R = pd.DataFrame(rows); R.to_csv(f3, index=False)
    g = lambda arm, seed: float(R[(R.arm == arm) & (R.seed == seed)]["test_rho"].iloc[0]) if ((R.arm == arm) & (R.seed == seed)).any() else float("nan")
    summ = dict(enzyme=ENZYME, n_test=int(len(te)), shipped=g("shipped weights", "-"), published_mean=g("published", "mean of seeds"),
                published_ensemble=g("published", "ensemble"), retuned_mean=g("retuned", "mean of seeds"), retuned_ensemble=g("retuned", "ensemble"),
                winner_draw=winner, n_draws=N_DRAWS, slicer_single=SPEC["slicer"][0], slicer_avg5=SPEC["slicer"][1])
    summ["gain_from_tuning_ensemble"] = summ["retuned_ensemble"] - summ["published_ensemble"]
    summ["slicer_lead_over_retuned_ensemble"] = SPEC["slicer"][1] - summ["retuned_ensemble"]
    json.dump(summ, open(f"{OUTDIR}/retune_{SPEC['tag']}_summary.json", "w"), indent=1)
    print("\n" + "=" * 78 + f"\n{ENZYME}: test rho on {len(te):,} protospacers (their test file, scored once)\n" + "=" * 78)
    for k, v in summ.items(): print(f"  {k:36s} {v:.4f}" if isinstance(v, float) else f"  {k:36s} {v}")
    print(f"\n  positive control 1: shipped weights {summ['shipped']:.4f} (expect ~{SPEC['published_test']})")
    print(f"  positive control 2: published configuration retrained {summ['published_mean']:.4f} (expect within ~0.01 of the shipped weights)")
    print("  Reading rule (fixed in advance): gain from tuning <= ~0.005 -> the lead stands; between -> it narrows by the gain; "
          "reaches SLICER -> parity. Bring back OUTDIR for the paired bootstrap.")
