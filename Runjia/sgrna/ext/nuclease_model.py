"""Does telling a model which nuclease it is scoring help, and does it help for an unseen one?

Data: Kim et al. 2020 (Nat Biotechnol 38:1328), Supplementary Table 6 -- 40,128 target sequences
(4 nt + 20-nt protospacer + 3-nt PAM + 3 nt, 30-mer) measured in the same library in nine nucleases
(SpCas9, SpCas9-NG, VRQR, xCas9, Sniper-Cas9, eSpCas9(1.1), SpCas9-HF1, HypaCas9, evoCas9), with the
authors' own train/test split (34,713 / 5,415) and a library covariate (tRNA-N20 versus (G/g)N19
sgRNA). The sheet lumps four kinds of column: the input sequence, a library covariate, the split
flag, and nine outcome columns that are not all measured on all targets.

Every arm uses the same compact featurisation of the 30-mer (one-hot, dinucleotide one-hot, a few
composition columns, the tRNA covariate) and the same XGBoost settings, so arms differ only in what
they are told about the nuclease.

    python -m sgrna.ext.nuclease_model --stage 1      # in-distribution
    python -m sgrna.ext.nuclease_model --stage 2      # leave one nuclease out
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

from . import nuclease_desc as nd
from .common import KIM_CSV, OUT

CODE = np.full(256, 0, dtype=np.int8)
for i, ch in enumerate("ACGT"):
    CODE[ord(ch)] = i
for i, ch in enumerate("acgt"):
    CODE[ord(ch)] = i

XGB = dict(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.6,
           min_child_weight=5, tree_method="hist", n_jobs=-1, random_state=41)


def load():
    d = pd.read_csv(KIM_CSV).iloc[:, :12]
    d.columns = ["seq", "trna", "split"] + nd.SHORT
    for c in nd.SHORT:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def featurise(seqs, trna):
    """Column groups, returned separately so they can be ablated by kind."""
    c = CODE[np.frombuffer("".join(seqs).encode(), dtype=np.uint8)].reshape(len(seqs), 30).astype(np.int64)
    n = len(seqs)
    oh = np.zeros((n, 30, 4), np.float32)
    oh[np.arange(n)[:, None], np.arange(30)[None], c] = 1
    di = np.zeros((n, 29, 16), np.float32)
    di[np.arange(n)[:, None], np.arange(29)[None], c[:, :-1] * 4 + c[:, 1:]] = 1
    sp = c[:, 4:24]
    gc = ((sp == 1) | (sp == 2)).astype(np.float32)
    comp = np.column_stack([gc.sum(1), gc[:, :10].sum(1), gc[:, 10:].sum(1),
                            *[(sp == k).sum(1) for k in range(4)]]).astype(np.float32)
    groups = {
        "spacer_onehot": oh[:, 4:24].reshape(n, -1),
        "pam_onehot": oh[:, 24:27].reshape(n, -1),
        "flank_onehot": np.hstack([oh[:, :4].reshape(n, -1), oh[:, 27:].reshape(n, -1)]),
        "dinuc": di.reshape(n, -1),
        "composition": comp,
        "library_trna": np.asarray(trna, np.float32)[:, None],
    }
    names = {k: [f"{k}.{i}" for i in range(v.shape[1])] for k, v in groups.items()}
    return groups, names


GROUPS_ALL = ["spacer_onehot", "pam_onehot", "flank_onehot", "dinuc", "composition", "library_trna"]


def stack(groups, use):
    return np.hstack([groups[g] for g in use])


def fit_predict(Xtr, ytr, Xte, params=None):
    m = xgb.XGBRegressor(**(params or XGB))
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def rho(y, p):
    return float(spearmanr(y, p).statistic) if len(y) > 2 else np.nan


def rows_for(d, nuc, split):
    m = (d.split == split) & d[nuc].notna()
    return np.where(m.to_numpy())[0]


def append(path, rows):
    df = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)


def descriptor_matrix(group_names):
    g = nd.all_groups()
    df = pd.concat([g[k] for k in group_names], axis=1)
    return (df - df.mean()) / df.std().replace(0, 1)       # z-scored over the nine enzymes


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--perms", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args(argv)
    d = load()
    G, names = featurise(d.seq.tolist(), d.trna.to_numpy())
    pam = d.seq.str[25:27].to_numpy()
    ngg = (pam == "GG")
    base_use = GROUPS_ALL

    if a.stage == 1:
        out = OUT / "nuclease_indist.csv"
        done = set(pd.read_csv(out).arm) if out.exists() else set()
        # --- per-nuclease models, and column-kind ablations on them
        for label, use in (("all", GROUPS_ALL),
                           ("-library_trna", [g for g in GROUPS_ALL if g != "library_trna"]),
                           ("-pam_onehot", [g for g in GROUPS_ALL if g != "pam_onehot"]),
                           ("-flank_onehot", [g for g in GROUPS_ALL if g != "flank_onehot"]),
                           ("-dinuc", [g for g in GROUPS_ALL if g != "dinuc"]),
                           ("-composition", [g for g in GROUPS_ALL if g != "composition"]),
                           ("spacer_onehot only", ["spacer_onehot"]),
                           ("spacer+pam onehot only", ["spacer_onehot", "pam_onehot"])):
            arm = f"per-nuclease [{label}]"
            if arm in done:
                continue
            X = stack(G, use)
            rws = []
            for k, nuc in zip(nd.SHORT, nd.NUCLEASES):
                tr, te = rows_for(d, k, "Training"), rows_for(d, k, "Test")
                t0 = time.time()
                p = fit_predict(X[tr], d[k].to_numpy()[tr], X[te])
                rws.append(dict(arm=arm, nuclease=nuc, n_train=len(tr), n_test=len(te),
                                rho=rho(d[k].to_numpy()[te], p),
                                rho_ngg=rho(d[k].to_numpy()[te][ngg[te]], p[ngg[te]])))
                print(f"  {arm:34s} {nuc:14s} rho {rws[-1]['rho']:.4f} (NGG {rws[-1]['rho_ngg']:.4f}) "
                      f"{time.time() - t0:.0f}s", flush=True)
            append(out, rws)
        # --- pooled arms
        X = stack(G, base_use)
        idx_tr = {k: rows_for(d, k, "Training") for k in nd.SHORT}
        idx_te = {k: rows_for(d, k, "Test") for k in nd.SHORT}
        desc_sets = {"pooled [no nuclease info]": None, "pooled [one-hot identity]": "onehot"}
        for gname in ("mutations", "structure", "pam", "scale"):
            desc_sets[f"pooled [{gname}]"] = [gname]
        desc_sets["pooled [mutations+structure]"] = ["mutations", "structure"]
        desc_sets["pooled [all descriptors]"] = ["mutations", "structure", "pam", "scale"]
        for arm, spec in desc_sets.items():
            if arm in done:
                continue
            if spec is None:
                D = np.zeros((9, 0), np.float32)
            elif spec == "onehot":
                D = np.eye(9, dtype=np.float32)
            else:
                D = descriptor_matrix(spec).to_numpy().astype(np.float32)
            Xtr = np.vstack([np.hstack([X[idx_tr[k]], np.tile(D[j], (len(idx_tr[k]), 1))])
                             for j, k in enumerate(nd.SHORT)])
            ytr = np.concatenate([d[k].to_numpy()[idx_tr[k]] for k in nd.SHORT])
            t0 = time.time()
            m = xgb.XGBRegressor(**XGB).fit(Xtr, ytr)
            rws = []
            for j, k in enumerate(nd.SHORT):
                te = idx_te[k]
                p = m.predict(np.hstack([X[te], np.tile(D[j], (len(te), 1))]))
                rws.append(dict(arm=arm, nuclease=nd.NUCLEASES[j], n_train=len(idx_tr[k]), n_test=len(te),
                                rho=rho(d[k].to_numpy()[te], p),
                                rho_ngg=rho(d[k].to_numpy()[te][ngg[te]], p[ngg[te]])))
            append(out, rws)
            print(f"  {arm:34s} mean rho {np.mean([r['rho'] for r in rws]):.4f}  {time.time() - t0:.0f}s",
                  flush=True)
    elif a.stage == 3:
        # in-distribution null for the descriptor gain: same descriptors, nuclease rows permuted
        out = OUT / "nuclease_indist_null.csv"
        X = stack(G, base_use)
        idx_tr = {k: rows_for(d, k, "Training") for k in nd.SHORT}
        idx_te = {k: rows_for(d, k, "Test") for k in nd.SHORT}
        rng = np.random.default_rng(1)
        for spec_name, spec in (("all descriptors", ["mutations", "structure", "pam", "scale"]),
                                ("structure", ["structure"])):
            D0 = descriptor_matrix(spec).to_numpy().astype(np.float32)
            for pi in range(a.perms):
                perm = rng.permutation(9)
                D = D0[perm]
                Xtr = np.vstack([np.hstack([X[idx_tr[k]], np.tile(D[j], (len(idx_tr[k]), 1))])
                                 for j, k in enumerate(nd.SHORT)])
                ytr = np.concatenate([d[k].to_numpy()[idx_tr[k]] for k in nd.SHORT])
                m = xgb.XGBRegressor(**XGB).fit(Xtr, ytr)
                rws = []
                for j, k in enumerate(nd.SHORT):
                    te = idx_te[k]
                    p = m.predict(np.hstack([X[te], np.tile(D[j], (len(te), 1))]))
                    rws.append(dict(arm=f"pooled [{spec_name}] permuted", perm=pi, nuclease=nd.NUCLEASES[j],
                                    rho=rho(d[k].to_numpy()[te], p),
                                    rho_ngg=rho(d[k].to_numpy()[te][ngg[te]], p[ngg[te]])))
                append(out, rws)
                print(f"  {spec_name} perm {pi}: mean rho {np.mean([r['rho'] for r in rws]):.4f} "
                      f"NGG {np.mean([r['rho_ngg'] for r in rws]):.4f}", flush=True)
    elif a.stage == 4:
        # replicated leave-one-nuclease-out, predictions saved so any metric can be recomputed
        out = OUT / "nuclease_lono_rep.csv"
        pred_dir = OUT / "lono_preds"; pred_dir.mkdir(exist_ok=True)
        done = set(zip(*[pd.read_csv(out)[c] for c in ("arm", "nuclease", "rep")])) if out.exists() else set()
        X = stack(G, base_use)
        idx_tr = {k: rows_for(d, k, "Training") for k in nd.SHORT}
        idx_te = {k: rows_for(d, k, "Test") for k in nd.SHORT}
        free = ["mutations", "structure"]                       # no measurement on the new enzyme
        labelled = ["mutations", "structure", "pam", "scale"]   # PAM screen + training-split activity
        specs = {"U": None, "D_mutations": ["mutations"], "D_structure": ["structure"],
                 "D_label-free": free, "D_pam": ["pam"], "D_scale": ["scale"], "D_all": labelled}
        for rep in range(a.reps):
            rng = np.random.default_rng(100 + rep)
            params = dict(XGB, random_state=41 + rep)
            for hi, (hk, hn) in enumerate(zip(nd.SHORT, nd.NUCLEASES)):
                others = [j for j in range(9) if j != hi]
                te = idx_te[hk]; y_te = d[hk].to_numpy()[te]
                sub = {j: rng.choice(idx_tr[nd.SHORT[j]], size=min(len(idx_tr[nd.SHORT[j]]), 15000), replace=False)
                       for j in others}
                ytr = np.concatenate([d[nd.SHORT[j]].to_numpy()[sub[j]] for j in others])
                rows = []
                def rec(arm, p):
                    np.save(pred_dir / f"{arm}__{hn}__r{rep}.npy", p.astype(np.float32))
                    rows.append(dict(arm=arm, nuclease=hn, rep=rep, rho=rho(y_te, p),
                                     rho_ngg=rho(y_te[ngg[te]], p[ngg[te]])))
                def fit_with(D, perm=None):
                    Dm = D if perm is None else D[perm]
                    Xtr = np.vstack([np.hstack([X[sub[j]], np.tile(Dm[j], (len(sub[j]), 1))]) for j in others])
                    return xgb.XGBRegressor(**params).fit(Xtr, ytr).predict(
                        np.hstack([X[te], np.tile(Dm[hi], (len(te), 1))]))
                for arm, spec in specs.items():
                    if (arm, hn, rep) in done:
                        continue
                    D = np.zeros((9, 0), np.float32) if spec is None else descriptor_matrix(spec).to_numpy().astype(np.float32)
                    rec(arm, fit_with(D))
                for nm_, spec in (("D_label-free", free), ("D_all", labelled)):
                    D = descriptor_matrix(spec).to_numpy().astype(np.float32)
                    for pi in range(a.perms):
                        arm = f"{nm_}_perm{pi}"
                        if (arm, hn, rep) in done:
                            continue
                        perm = rng.permutation(9)
                        while perm[hi] == hi:
                            perm = rng.permutation(9)
                        rec(arm, fit_with(D, perm))
                # per-source models applied to the held-out nuclease, and their mean
                if ("ensemble-of-sources", hn, rep) not in done:
                    ps = []
                    for j in others:
                        k = nd.SHORT[j]
                        ps.append(fit_predict(X[sub[j]], d[k].to_numpy()[sub[j]], X[te], params))
                    rec("ensemble-of-sources", np.mean(ps, axis=0))
                if ("own", hn, rep) not in done:
                    tr = idx_tr[hk]
                    rec("own", fit_predict(X[tr], d[hk].to_numpy()[tr], X[te], params))
                append(out, rows)
                print(f"  rep {rep} held out {hn:13s} " + " ".join(
                    f"{r['arm']}={r['rho']:.3f}" for r in rows if "perm" not in r["arm"]), flush=True)
    elif a.stage == 6:
        # few-shot: k labelled targets of the "new" nuclease are added to the pooled training set
        out = OUT / "nuclease_fewshot.csv"
        done = set(zip(*[pd.read_csv(out)[c] for c in ("arm", "nuclease", "k", "rep")])) if out.exists() else set()
        X = stack(G, base_use)
        idx_tr = {k: rows_for(d, k, "Training") for k in nd.SHORT}
        idx_te = {k: rows_for(d, k, "Test") for k in nd.SHORT}
        free = ["mutations", "structure"]
        Dfree = descriptor_matrix(free).to_numpy().astype(np.float32)
        Done = np.eye(9, dtype=np.float32)
        for rep in range(a.reps):
            rng = np.random.default_rng(300 + rep)
            params = dict(XGB, random_state=41 + rep)
            for hi, (hk, hn) in enumerate(zip(nd.SHORT, nd.NUCLEASES)):
                others = [j for j in range(9) if j != hi]
                te = idx_te[hk]; y_te = d[hk].to_numpy()[te]
                sub = {j: rng.choice(idx_tr[nd.SHORT[j]], size=min(len(idx_tr[nd.SHORT[j]]), 15000), replace=False)
                       for j in others}
                ytr_o = np.concatenate([d[nd.SHORT[j]].to_numpy()[sub[j]] for j in others])
                n_o = len(ytr_o)
                pool = rng.permutation(idx_tr[hk])
                rows = []
                for kk in (0, 100, 500, 2000):
                    ks = pool[:kk]
                    for arm, D in (("pooled + label-free descriptors + k", Dfree), ("pooled + one-hot + k", Done)):
                        if (arm, hn, kk, rep) in done:
                            continue
                        Xtr = np.vstack([np.hstack([X[sub[j]], np.tile(D[j], (len(sub[j]), 1))]) for j in others]
                                        + ([np.hstack([X[ks], np.tile(D[hi], (kk, 1))])] if kk else []))
                        ytr = np.concatenate([ytr_o] + ([d[hk].to_numpy()[ks]] if kk else []))
                        w = np.concatenate([np.ones(n_o)] + ([np.full(kk, 0.1 * n_o / (0.9 * kk))] if kk else []))
                        m = xgb.XGBRegressor(**params).fit(Xtr, ytr, sample_weight=w)
                        p = m.predict(np.hstack([X[te], np.tile(D[hi], (len(te), 1))]))
                        rows.append(dict(arm=arm, nuclease=hn, k=kk, rep=rep, rho=rho(y_te, p),
                                         rho_ngg=rho(y_te[ngg[te]], p[ngg[te]])))
                    if kk and ("own model on k only", hn, kk, rep) not in done:
                        p = fit_predict(X[ks], d[hk].to_numpy()[ks], X[te], dict(params, n_estimators=200, max_depth=4))
                        rows.append(dict(arm="own model on k only", nuclease=hn, k=kk, rep=rep, rho=rho(y_te, p),
                                         rho_ngg=rho(y_te[ngg[te]], p[ngg[te]])))
                append(out, rows)
                print(f"  rep {rep} {hn:13s} " + " ".join(f"{r['arm'][:14]}|{r['k']}={r['rho']:.3f}" for r in rows if r['k'] in (0, 500)), flush=True)
    elif a.stage == 9:
        # a stronger k-only baseline: hyperparameters chosen by 4-fold CV on the k labelled targets themselves
        from sklearn.model_selection import KFold
        out = OUT / "nuclease_fewshot_tuned.csv"
        X = stack(G, base_use)
        idx_tr = {k: rows_for(d, k, "Training") for k in nd.SHORT}
        idx_te = {k: rows_for(d, k, "Test") for k in nd.SHORT}
        grid = [dict(n_estimators=ne, max_depth=md, learning_rate=lr, colsample_bytree=cs)
                for ne in (100, 300) for md in (3, 5, 7) for lr in (0.05, 0.15) for cs in (0.3, 0.6)]
        for rep in range(a.reps):
            rng = np.random.default_rng(300 + rep)          # same draw of the k targets as stage 6
            rows = []
            for hi, (hk, hn) in enumerate(zip(nd.SHORT, nd.NUCLEASES)):
                others = [j for j in range(9) if j != hi]
                te = idx_te[hk]; y_te = d[hk].to_numpy()[te]
                for j in others:                            # consume the rng as stage 6 does
                    rng.choice(idx_tr[nd.SHORT[j]], size=min(len(idx_tr[nd.SHORT[j]]), 15000), replace=False)
                pool = rng.permutation(idx_tr[hk])
                for kk in (100, 500, 2000):
                    ks = pool[:kk]; yk = d[hk].to_numpy()[ks]
                    best = (-9, None)
                    for g_ in grid:
                        sc = []
                        for tr, va in KFold(4, shuffle=True, random_state=0).split(ks):
                            pr = xgb.XGBRegressor(**dict(XGB, **g_, n_jobs=-1)).fit(X[ks][tr], yk[tr]).predict(X[ks][va])
                            sc.append(rho(yk[va], pr))
                        if np.mean(sc) > best[0]:
                            best = (np.mean(sc), g_)
                    p = xgb.XGBRegressor(**dict(XGB, **best[1], n_jobs=-1)).fit(X[ks], yk).predict(X[te])
                    rows.append(dict(arm="own model on k only (tuned by CV)", nuclease=hn, k=kk, rep=rep, rho=rho(y_te, p),
                                     rho_ngg=rho(y_te[ngg[te]], p[ngg[te]]), chosen=str(best[1])))
                print(f"  rep {rep} {hn:13s} " + " ".join(f"k={r['k']}:{r['rho']:.3f}" for r in rows if r['nuclease'] == hn), flush=True)
            append(out, rows)
    elif a.stage == 8:
        # replicates 1..R-1 of the high-capacity leave-one-nuclease-out run (stage 7 is replicate 0)
        out = OUT / "nuclease_lono_bigcap_reps.csv"
        pred_dir = OUT / "lono_preds_bigcap"; pred_dir.mkdir(exist_ok=True)
        done = set(zip(*[pd.read_csv(out)[c] for c in ("arm", "nuclease", "rep")])) if out.exists() else set()
        X = stack(G, base_use)
        idx_tr = {k: rows_for(d, k, "Training") for k in nd.SHORT}
        idx_te = {k: rows_for(d, k, "Test") for k in nd.SHORT}
        specs = {"U": None, "D_label-free": ["mutations", "structure"], "D_all": ["mutations", "structure", "pam", "scale"]}
        for rep in range(1, a.reps):
            rng = np.random.default_rng(100 + rep)
            params = dict(XGB, n_estimators=1500, max_depth=8, random_state=41 + rep)
            for hi, (hk, hn) in enumerate(zip(nd.SHORT, nd.NUCLEASES)):
                others = [j for j in range(9) if j != hi]
                te = idx_te[hk]; y_te = d[hk].to_numpy()[te]
                sub = {j: rng.choice(idx_tr[nd.SHORT[j]], size=min(len(idx_tr[nd.SHORT[j]]), 15000), replace=False)
                       for j in others}
                ytr = np.concatenate([d[nd.SHORT[j]].to_numpy()[sub[j]] for j in others])
                rows = []
                for arm, spec in specs.items():
                    if (arm, hn, rep) in done:
                        continue
                    D = np.zeros((9, 0), np.float32) if spec is None else descriptor_matrix(spec).to_numpy().astype(np.float32)
                    Xtr = np.vstack([np.hstack([X[sub[j]], np.tile(D[j], (len(sub[j]), 1))]) for j in others])
                    p = xgb.XGBRegressor(**params).fit(Xtr, ytr).predict(np.hstack([X[te], np.tile(D[hi], (len(te), 1))]))
                    np.save(pred_dir / f"{arm}__{hn}__r{rep}.npy", p.astype(np.float32))
                    rows.append(dict(arm=arm, nuclease=hn, rep=rep, rho=rho(y_te, p), rho_ngg=rho(y_te[ngg[te]], p[ngg[te]])))
                append(out, rows)
                print(f"  rep {rep} held out {hn:13s} " + " ".join(f"{r['arm']}={r['rho']:.3f}" for r in rows), flush=True)
    elif a.stage == 7:
        # is the leave-one-nuclease-out result a capacity artefact? repeat at 1500 trees, depth 8
        out = OUT / "nuclease_lono_bigcap.csv"
        pred_dir = OUT / "lono_preds_bigcap"; pred_dir.mkdir(exist_ok=True)
        done = set(zip(*[pd.read_csv(out)[c] for c in ("arm", "nuclease")])) if out.exists() else set()
        X = stack(G, base_use)
        idx_tr = {k: rows_for(d, k, "Training") for k in nd.SHORT}
        idx_te = {k: rows_for(d, k, "Test") for k in nd.SHORT}
        specs = {"U": None, "D_label-free": ["mutations", "structure"], "D_all": ["mutations", "structure", "pam", "scale"]}
        rng = np.random.default_rng(100)                  # same subsamples as replicate 0 of stage 4
        params = dict(XGB, n_estimators=1500, max_depth=8, random_state=41)
        for hi, (hk, hn) in enumerate(zip(nd.SHORT, nd.NUCLEASES)):
            others = [j for j in range(9) if j != hi]
            te = idx_te[hk]; y_te = d[hk].to_numpy()[te]
            sub = {j: rng.choice(idx_tr[nd.SHORT[j]], size=min(len(idx_tr[nd.SHORT[j]]), 15000), replace=False)
                   for j in others}
            ytr = np.concatenate([d[nd.SHORT[j]].to_numpy()[sub[j]] for j in others])
            rows = []
            for arm, spec in specs.items():
                if (arm, hn) in done:
                    continue
                D = np.zeros((9, 0), np.float32) if spec is None else descriptor_matrix(spec).to_numpy().astype(np.float32)
                Xtr = np.vstack([np.hstack([X[sub[j]], np.tile(D[j], (len(sub[j]), 1))]) for j in others])
                p = xgb.XGBRegressor(**params).fit(Xtr, ytr).predict(np.hstack([X[te], np.tile(D[hi], (len(te), 1))]))
                np.save(pred_dir / f"{arm}__{hn}.npy", p.astype(np.float32))
                rows.append(dict(arm=arm, nuclease=hn, rho=rho(y_te, p), rho_ngg=rho(y_te[ngg[te]], p[ngg[te]])))
            append(out, rows)
            print(f"  held out {hn:13s} " + " ".join(f"{r['arm']}={r['rho']:.3f}" for r in rows), flush=True)
    elif a.stage == 5:
        # is "pooling does not beat per-nuclease fits" a capacity artefact? give the pooled model more room
        out = OUT / "nuclease_capacity.csv"
        done = set(pd.read_csv(out).arm) if out.exists() else set()
        X = stack(G, base_use)
        idx_tr = {k: rows_for(d, k, "Training") for k in nd.SHORT}
        idx_te = {k: rows_for(d, k, "Test") for k in nd.SHORT}
        D = np.eye(9, dtype=np.float32)
        Xtr = np.vstack([np.hstack([X[idx_tr[k]], np.tile(D[j], (len(idx_tr[k]), 1))]) for j, k in enumerate(nd.SHORT)])
        ytr = np.concatenate([d[k].to_numpy()[idx_tr[k]] for k in nd.SHORT])
        for ne, md in ((500, 6), (1500, 6), (500, 10), (1500, 10), (3000, 8)):
            for kind in ("pooled one-hot", "per-nuclease"):
                arm = f"{kind} n_est={ne} depth={md}"
                if arm in done:
                    continue
                params = dict(XGB, n_estimators=ne, max_depth=md)
                rws = []
                t0 = time.time()
                if kind == "pooled one-hot":
                    m = xgb.XGBRegressor(**params).fit(Xtr, ytr)
                for j, k in enumerate(nd.SHORT):
                    te = idx_te[k]
                    if kind == "pooled one-hot":
                        p = m.predict(np.hstack([X[te], np.tile(D[j], (len(te), 1))]))
                    else:
                        tr = idx_tr[k]
                        p = fit_predict(X[tr], d[k].to_numpy()[tr], X[te], params)
                    rws.append(dict(arm=arm, nuclease=nd.NUCLEASES[j], rho=rho(d[k].to_numpy()[te], p),
                                    rho_ngg=rho(d[k].to_numpy()[te][ngg[te]], p[ngg[te]])))
                append(out, rws)
                print(f"  {arm:40s} mean rho {np.mean([r['rho'] for r in rws]):.4f} "
                      f"NGG {np.mean([r['rho_ngg'] for r in rws]):.4f}  {time.time() - t0:.0f}s", flush=True)
    elif a.stage == 2:
        out = OUT / "nuclease_lono.csv"
        done = set(zip(*[pd.read_csv(out)[c] for c in ("arm", "nuclease")])) if out.exists() else set()
        X = stack(G, base_use)
        idx_tr = {k: rows_for(d, k, "Training") for k in nd.SHORT}
        idx_te = {k: rows_for(d, k, "Test") for k in nd.SHORT}
        rng = np.random.default_rng(0)
        specs = {"U [no nuclease info]": None}
        for g in ("mutations", "structure", "pam", "scale"):
            specs[f"D [{g}]"] = [g]
        specs["D [mutations+structure]"] = ["mutations", "structure"]
        specs["D [all]"] = ["mutations", "structure", "pam", "scale"]
        for hi, (hk, hn) in enumerate(zip(nd.SHORT, nd.NUCLEASES)):
            others = [j for j in range(9) if j != hi]
            te = idx_te[hk]
            y_te = d[hk].to_numpy()[te]
            rows = []
            def score(arm, p, **kw):
                rows.append(dict(arm=arm, nuclease=hn, n_test=len(te), rho=rho(y_te, p),
                                 rho_ngg=rho(y_te[ngg[te]], p[ngg[te]]), **kw))
                print(f"  held out {hn:13s} {arm:28s} rho {rows[-1]['rho']:.4f} "
                      f"(NGG {rows[-1]['rho_ngg']:.4f})", flush=True)
            sub = {}
            for j in others:                       # training rows, subsampled for speed
                r = idx_tr[nd.SHORT[j]]
                sub[j] = rng.choice(r, size=min(len(r), 15000), replace=False)
            ytr = np.concatenate([d[nd.SHORT[j]].to_numpy()[sub[j]] for j in others])
            def fit_with(D, perm=None):
                Dm = D if perm is None else D[perm]
                Xtr = np.vstack([np.hstack([X[sub[j]], np.tile(Dm[j], (len(sub[j]), 1))]) for j in others])
                m = xgb.XGBRegressor(**XGB).fit(Xtr, ytr)
                return m.predict(np.hstack([X[te], np.tile(Dm[hi], (len(te), 1))]))
            for arm, spec in specs.items():
                if (arm, hn) in done:
                    continue
                D = np.zeros((9, 0), np.float32) if spec is None else descriptor_matrix(spec).to_numpy().astype(np.float32)
                score(arm, fit_with(D))
            # null: the same descriptors with the nuclease rows permuted
            if (f"D [all] permuted x{a.perms}", hn) not in done:
                D = descriptor_matrix(["mutations", "structure", "pam", "scale"]).to_numpy().astype(np.float32)
                rs, rn = [], []
                for _ in range(a.perms):
                    perm = rng.permutation(9)
                    while perm[hi] == hi:
                        perm = rng.permutation(9)
                    p = fit_with(D, perm)
                    rs.append(rho(y_te, p)); rn.append(rho(y_te[ngg[te]], p[ngg[te]]))
                rows.append(dict(arm=f"D [all] permuted x{a.perms}", nuclease=hn, n_test=len(te),
                                 rho=float(np.mean(rs)), rho_ngg=float(np.mean(rn)),
                                 rho_sd=float(np.std(rs)), rho_min=float(np.min(rs)),
                                 rho_max=float(np.max(rs))))
                print(f"  held out {hn:13s} D [all] permuted         mean {np.mean(rs):.4f} "
                      f"[{np.min(rs):.4f}, {np.max(rs):.4f}]", flush=True)
            # transfer baselines: a single other nuclease's own model applied to the held-out one
            Xh = X[te]
            best = (-9, None)
            for j in others:
                k = nd.SHORT[j]
                p = fit_predict(X[idx_tr[k]], d[k].to_numpy()[idx_tr[k]], Xh)
                r = rho(y_te, p)
                if k == "SpCas9":
                    score("single-source transfer from SpCas9", p)
                if r > best[0]:
                    best = (r, nd.NUCLEASES[j], p)
            score(f"single-source transfer, best source ({best[1]})", best[2])
            # the unreachable reference: a model trained on the held-out nuclease itself
            tr = idx_tr[hk]
            score("own model (not held out)", fit_predict(X[tr], d[hk].to_numpy()[tr], Xh))
            append(out, rows)


if __name__ == "__main__":
    raise SystemExit(main())
