"""Beyond rank correlation: the other things a guide-efficiency model is judged on, for SLICER and crisprHAL 2 on crisprHAL 2's own test files.

A guide-design tool is used to pick guides, so the question is not only how well the whole ranking correlates with the label but how well the top is
identified, how well duds are avoided, and whether the scores mean the same thing on a new batch. On each of crisprHAL 2's shipped test splits
(eSpCas9: 11,898 guides; SpCas9: 6,737), with SLICER trained on the matching training file and crisprHAL 2 as shipped, per model:

  rank / linear   Spearman rho, Pearson r, Kendall tau, R2 (squared error on the label's own scale)
  top of the list precision and recall of the true top 10% among the predicted top 10%, precision of the top 1%, NDCG over the top 10% (gain = label
                  percentile), and the 'pick percentile': the mean label percentile of the k guides a user would take (k = 10, 50, 100)
  classification  AUROC and AUPRC for 'efficient' (label in the top quartile) and AUROC for 'dud' (label in the bottom decile)
  calibration     the slope of the label on the prediction (1 = the score is in label units); a rank metric does not see it

Differences carry paired 95% bootstrap intervals over test guides (1,000 draws, one resample used for every metric). SLICER: the default model (champion, 300
columns, 400 trees, nothing searched) and the tuned model (d099x).

    python -m sgrna.compare.metrics_panel
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score

from .. import config
from ..common import OUT
from .vs_crisprhal import SRC, member_specs, train_predict

RESULT = OUT / "metrics_panel.csv"


def _pct(y):
    return 100.0 * (rankdata(y) - 0.5) / len(y)


def metrics(y, p):
    n = len(y)
    pc = _pct(y)
    top = int(round(0.10 * n))
    order = np.argsort(-p)
    true_top = set(np.argsort(-y)[:top])
    pred_top = set(order[:top])
    hit = len(true_top & pred_top)
    k1 = max(int(round(0.01 * n)), 1)
    top1 = len(set(np.argsort(-y)[:k1]) & set(order[:k1])) / k1
    disc = 1.0 / np.log2(np.arange(2, top + 2))
    ideal = np.sort(pc)[::-1][:top]
    ndcg = float((pc[order[:top]] * disc).sum() / (ideal * disc).sum())
    out = {
        "spearman": float(spearmanr(y, p).statistic), "pearson": float(pearsonr(y, p).statistic), "kendall": float(kendalltau(y, p).statistic),
        "r2": float(r2_score(y, p)), "precision_top10": hit / top, "recall_top10": hit / top, "precision_top1": top1, "ndcg_top10": ndcg,
        "pick10_pctile": float(pc[order[:10]].mean()), "pick50_pctile": float(pc[order[:50]].mean()), "pick100_pctile": float(pc[order[:100]].mean()),
        "auroc_efficient": float(roc_auc_score(y >= np.quantile(y, 0.75), p)), "auprc_efficient": float(average_precision_score(y >= np.quantile(y, 0.75), p)),
        "auroc_dud": float(roc_auc_score(y <= np.quantile(y, 0.10), -p)),
        "calibration_slope": float(np.polyfit(p, y, 1)[0]),
    }
    return out


def _split(cond):
    """(y, {SLICER arm: prediction}, crisprHAL 2 prediction) on the guides both models scored."""
    specs = member_specs(["champion", "d099x"])
    if cond == "eSpCas9":
        from .final_eval import CACHE, _split_frames
        tr, te, _ = _split_frames()
        X = np.asarray(np.load(CACHE / "X.npy", mmap_mode="r"), np.float32)
        meta = json.loads((CACHE / "meta.json").read_text())
        protos = pd.Index(meta["protospacers"])
        y_all = np.asarray(meta["y"], float)
        i_tr, i_te = np.unique(protos.get_indexer(tr["protospacer"])), np.unique(protos.get_indexer(te["protospacer"]))
        i_tr = np.setdiff1d(i_tr, np.intersect1d(i_tr, i_te))
        preds, _ = train_predict(specs, X[i_tr], y_all[i_tr], X[i_te])
        hal = pd.read_csv(SRC / "preds_final_espcas9_crisprhal.csv").set_index("protospacer")
        order = pd.Index(protos[i_te])
        both = hal.loc[hal.index.intersection(order)]
        pos = order.get_indexer(both.index)
        return y_all[i_te][pos], {k: v[pos] for k, v in preds.items()}, both["pred"].to_numpy(), len(i_tr)
    from sgrna.cohorts.data_ablation import load
    X, y, names, protos, in_test, is_noshay, nos, n_base = load(verbose=False)
    te, tr = np.where(in_test)[0], np.where(~in_test)[0]
    preds, _ = train_predict(specs, X[tr], y[tr], X[te])
    hal = pd.read_csv(SRC / "preds_final_wt_crisprhal.csv").set_index("protospacer")
    order = pd.Index(np.asarray(protos)[te])
    keep = order.isin(hal.index) & ~order.duplicated()
    pos = np.where(keep)[0]
    return y[te][pos], {k: v[pos] for k, v in preds.items()}, hal.loc[order[pos], "pred"].to_numpy(), len(tr)


def main() -> int:
    rows = []
    for cond in ("SpCas9", "eSpCas9"):
        y, sl, ph, ntr = _split(cond)
        models = {"SLICER default": sl["champion"], "SLICER tuned": sl["d099x"], "crisprHAL 2": ph}
        point = {m: metrics(y, p) for m, p in models.items()}
        rng = np.random.default_rng(0)
        boot = {m: [] for m in models}
        for _ in range(1000):
            k = rng.integers(0, len(y), len(y))
            for m, p in models.items():
                boot[m].append(metrics(y[k], p[k]))
        for m in models:
            b = pd.DataFrame(boot[m])
            for name, v in point[m].items():
                lo, hi = np.percentile(b[name], [2.5, 97.5])
                row = dict(split=cond, model=m, metric=name, value=v, lo=float(lo), hi=float(hi), n_test=len(y), n_train=ntr)
                if m != "crisprHAL 2":
                    d = np.array(pd.DataFrame(boot[m])[name]) - np.array(pd.DataFrame(boot["crisprHAL 2"])[name])
                    row.update(diff_vs_crisprhal=float(v - point["crisprHAL 2"][name]), diff_lo=float(np.percentile(d, 2.5)), diff_hi=float(np.percentile(d, 97.5)))
                rows.append(row)
        print(f"  {cond}: n_test {len(y):,}, n_train {ntr:,}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(RESULT, index=False)
    w = d[d.metric.isin(["spearman", "pearson", "precision_top10", "ndcg_top10", "pick100_pctile", "auroc_efficient", "auroc_dud", "calibration_slope"])]
    print(w.pivot_table(index=["split", "metric"], columns="model", values="value").round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
