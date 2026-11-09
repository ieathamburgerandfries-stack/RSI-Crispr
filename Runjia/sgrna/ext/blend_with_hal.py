"""Is crisprHAL 2's network complementary to the tuned trees? Equal-weight blends on its own splits (no weight is fitted on test data).

If the two models err differently, averaging them beats both; if the trees already capture what the network does, it does not. Uses the
raw test predictions saved by `panel_vs_hal.py`; z-scored average with weight 1/2 each, plus the Spearman of the correlation between
the two models' residuals. Paired bootstrap over guides against the better single model.

    python -m sgrna.ext.blend_with_hal
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import panel_vs_hal as P
from .common import OUT


def _z(p):
    return (p - p.mean()) / p.std()


def run(n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for split in ("espcas9", "wt"):
        z = np.load(P.DIR / f"{split}.npz"); y, hal, tuned = z["y"], z["hal"], z["d099x"]
        boots = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
        rho = lambda p, b=slice(None): float(spearmanr(y[b], p[b]).statistic)
        for w in (0.0, 0.25, 0.5, 0.75, 1.0):
            blend = w * _z(hal) + (1 - w) * _z(tuned)
            r = rho(blend)
            if w == 0.5:
                d = np.array([rho(blend, b) - max(rho(tuned, b), rho(hal, b)) for b in boots])
                lo, hi = np.percentile(d, [2.5, 97.5])
            else:
                lo = hi = np.nan
            rows.append(dict(split=split, weight_on_crisprhal=w, rho=r, vs_best_single=r - max(rho(tuned), rho(hal)) if w == 0.5 else np.nan,
                             lo=lo, hi=hi))
        res_t, res_h = _z(y) - _z(tuned), _z(y) - _z(hal)
        rows.append(dict(split=split, weight_on_crisprhal=np.nan, rho=float(np.corrcoef(res_t, res_h)[0, 1]), vs_best_single=np.nan, lo=np.nan, hi=np.nan))
    out = pd.DataFrame(rows); out.to_csv(OUT / "blend_with_hal.csv", index=False)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(run().round(4).to_string(index=False))
