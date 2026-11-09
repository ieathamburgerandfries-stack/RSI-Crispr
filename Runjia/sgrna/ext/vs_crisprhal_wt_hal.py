"""crisprHAL 2's shipped WT-SpCas9 weights on their own WT test file, saved for a paired comparison (TensorFlow environment).

The WT model takes a 378-nt context (189 nt upstream, 20-nt protospacer, 3-nt PAM, 166 nt downstream); the protospacer starts at
offset 189. Guides scored more than once are averaged by protospacer, as in `final_eval.py` for eSpCas9.

    .venv-crisprhal/bin/python sgrna/ext/vs_crisprhal_wt_hal.py <project root>
"""
import os
import sys

import numpy as np
import pandas as pd

root = sys.argv[1]
HAL = os.path.join(root, "external_data", "crisprHAL")
d = pd.read_csv(os.path.join(HAL, "data", "WT-SpCas9_testing_data.csv"), header=None, names=["context", "score"]).dropna(subset=["context"])
d["context"] = d["context"].str.upper()
assert (d["context"].str.len() == 378).all(), d["context"].str.len().value_counts()
d["protospacer"] = d["context"].str[189:209]
assert d["context"].str[210:212].eq("GG").mean() > 0.99, "protospacer offset is wrong: PAM is not NGG"
d = d[~d["protospacer"].str.contains("N")]
d = d.groupby("protospacer", as_index=False).agg(context=("context", "first"), score=("score", "mean"))
sys.path.insert(0, HAL)
os.chdir(HAL)
import keras
from processing import processing
model = keras.models.load_model("models/WT-SPCAS9.keras", compile=False)
X = processing().onehotencode(d["context"].to_numpy())
pred = np.asarray(model.predict(X, verbose=0)).ravel()
assert np.isfinite(pred).all() and pred.std() > 0
out = os.path.join(root, "results", "preds_final_wt_crisprhal.csv")
pd.DataFrame({"protospacer": d["protospacer"].to_numpy(), "y": d["score"].to_numpy(float), "pred": pred}).to_csv(out, index=False)
from scipy.stats import spearmanr
print(len(d), "guides; Spearman of their shipped WT weights on their WT test file:", round(float(spearmanr(d["score"], pred).statistic), 4))
