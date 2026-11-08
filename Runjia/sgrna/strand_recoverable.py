"""How well can the strand call be read back from each family's columns, after proxy removal?"""
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sgrna import config
from sgrna.cui_screen import _essential_frame
from sgrna.strand_families import _blocks, PROXY

f = _essential_frame()
meta = json.loads((config.INTERIM/"cui_crispri"/"meta.json").read_text())
ess = np.asarray([str(e) for e in meta["essential"]], dtype=object)=="True"
gene = np.asarray([g if isinstance(g,str) else "" for g in meta["gene"]],dtype=object)
hg = (gene!="")&(gene!="nan")&(gene!="None")
protos = np.asarray(meta["protospacers"])[ess&hg]
blocks = _blocks(protos)
cod = f["coding"].ravel()
ok = cod>=0
y = cod[ok].astype(int)
grp = np.asarray(dict(f["groupings"])["genes"])[ok]
print("guides with strand call:", ok.sum(), "coding share", y.mean().round(3))
def auc(X):
    X = X[ok]; p = np.zeros(len(y))
    for tr,te in GroupKFold(5).split(X,y,grp):
        m = xgb.XGBClassifier(n_estimators=200,max_depth=3,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,n_jobs=4,verbosity=0)
        m.fit(X[tr],y[tr]); p[te]=m.predict_proba(X[te])[:,1]
    return roc_auc_score(y,p)
rows=[]
for fam,(v,fn) in blocks.items():
    r = np.nan_to_num(np.array([abs(spearmanr(v[ok,j],cod[ok],nan_policy="omit").statistic) if np.nanstd(v[ok,j])>0 else 0 for j in range(v.shape[1])]))
    keep = r<=PROXY
    a_all = auc(v); a_keep = auc(v[:,keep]) if keep.any() else np.nan
    rows.append(dict(family=fam,n_cols=v.shape[1],n_proxy=int((~keep).sum()),auc_all=a_all,auc_after_removal=a_keep))
    print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(config.RESULTS/"strand_families_recoverable.csv",index=False)
# g_transcription column rhos
v,fn = blocks["g_transcription"]
r = [(abs(spearmanr(v[ok,j],cod[ok],nan_policy="omit").statistic),fn[j]) for j in range(v.shape[1])]
for a,b in sorted(r,reverse=True)[:8]: print(f"  {a:.3f} {b}")
