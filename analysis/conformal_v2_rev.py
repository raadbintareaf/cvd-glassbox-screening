import numpy as np, pandas as pd
from pathlib import Path
from src.utils.common import load_config, stratified_splits, seed_everything
from experiments.run_benchmark import load_analytic
from src.data.variable_map import OUTCOME_LABEL

ALPHA = 0.1
cfg = load_config("configs/default.yaml")
df = load_analytic(cfg, "brfss2022")
df["AgeBand"] = pd.cut(df["AgeCategory"], [0,4.5,8.5,99],
                       labels=["18-39","40-59","60+"]).astype(str)
y_all = df[OUTCOME_LABEL].values.astype(int)
rows = []
for model in ["ebm","xgboost","tabicl"]:
    for seed in ([0,1,2,3,4] if model=="ebm" else [0]):
        pq = Path(f"results/predictions/brfss2022/T1/{model}/seed{seed}.parquet")
        if not pq.exists(): continue
        d = pd.read_parquet(pq)
        val = d[d.split=="val"]; tst = d[d.split=="test"]
        Gv = {c: df[c].astype(str).values[val.row_id.values]
              for c in ("Sex","AgeBand")}
        Gt = {c: df[c].astype(str).values[tst.row_id.values]
              for c in ("Sex","AgeBand")}
        yv, pv = val.y.values, val.p.values
        yt, pt = tst.y.values, tst.p.values
        def qhat(scores, alpha=ALPHA):
            n = len(scores)
            k = int(np.ceil((n+1)*(1-alpha)))
            return np.sort(scores)[min(k,n)-1]
        def sets_from(pcal_mask_q, pt):
            s0, s1 = pt, 1-pt          # score(y)=1-p_y
            in0 = s0 <= pcal_mask_q(0); in1 = s1 <= pcal_mask_q(1)
            return in0, in1
        def eval_sets(in0, in1, yt, gt, method, gvname, extra=""):
            for g in np.unique(gt):
                m = gt==g
                cov = np.where(yt[m]==1, in1[m], in0[m]).mean()
                size = (in0[m].astype(int)+in1[m].astype(int))
                empty = int((size==0).sum())
                singles = size==1
                sing_err = float((~np.where(yt[m]==1,in1[m],in0[m]))[singles].mean()) \
                           if singles.any() else np.nan
                rows.append(dict(model=model, seed=seed, method=method+extra,
                    group_var=gvname, group=g, coverage=round(cov,4),
                    avg_set_size=round(size.mean(),4),
                    deferral=round((size==2).mean(),4),
                    empty_sets=empty, singleton_error=round(sing_err,4)
                    if sing_err==sing_err else np.nan, n=int(m.sum())))
        sv = np.where(yv==1, 1-pv, pv)                 # true-label scores
        for half,tag in [(slice(None),""),
                         (np.arange(len(yv))%2==0,"_halfval")]:
            svh = sv[half] if not isinstance(half,slice) else sv
            q_marg = qhat(svh)
            in0 = pv[:0]  # noop
            i0 = (pt) >= 1-q_marg*0  # placeholder
            in0m = (pt <= q_marg); in1m = ((1-pt) <= q_marg)
            for gvname in ("Sex","AgeBand"):
                eval_sets(in0m, in1m, yt, Gt[gvname], "marginal", gvname, tag)
            # mondrian sex x age
            key_v = np.char.add(Gv["Sex"], Gv["AgeBand"])
            key_t = np.char.add(Gt["Sex"], Gt["AgeBand"])
            kv = key_v[half] if not isinstance(half,slice) else key_v
            qg = {k: qhat(svh[kv==k]) for k in np.unique(kv)}
            qq = np.vectorize(lambda k: qg.get(k, q_marg))(key_t)
            in0g = pt <= qq; in1g = (1-pt) <= qq
            for gvname in ("Sex","AgeBand"):
                eval_sets(in0g, in1g, yt, Gt[gvname],
                          "mondrian_sex_age", gvname, tag)
            # carve-out: 18-39 uses marginal
            carve = Gt["AgeBand"]=="18-39"
            in0c = np.where(carve, in0m, in0g)
            in1c = np.where(carve, in1m, in1g)
            for gvname in ("Sex","AgeBand"):
                eval_sets(in0c, in1c, yt, Gt[gvname],
                          "mondrian_carveout1839", gvname, tag)
pd.DataFrame(rows).to_csv("results/revision/conformal_v2.csv", index=False)
e = pd.DataFrame(rows)
print(e[(e.model=="ebm")&(e.seed==0)&(e.group_var=="AgeBand")]
      .to_string(index=False))
print("total empty sets anywhere:", int(e.empty_sets.sum()))
