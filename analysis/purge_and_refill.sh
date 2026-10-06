#!/usr/bin/env bash
# purge_and_refill.sh — run from repo root: bash purge_and_refill.sh
# Removes the two demo-contaminated cells (T1 tabpfn_v2/tabicl seed0, n=4,800),
# reruns them at full scale, refreshes everything downstream, permanently
# prevents smoke/paper key collisions, and packs patch_bundle2.tgz.
set -euo pipefail
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate

echo "== [1/7] Backup + purge contaminated rows and artifacts =="
cp results/raw_results.csv results/raw_results.backup_$(date +%s).csv
python - <<'EOF'
import pandas as pd
df = pd.read_csv("results/raw_results.csv")
n0 = len(df)
bad_bench = ((df.stage=="benchmark") & (df.tier=="T1")
             & (df.model.isin(["tabpfn_v2","tabicl"])) & (df.seed.astype(str)=="0"))
bad_conf = ((df.stage=="conformal") & (df.tier=="T1") & (df.model=="tabicl")
            & (df.seed.astype(str)=="0"))
bad_sens = (df.stage.isin(["recalibration","weighted"]) & (df.tier=="T1")
            & (df.model.isin(["tabpfn_v2","tabicl"])) & (df.seed.astype(str)=="0"))
df = df[~(bad_bench | bad_conf | bad_sens)]
df.to_csv("results/raw_results.csv", index=False)
print(f"purged {n0-len(df)} rows")
EOF
rm -f results/predictions/brfss2022/T1/tabpfn_v2/seed0.parquet
rm -f results/predictions/brfss2022/T1/tabicl/seed0.parquet
rm -rf results/conformal/brfss2022/T1/tabicl/seed0

echo "== [2/7] Make smoke/paper collision impossible (results_dir separation) =="
python - <<'EOF'
s = open("configs/smoke.yaml").read()
if "results_dir" not in s:
    s = s.replace("inherit: default.yaml\n",
                  "inherit: default.yaml\nresults_dir: results_demo   # demo runs can never occupy paper keys\n")
    open("configs/smoke.yaml","w").write(s)
    print("smoke.yaml patched")
else:
    print("already patched")
EOF

echo "== [3/7] Rerun the two cells at FULL scale (GPU; ~10 + ~20 min) =="
python -m experiments.run_benchmark --config configs/default.yaml --tier T1 --model tabpfn_v2 --seed 0
python -m experiments.run_benchmark --config configs/default.yaml --tier T1 --model tabicl    --seed 0
python - <<'EOF'
import pandas as pd
df = pd.read_csv("results/raw_results.csv")
nt = df[(df.stage=="benchmark")&(df.metric=="n_train")&(df.tier=="T1")
        &(df.model.isin(["tabpfn_v2","tabicl"]))&(df.seed.astype(str)=="0")]
assert (nt.value > 200000).all(), "REFILL FAILED - cells still demo-scale, tell Claude"
print("verified: both cells now full-scale (n_train:", nt.value.astype(int).tolist(), ")")
EOF

echo "== [4/7] Refresh downstream: conformal + sensitivities for the 2 cells =="
python -m experiments.run_conformal --config configs/default.yaml --tier T1 --model tabicl --seed 0
python -m experiments.run_recalibration --config configs/default.yaml
python -m experiments.run_weighted --config configs/default.yaml

echo "== [5/7] Retry temporal tabicl seed0 (RAM permitting) =="
FREE=$(free -g | awk '/^Mem:/{print $7}')
if [ "$FREE" -ge 35 ]; then
  python -m experiments.run_temporal --config configs/default.yaml --model tabicl --seed 0 \
    || echo "[warn] tabicl temporal seed0 failed - tell Claude"
else
  echo "[warn] only ${FREE}G free - rerun this script later for this one cell"
fi

echo "== [6/7] Recompute CIs + aggregate =="
python -m analysis.compute_cis
python -m analysis.aggregate_results --config configs/default.yaml
python -m analysis.make_tables       --config configs/default.yaml

echo "== [7/7] Pack patch_bundle2 =="
tar czf patch_bundle2.tgz \
  results/raw_results.csv results/summary.csv results/wilcoxon.csv \
  results/delong.csv results/bootstrap_cis.csv \
  results/conformal/brfss2022/T1/tabicl paper/tables
ls -lh patch_bundle2.tgz
echo "== PURGE+REFILL DONE - upload patch_bundle2.tgz to Claude =="
