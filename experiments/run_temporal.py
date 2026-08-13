"""Temporal external validation (contribution C6).

Applies FROZEN 2022-trained models of the portable tier to the entire 2023
cohort as a pure external test set — no refitting, thresholds carried over
from 2022 validation. Also refits one EBM on 2023 (seed 0) purely for the
shape-replication figure (are the learned risk curves stable across years?).

Usage:
  python -m experiments.run_temporal --config configs/default.yaml \
      --model ebm --seed 0
(requires: 2023 analytic parquet built; benchmark run for tier T1_portable)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.variable_map import OUTCOME_LABEL, SUBGROUP_LABELS, tier_labels
from src.evaluation.fairness import subgroup_audit
from src.evaluation.metrics import all_probability_metrics, confusion_at
from src.models.registry import REGISTRY, make_model
from src.utils.common import append_rows, load_config, seed_everything
from experiments.run_benchmark import load_analytic, model_params


TIER = "T1_portable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--refit-ebm-2023", action="store_true",
                    help="also refit an EBM on 2023 for the shape figure")
    a = ap.parse_args()
    cfg = load_config(a.config)
    seed_everything(a.seed)

    df23 = load_analytic(cfg, "brfss2023")
    feats = tier_labels(TIER, drop_stroke=cfg.get("drop_stroke", False))
    X23, y23 = df23[feats], df23[OUTCOME_LABEL].values.astype(int)
    G23 = df23[SUBGROUP_LABELS]

    adapter_cls = REGISTRY[a.model]
    if getattr(adapter_cls, "requires_gpu", False):
        # FMs are not persisted; refit 2022 context then predict on 2023
        df22 = load_analytic(cfg, "brfss2022")
        from src.utils.common import stratified_splits
        y22 = df22[OUTCOME_LABEL].values.astype(int)
        idx = stratified_splits(y22, a.seed, tuple(cfg["split_fracs"]))
        m = make_model(a.model, model_params(cfg, a.model, TIER), a.seed)
        m.fit(df22[feats].iloc[idx["train"]], y22[idx["train"]])
    else:
        mdir = (Path(cfg["results_dir"]) / "models" / "brfss2022" / TIER /
                a.model / f"seed{a.seed}")
        m = adapter_cls.load(str(mdir))

    p23 = m.predict_proba_pos(X23)

    # thresholds carried from the 2022 run (recorded in raw_results extra)
    res = pd.read_csv(Path(cfg["results_dir"]) / "raw_results.csv")
    thr_rows = res[(res.stage == "benchmark") & (res.dataset == "brfss2022")
                   & (res.tier == TIER) & (res.model == a.model)
                   & (res.seed.astype(str) == str(a.seed))
                   & (res.split == "val")
                   & (res.metric.str.endswith("_sensitivity"))]
    thrs = {}
    for _, r in thr_rows.iterrows():
        name = r["metric"].rsplit("_", 1)[0]
        thrs[name] = float(str(r["extra"]).split("=")[1])

    rows = []
    base = dict(run_id=f"temporal-{TIER}-{a.model}-s{a.seed}",
                stage="temporal", dataset="brfss2023", tier=TIER,
                model=a.model, seed=a.seed, split="external", subgroup="ALL")
    for k, v in all_probability_metrics(y23, p23).items():
        rows.append({**base, "metric": k, "value": v})
    for name, t in thrs.items():
        for k, v in confusion_at(y23, p23, t).items():
            rows.append({**base, "metric": f"{name}_{k}", "value": v,
                         "extra": f"thr={t:.6f}(from2022val)"})
    out_dir = Path(cfg["results_dir"]) / "temporal" / a.model / f"seed{a.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if "screening" in thrs:
        subgroup_audit(y23, p23, G23, thrs["screening"]).to_csv(
            out_dir / "audit_screening_2023.csv", index=False)
    pd.DataFrame({"y": y23, "p": p23}).to_parquet(
        out_dir / "predictions_2023.parquet", index=False)
    append_rows(str(Path(cfg["results_dir"]) / "raw_results.csv"), rows)
    auroc = [r["value"] for r in rows if r["metric"] == "auroc"][0]
    print(f"[temporal] {a.model} seed{a.seed}: 2023 external AUROC="
          f"{auroc:.4f} (n={len(y23):,})")

    if a.refit_ebm_2023 and a.model == "ebm":
        m23 = make_model("ebm", model_params(cfg, "ebm", TIER), a.seed)
        from src.utils.common import stratified_splits
        idx23 = stratified_splits(y23, a.seed, tuple(cfg["split_fracs"]))
        m23.fit(X23.iloc[idx23["train"]], y23[idx23["train"]])
        m23.save(str(Path(cfg["results_dir"]) / "models" / "brfss2023" /
                     TIER / "ebm" / f"seed{a.seed}"))
        print("[temporal] EBM-2023 refit saved for shape-replication figure")


if __name__ == "__main__":
    main()
