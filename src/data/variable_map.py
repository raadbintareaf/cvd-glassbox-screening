"""Variable map for CDC BRFSS LLCP 2022 (development) and 2023 (temporal validation).

Every raw variable name below was verified to exist in the corresponding
LLCP{year}.XPT file (checked programmatically against the downloaded CDC files
on 2026-08-09; see README 'Data provenance'). Recode rules follow the CDC BRFSS
2022/2023 codebooks: 7 = Don't know/Not sure, 9 = Refused (77/99 for 2-digit
items) are set to NaN; 88 = "None" maps to 0 for day-count items.

Feature tiers implement the leakage-severity gradient (paper contribution C1):

  T0_full    : all analytic predictors, matching the feature scope of the
               prior literature on this data family (incl. the 2026 draft).
  T1_screen  : T0 minus DIRECT post-diagnostic markers of coronary disease —
               angina/CHD diagnosis (CVDCRHD4) and chest CT (LCSCTSC1).
               Rationale: both are consequences/correlates of the diagnostic
               process for the outcome disease itself (Davis et al., JAMIA
               2023 leakage taxonomy: "outcome-adjacent diagnostic signals").
  T2_minimal : self-knowable screening set — demographics, lifestyle,
               anthropometrics, self-rated health; no clinician-mediated
               variables (no diagnoses, no vaccinations/tests, no disability
               items that may follow a cardiac event).
  T1_portable: T1 restricted to variables present in BOTH 2022 and 2023 cores
               (drops SleepHours, RemovedTeeth, CovidPos), enabling strict
               temporal external validation without imputation.

An ablation flag (configs) optionally also removes CVDSTRK3 (stroke) from T1,
since stroke may share post-event care pathways with MI.
"""
from __future__ import annotations

# ----------------------------------------------------------------------------
# Recode helpers (value-level rules referenced by the spec table)
# ----------------------------------------------------------------------------
YES_NO = {"kind": "binary", "map": {1: 1, 2: 0}, "na": [7, 9]}
YES_NO_2DIGIT_NA = {"kind": "binary", "map": {1: 1, 2: 0}, "na": [77, 99]}

SPEC_COMMON = {
    # --- identifiers / design (never predictors) --------------------------
    "_LLCPWT": {"role": "design", "kind": "numeric", "na": []},
    "_STSTR":  {"role": "design", "kind": "numeric", "na": []},
    "_PSU":    {"role": "design", "kind": "numeric", "na": []},

    # --- outcome ----------------------------------------------------------
    "CVDINFR4": {"role": "outcome", **YES_NO, "label": "HadHeartAttack"},
    # secondary composite component:
    "CVDCRHD4": {"role": "predictor", **YES_NO, "label": "HadAngina",
                 "tiers": ["T0"], "leakage": "direct_post_diagnostic",
                 "family": "clinical_history"},

    # --- demographics -----------------------------------------------------
    "_STATE":   {"role": "predictor", "kind": "categorical", "na": [],
                 "label": "State", "tiers": ["T0", "T1", "T2"],
                 "family": "demographics"},
    "SEXVAR":   {"role": "predictor", "kind": "categorical", "na": [],
                 "map": {1: "Male", 2: "Female"}, "label": "Sex",
                 "tiers": ["T0", "T1", "T2"], "family": "demographics"},
    "_AGEG5YR": {"role": "predictor", "kind": "ordinal", "na": [14],
                 "label": "AgeCategory", "tiers": ["T0", "T1", "T2"],
                 "family": "demographics"},
    "_IMPRACE": {"role": "predictor", "kind": "categorical", "na": [],
                 "map": {1: "White NH", 2: "Black NH", 3: "Asian NH",
                         4: "AIAN NH", 5: "Hispanic", 6: "Other NH"},
                 "label": "RaceEthnicity", "tiers": ["T0", "T1", "T2"],
                 "family": "demographics"},

    # --- general health & lifestyle --------------------------------------
    "GENHLTH":  {"role": "predictor", "kind": "ordinal", "na": [7, 9],
                 "label": "GeneralHealth", "tiers": ["T0", "T1", "T2"],
                 "family": "general_health"},
    "PHYSHLTH": {"role": "predictor", "kind": "numeric", "na": [77, 99],
                 "map": {88: 0}, "label": "PhysicalHealthDays",
                 "tiers": ["T0", "T1", "T2"], "family": "general_health"},
    "MENTHLTH": {"role": "predictor", "kind": "numeric", "na": [77, 99],
                 "map": {88: 0}, "label": "MentalHealthDays",
                 "tiers": ["T0", "T1", "T2"], "family": "general_health"},
    "CHECKUP1": {"role": "predictor", "kind": "ordinal", "na": [7, 9],
                 "map": {8: 5}, "label": "LastCheckupTime",
                 "tiers": ["T0", "T1"], "family": "care_engagement"},
    "EXERANY2": {"role": "predictor", **YES_NO, "label": "PhysicalActivities",
                 "tiers": ["T0", "T1", "T2"], "family": "lifestyle"},
    "_SMOKER3": {"role": "predictor", "kind": "categorical", "na": [9],
                 "map": {1: "Current daily", 2: "Current someday",
                         3: "Former", 4: "Never"},
                 "label": "SmokerStatus", "tiers": ["T0", "T1", "T2"],
                 "family": "lifestyle"},
    "ECIGNOW2": {"role": "predictor", "kind": "categorical", "na": [7, 9],
                 "map": {1: "Never", 2: "Current daily",
                         3: "Current someday", 4: "Former"},
                 "label": "ECigaretteUsage", "tiers": ["T0", "T1", "T2"],
                 "family": "lifestyle"},
    "DRNKANY6": {"role": "predictor", **YES_NO, "label": "AlcoholDrinkers",
                 "tiers": ["T0", "T1", "T2"], "family": "lifestyle"},

    # --- anthropometrics (CDC-derived, 2 implied decimals) ----------------
    "HTM4":  {"role": "predictor", "kind": "numeric", "na": [], "scale": 0.01,
              "label": "HeightMeters", "tiers": ["T0", "T1", "T2"],
              "family": "anthropometrics"},
    "WTKG3": {"role": "predictor", "kind": "numeric", "na": [99999],
              "scale": 0.01, "label": "WeightKilograms",
              "tiers": ["T0", "T1", "T2"], "family": "anthropometrics"},
    "_BMI5": {"role": "predictor", "kind": "numeric", "na": [], "scale": 0.01,
              "label": "BMI", "tiers": ["T0", "T1", "T2"],
              "family": "anthropometrics"},

    # --- clinical history (self-reported diagnoses) -----------------------
    "CVDSTRK3": {"role": "predictor", **YES_NO, "label": "HadStroke",
                 "tiers": ["T0", "T1"], "family": "clinical_history",
                 "leakage": "ablation_candidate"},
    "ASTHMA3":  {"role": "predictor", **YES_NO, "label": "HadAsthma",
                 "tiers": ["T0", "T1"], "family": "clinical_history"},
    "CHCSCNC1": {"role": "predictor", **YES_NO, "label": "HadSkinCancer",
                 "tiers": ["T0", "T1"], "family": "clinical_history"},
    "CHCOCNC1": {"role": "predictor", **YES_NO, "label": "HadOtherCancer",
                 "tiers": ["T0", "T1"], "family": "clinical_history"},
    "CHCCOPD3": {"role": "predictor", **YES_NO, "label": "HadCOPD",
                 "tiers": ["T0", "T1"], "family": "clinical_history"},
    "ADDEPEV3": {"role": "predictor", **YES_NO, "label": "HadDepression",
                 "tiers": ["T0", "T1"], "family": "clinical_history"},
    "CHCKDNY2": {"role": "predictor", **YES_NO, "label": "HadKidneyDisease",
                 "tiers": ["T0", "T1"], "family": "clinical_history"},
    "HAVARTH4": {"role": "predictor", **YES_NO, "label": "HadArthritis",
                 "tiers": ["T0", "T1"], "family": "clinical_history"},
    "DIABETE4": {"role": "predictor", "kind": "categorical", "na": [7, 9],
                 "map": {1: "Yes", 2: "Gestational only", 3: "No",
                         4: "Prediabetes"},
                 "label": "DiabetesStatus", "tiers": ["T0", "T1"],
                 "family": "clinical_history"},

    # --- functional / sensory --------------------------------------------
    "DEAF":     {"role": "predictor", **YES_NO, "label": "DeafHardHearing",
                 "tiers": ["T0", "T1"], "family": "functional"},
    "BLIND":    {"role": "predictor", **YES_NO, "label": "BlindVisionDiff",
                 "tiers": ["T0", "T1"], "family": "functional"},
    "DECIDE":   {"role": "predictor", **YES_NO, "label": "DiffConcentrating",
                 "tiers": ["T0", "T1"], "family": "functional"},
    "DIFFWALK": {"role": "predictor", **YES_NO, "label": "DiffWalking",
                 "tiers": ["T0", "T1"], "family": "functional"},
    "DIFFDRES": {"role": "predictor", **YES_NO, "label": "DiffDressing",
                 "tiers": ["T0", "T1"], "family": "functional"},
    "DIFFALON": {"role": "predictor", **YES_NO, "label": "DiffErrands",
                 "tiers": ["T0", "T1"], "family": "functional"},

    # --- preventive care / testing ---------------------------------------
    "LCSCTSC1": {"role": "predictor", **YES_NO, "label": "ChestScan",
                 "tiers": ["T0"], "leakage": "direct_post_diagnostic",
                 "family": "care_engagement"},
    "HIVTST7":  {"role": "predictor", **YES_NO, "label": "HIVTesting",
                 "tiers": ["T0", "T1"], "family": "care_engagement"},
    "FLUSHOT7": {"role": "predictor", **YES_NO, "label": "FluVaxLast12",
                 "tiers": ["T0", "T1"], "family": "care_engagement"},
    "PNEUVAC4": {"role": "predictor", **YES_NO, "label": "PneumoVaxEver",
                 "tiers": ["T0", "T1"], "family": "care_engagement"},
    "TETANUS1": {"role": "predictor", "kind": "categorical", "na": [7, 9],
                 "map": {1: "Yes Tdap", 2: "Yes not Tdap", 3: "Yes unknown",
                         4: "No"},
                 "label": "TetanusLast10", "tiers": ["T0", "T1"],
                 "family": "care_engagement"},
}

# Year-specific items --------------------------------------------------------
SPEC_2022_ONLY = {
    "SLEPTIM1": {"role": "predictor", "kind": "numeric", "na": [77, 99],
                 "label": "SleepHours", "tiers": ["T0", "T1", "T2"],
                 "family": "lifestyle"},
    "RMVTETH4": {"role": "predictor", "kind": "ordinal", "na": [7, 9],
                 "map": {8: 0, 1: 1, 2: 2, 3: 3},  # None<1-5<6+<All
                 "label": "RemovedTeeth", "tiers": ["T0", "T1"],
                 "family": "clinical_history"},
    "COVIDPOS": {"role": "predictor", "kind": "categorical", "na": [7, 9],
                 "map": {1: "Yes", 2: "No", 3: "Yes home test"},
                 "label": "CovidPositive", "tiers": ["T0", "T1"],
                 "family": "clinical_history"},
}
SPEC_2023_RENAMES = {  # raw 2023 name -> treat exactly as this 2022 spec entry
    "COVIDPO1": "COVIDPOS",
}

NOT_IN_2023 = {"SLEPTIM1", "RMVTETH4"}  # asked in even-year core only
# Empirically module-only in 2023 (measured missingness in the built cohort:
# TETANUS1 97.3%, LCSCTSC1 94.7% — state-optional modules). LCSCTSC1 is
# already excluded from T1 as leakage; TETANUS1 must leave the portable tier.
MODULE_ONLY_2023 = {"TETANUS1", "LCSCTSC1"}


def spec_for_year(year: int) -> dict:
    spec = dict(SPEC_COMMON)
    if year == 2022:
        spec.update(SPEC_2022_ONLY)
    elif year == 2023:
        for raw23, raw22 in SPEC_2023_RENAMES.items():
            spec[raw23] = dict(SPEC_2022_ONLY[raw22])
    else:
        raise ValueError(f"No variable map for BRFSS year {year}")
    return spec


def tier_labels(tier: str, drop_stroke: bool = False) -> list[str]:
    """Return analytic column labels for a tier (labels, not raw names)."""
    base = {"T0": "T0", "T1": "T1", "T2": "T2", "T1_portable": "T1"}[tier]
    spec = spec_for_year(2022)
    labels = [v["label"] for v in spec.values()
              if v.get("role") == "predictor" and base in v.get("tiers", [])]
    if tier == "T1_portable":
        portable_drop = {SPEC_2022_ONLY[r]["label"] for r in NOT_IN_2023}
        portable_drop |= {SPEC_COMMON[r]["label"] for r in MODULE_ONLY_2023
                          if r in SPEC_COMMON}
        labels = [l for l in labels if l not in portable_drop]
    if drop_stroke and tier != "T0":
        labels = [l for l in labels if l != "HadStroke"]
    return sorted(labels)


OUTCOME_LABEL = "HadHeartAttack"
DESIGN_LABELS = {"_LLCPWT": "SurveyWeight", "_STSTR": "Stratum", "_PSU": "PSU"}
SUBGROUP_LABELS = ["Sex", "AgeCategory", "RaceEthnicity"]

STATE_FIPS = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE",
    11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN",
    19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD", 25: "MA",
    26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE", 32: "NV",
    33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH",
    40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD", 47: "TN",
    48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV", 55: "WI",
    56: "WY", 66: "GU", 72: "PR", 78: "VI",
}


# --- revision tier T1_nostroke (M1): via the existing drop_stroke switch ---
_base_tier_labels_rev = tier_labels
def tier_labels(t, **kw):  # noqa: F811
    if t == "T1_nostroke":
        kw = dict(kw)
        kw["drop_stroke"] = True
        return _base_tier_labels_rev("T1", **kw)
    return _base_tier_labels_rev(t, **kw)
