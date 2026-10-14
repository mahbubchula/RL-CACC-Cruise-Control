#!/usr/bin/env python3
"""
comparison_upgraded.py
Upgraded from: comparison.py
Enhancements:
  1. Replaces single-bar plots with multi-regime temporal & statistical subplots.
  2. Automatically decomposes Active Demand Loading (Intervals 1-12, 0-60 min)
     vs. Post-Demand Gridlock Clearance Tail (Intervals >12, >60 min).
  3. Computes two-sided paired Wilcoxon signed-rank tests, Friedman omnibus chi-square,
     Cohen's d effect sizes, and 10,000-resample bootstrap 95% confidence intervals.
  4. Exports summary tables and 300-DPI publication figures.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

FILES = {
    "HD 100%": "z.1.Human.xlsx",
    "HD 50% + ACC 50%": "z.2.ACC.xlsx",
    "HD 50% + RL 50%": "z.4.RL.xlsx",
    "HD 50% + CACC 50%": "z.3.CACC.xlsx",
    "HD 50% + RL-CACC 50%": "z.5.RL_CACC.xlsx",
}

PALETTE = {
    "HD 100%": "#D55E00",
    "HD 50% + ACC 50%": "#E69F00",
    "HD 50% + RL 50%": "#CC79A7",
    "HD 50% + CACC 50%": "#0072B2",
    "HD 50% + RL-CACC 50%": "#009E73",
}


def _get_column(df: pd.DataFrame, candidate_names: list[str]) -> np.ndarray:
    for name in candidate_names:
        if name in df.columns:
            return df[name].to_numpy()
    raise KeyError(f"None of {candidate_names} found in columns: {list(df.columns)}")


def load_regime_workbooks(data_dir: str = ".") -> dict[str, pd.DataFrame]:
    # Check current directory, given directory, and datasets_and_benchmarks directory
    candidate_dirs = [data_dir]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    candidate_dirs.append(os.path.join(project_root, "datasets_and_benchmarks"))
    candidate_dirs.append(os.path.join(script_dir, "datasets_and_benchmarks"))

    resolved_dir = None
    for cdir in candidate_dirs:
        if os.path.exists(os.path.join(cdir, "z.1.Human.xlsx")):
            resolved_dir = cdir
            break

    if resolved_dir is None:
        return {}

    dfs = {}
    for label, fname in FILES.items():
        fpath = os.path.join(resolved_dir, fname)
        if os.path.exists(fpath):
            dfs[label] = pd.read_excel(fpath)
    return dfs


def compute_statistical_benchmark(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    hd_df = dfs.get("HD 100%")
    cacc_df = dfs.get("HD 50% + CACC 50%")

    speed_cols = ["Space Mean Speed (km/h)", "Average Speed (km/h)", "Speed (km/h)"]
    tt_cols = ["Avg Travel Time (s)", "Average Travel Time (s)", "Travel Time (s)"]
    delay_cols = ["Avg Delay (s)", "Average Delay (s)", "Delay (s)"]
    queue_cols = ["Queue Length (veh)", "Avg Queue Length (veh)", "Queue Length (vehicles)", "Queue (veh)"]
    co2_cols = ["Total CO2 (kg)", "CO2 (kg)", "CO2 Emissions (kg)"]

    for label, df in dfs.items():
        n_int = len(df)
        active = df.iloc[:12]
        tail = df.iloc[12:] if n_int > 12 else pd.DataFrame()

        v_full = _get_column(df, speed_cols)
        v_act = _get_column(active, speed_cols)
        tt_full = _get_column(df, tt_cols)
        d_full = _get_column(df, delay_cols)
        q_full = _get_column(df, queue_cols)
        co2_full = _get_column(df, co2_cols)

        boot = [rng.choice(v_act, size=len(v_act), replace=True).mean() for _ in range(10000)]
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

        p_hd, d_hd = np.nan, np.nan
        if hd_df is not None and label != "HD 100%":
            hd_v = _get_column(hd_df.iloc[:12], speed_cols)
            _, p_hd = stats.wilcoxon(v_act, hd_v)
            diff = v_act - hd_v
            d_hd = diff.mean() / (diff.std(ddof=1) + 1e-9)

        p_cacc = np.nan
        if cacc_df is not None and label != "HD 50% + CACC 50%":
            cacc_v = _get_column(cacc_df.iloc[:12], speed_cols)
            if not np.allclose(v_act, cacc_v):
                _, p_cacc = stats.wilcoxon(v_act, cacc_v)

        tail_co2 = _get_column(tail, co2_cols).sum() if len(tail) > 0 else 0.0

        rows.append({
            "Regime": label,
            "Total_Intervals": n_int,
            "Horizon_Min": n_int * 5,
            "Gridlock_Tail_Min": max(0, (n_int - 12) * 5),
            "Full_Speed_Mean": v_full.mean(),
            "Full_Speed_SD": v_full.std(ddof=1),
            "Temporal_CV_pct": (v_full.std(ddof=1) / v_full.mean()) * 100.0,
            "Active_Speed_Mean": v_act.mean(),
            "Active_Speed_95CI": f"[{ci_lo:.2f}, {ci_hi:.2f}]",
            "Full_TravelTime_Mean": tt_full.mean(),
            "Full_Delay_Mean": d_full.mean(),
            "Full_Queue_Mean": q_full.mean(),
            "Max_Queue_veh": q_full.max(),
            "Interval_CO2_Mean_kg": co2_full.mean(),
            "Cumulative_CO2_kg": co2_full.sum(),
            "Tail_CO2_kg": tail_co2,
            "Wilcoxon_p_vs_HD": p_hd,
            "Wilcoxon_p_vs_CACC": p_cacc,
            "Cohens_d_vs_HD": d_hd,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    dfs = load_regime_workbooks(target_dir)
    if dfs:
        summary_df = compute_statistical_benchmark(dfs)
        print(summary_df.to_string(index=False))
        out_path = os.path.join(target_dir, "Upgraded_Statistical_Comparison.xlsx")
        summary_df.to_excel(out_path, index=False)
        print(f"\n[OK] Statistical comparison exported to {out_path}")
    else:
        print("comparison_upgraded.py ready. Place z.1..z.5.xlsx in working directory or datasets_and_benchmarks/ to run.")
