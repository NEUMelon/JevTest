"""
E4: Calibration Analysis (RQ3)
1. Evaluates reliability diagrams, ECE, Brier Score, and NLL on test set:
   - J-H1n (Noul)
   - J-H1s (Score)
   - J-H1c (Choice)
   - L1-H1 (GPT-6 Luna verbalized prob)
   - L2-H1 (DeepSeek V4 Flash prob)
   - J-D+LR & J-D+C+LR (aggregator outputs)
2. Temperature Scaling on pool2000:
   - Fit T on pool2000 labels by minimizing NLL
   - Compare T direction with Scienthoon baseline (1.30 / 1.92 / 0.66)
   - Evaluate post-scaling ECE on test set
3. Selective Prediction (Risk-Coverage):
   - Coverage and error rate at confidence thresholds 0.90, 0.95, 0.99
4. Outputs:
   - reports/tables/t_calibration.csv
   - reports/tables/t_temperature.csv
   - reports/tables/t_selective_prediction.csv
   - reports/figures/f_reliability_*.png & .pdf
   - reports/figures/f_risk_coverage_all_4panels.png & .pdf
"""

import os, sys, json, sqlite3, yaml
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent.parent
QUESTIONS_DIR  = BASE_DIR / "questions"
POOLS_DIR      = BASE_DIR / "data" / "pools"
CANONICAL_DIR  = BASE_DIR / "data" / "canonical"
RUNS_DIR       = BASE_DIR / "runs"
REPORTS_TABLES = BASE_DIR / "reports" / "tables"
REPORTS_FIGS   = BASE_DIR / "reports" / "figures"

EM_DATASETS = ["wa", "ag", "da", "ab"]
EPS = 1e-4

def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))

def compute_ece(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    total = len(y_true)
    bin_accs, bin_confs, bin_counts = [], [], []

    for b in range(n_bins):
        mask = (bin_indices == b)
        count = np.sum(mask)
        bin_counts.append(count)
        if count > 0:
            acc = np.mean(y_true[mask])
            conf = np.mean(y_prob[mask])
            ece += (count / total) * abs(acc - conf)
            bin_accs.append(acc)
            bin_confs.append(conf)
        else:
            bin_accs.append(np.nan)
            bin_confs.append(np.nan)

    return ece * 100.0, bin_accs, bin_confs, bin_counts

def compute_brier(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    return float(np.mean((y_prob - y_true) ** 2))

def compute_nll(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), EPS, 1.0 - EPS)
    return float(-np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)))

def fit_temperature(y_val, p_val):
    z_val = logit(p_val)
    y_val = np.asarray(y_val, dtype=float)

    def nll_obj(T):
        p_scaled = sigmoid(z_val / T)
        p_scaled = np.clip(p_scaled, EPS, 1.0 - EPS)
        return -np.mean(y_val * np.log(p_scaled) + (1.0 - y_val) * np.log(1.0 - p_scaled))

    res = minimize_scalar(nll_obj, bounds=(0.05, 10.0), method="bounded")
    best_T = float(res.x)
    return best_T

def extract_pool_data(ds):
    from src.clients.systemone import SystemOneClient
    from src.experiments.e1_holistic import extract_noul, extract_score, extract_choice
    
    pairs = [json.loads(line) for line in open(POOLS_DIR / f"{ds}_pool2000.jsonl", encoding="utf-8")]
    with open(QUESTIONS_DIR / ds / "holistic.yaml", encoding="utf-8") as f:
        holistic_q = yaml.safe_load(f)

    client = SystemOneClient(model="jev-latest")
    rows = []
    for p in pairs:
        state = {"record_a": p["record_a"], "record_b": p["record_b"]}
        resp, _ = client.ask(state, holistic_q)
        ans = resp.get("answers", {})
        p_noul = extract_noul(ans.get("H1_noul", 0.5))
        p_score = extract_score(ans.get("H1_score", 0.5))
        p_choice = extract_choice(ans.get("H1_choice", {}))
        conf_choice = ans.get("H1_choice", {}).get("confidence", max(p_choice, 1 - p_choice))
        conf_score = ans.get("H1_score", {}).get("confidence", p_score)
        rows.append({
            "pair_id": p["pair_id"], "label": p["label"],
            "p_h1n": p_noul, "p_h1s": p_score, "p_h1c": p_choice,
            "conf_choice": conf_choice, "conf_score": conf_score
        })
    return pd.DataFrame(rows)

def run_e4():
    print("=" * 70, flush=True)
    print("E4: Calibration Analysis (RQ3)", flush=True)
    print("=" * 70, flush=True)

    REPORTS_TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS_FIGS.mkdir(parents=True, exist_ok=True)

    calib_records = []
    temp_records = []
    sel_records = []

    # Scienthoon reference temperatures
    scienthoon_refs = {
        "H1_noul": 1.30,
        "H1_choice": 1.92,
        "H1_score": 0.66
    }

    # Set plot style
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
    })

    fig_rel, axes_rel = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    axes_rel = axes_rel.flatten()

    fig_sel, axes_sel = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    axes_sel = axes_sel.flatten()

    ds_titles = {
        "wa": "Walmart-Amazon (WA)",
        "ag": "Amazon-Google (AG)",
        "da": "DBLP-ACM (DA)",
        "ab": "Abt-Buy (AB)"
    }

    for idx, ds in enumerate(EM_DATASETS):
        print(f"\nEvaluating calibration for {ds.upper()}...", flush=True)
        # 1. Load test predictions
        e1_test = pd.read_csv(RUNS_DIR / f"e1_preds_{ds}.csv")
        y_test = e1_test["label"].values

        # 2. Load pool2000 predictions for temperature scaling
        df_pool = extract_pool_data(ds)
        y_pool = df_pool["label"].values

        # Systems to evaluate for pre-scaling calibration
        systems = {
            "J-H1n (Noul)":    e1_test["p_h1n"].values,
            "J-H1s (Score)":   e1_test["p_h1s"].values,
            "J-H1c (Choice)":  e1_test["p_h1c"].values,
            "L1-H1 (Luna)":    e1_test["p_luna"].values,
            "L2-H1 (DeepSeek)":e1_test["p_deepseek"].values,
        }

        # Temperature scaling targets
        q_map = {
            "H1_noul":   ("p_h1n", "J-H1n (Noul)"),
            "H1_score":  ("p_h1s", "J-H1s (Score)"),
            "H1_choice": ("p_h1c", "J-H1c (Choice)"),
        }

        temp_scaling_results = {}
        for q_name, (col, sys_label) in q_map.items():
            p_val = df_pool[col].values
            best_T = fit_temperature(y_pool, p_val)
            p_test_orig = e1_test[col].values
            p_test_scaled = sigmoid(logit(p_test_orig) / best_T)

            ece_orig, _, _, _ = compute_ece(y_test, p_test_orig)
            ece_scaled, _, _, _ = compute_ece(y_test, p_test_scaled)
            brier_orig = compute_brier(y_test, p_test_orig)
            brier_scaled = compute_brier(y_test, p_test_scaled)
            nll_orig = compute_nll(y_test, p_test_orig)
            nll_scaled = compute_nll(y_test, p_test_scaled)

            ref_T = scienthoon_refs.get(q_name, np.nan)
            direction = "Overconfident (T > 1)" if best_T > 1.05 else ("Underconfident (T < 1)" if best_T < 0.95 else "Well-calibrated (T ≈ 1)")

            temp_records.append({
                "dataset": ds, "question_type": q_name, "system": sys_label,
                "optimal_T": round(best_T, 3), "scienthoon_ref_T": ref_T,
                "direction": direction,
                "ece_before": round(ece_orig, 2), "ece_after": round(ece_scaled, 2),
                "brier_before": round(brier_orig, 4), "brier_after": round(brier_scaled, 4),
                "nll_before": round(nll_orig, 4), "nll_after": round(nll_scaled, 4),
            })
            temp_scaling_results[sys_label] = (best_T, p_test_scaled, ece_scaled)
            print(f"  [{ds}] {q_name}: T={best_T:.3f} (ref={ref_T}), ECE: {ece_orig:.2f}% -> {ece_scaled:.2f}%", flush=True)

        # Baseline calibration table
        ax_rel = axes_rel[idx]
        ax_rel.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1.5, label="Perfect calibration")

        colors = {
            "J-H1n (Noul)": "#7570b3",
            "J-H1s (Score)": "#e7298a",
            "J-H1c (Choice)": "#d95f02",
            "L1-H1 (Luna)": "#1f78b4",
            "L2-H1 (DeepSeek)": "#e31a1c",
        }

        for sys_name, p_pred in systems.items():
            ece, b_accs, b_confs, _ = compute_ece(y_test, p_pred)
            brier = compute_brier(y_test, p_pred)
            nll = compute_nll(y_test, p_pred)

            calib_records.append({
                "dataset": ds, "system": sys_name,
                "ece": round(ece, 2), "brier": round(brier, 4), "nll": round(nll, 4),
                "post_temp_ece": round(temp_scaling_results[sys_name][2], 2) if sys_name in temp_scaling_results else np.nan
            })

            # Plot non-NaN bins
            valid_x = [c for c in b_confs if not np.isnan(c)]
            valid_y = [a for a in b_accs if not np.isnan(a)]
            if valid_x and valid_y:
                ax_rel.plot(valid_x, valid_y, marker="o", markersize=6, linewidth=1.8,
                            color=colors.get(sys_name, "#333333"), label=f"{sys_name} (ECE={ece:.1f}%)")

        ax_rel.set_xlim([0.0, 1.0])
        ax_rel.set_ylim([0.0, 1.0])
        ax_rel.set_title(ds_titles[ds], fontsize=13, fontweight="bold")
        ax_rel.set_xlabel("Mean Predicted Probability", fontsize=11)
        ax_rel.set_ylabel("Observed Accuracy / Empirical Positive Fraction", fontsize=11)
        ax_rel.grid(True)
        ax_rel.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

        # 3. Selective prediction (Risk-Coverage)
        # Using Jev choice confidence & noul confidence
        conf_noul = np.maximum(e1_test["p_h1n"].values, 1.0 - e1_test["p_h1n"].values)
        pred_noul = (e1_test["p_h1n"].values >= 0.5).astype(int)
        correct_noul = (pred_noul == y_test).astype(int)

        ax_sel = axes_sel[idx]
        coverages, selective_accs = [], []
        thresholds = np.linspace(0.5, 0.99, 50)
        for tau in thresholds:
            mask = (conf_noul >= tau)
            cov = np.mean(mask)
            acc = np.mean(correct_noul[mask]) if np.sum(mask) > 0 else 1.0
            coverages.append(cov * 100.0)
            selective_accs.append(acc * 100.0)

        ax_sel.plot(coverages, selective_accs, color="#7570b3", linewidth=2.2, label="J-H1n Selective Accuracy")
        ax_sel.set_title(f"Risk-Coverage: {ds_titles[ds]}", fontsize=13, fontweight="bold")
        ax_sel.set_xlabel("Coverage (% Samples Predicted)", fontsize=11)
        ax_sel.set_ylabel("Selective Accuracy (%)", fontsize=11)
        ax_sel.grid(True)
        ax_sel.legend(loc="lower left", fontsize=10)

        # Record specific points: 0.90, 0.95, 0.99
        for tau in [0.90, 0.95, 0.99]:
            mask = (conf_noul >= tau)
            cov = float(np.mean(mask) * 100.0)
            acc = float(np.mean(correct_noul[mask]) * 100.0) if np.sum(mask) > 0 else 100.0
            sel_records.append({
                "dataset": ds, "system": "J-H1n", "confidence_threshold": tau,
                "coverage_pct": round(cov, 2), "selective_accuracy_pct": round(acc, 2),
                "error_rate_pct": round(100.0 - acc, 2)
            })

    # Save Reliability Figures
    fig_rel.tight_layout()
    fig_rel.savefig(REPORTS_FIGS / "f_reliability_all_4panels.png")
    fig_rel.savefig(REPORTS_FIGS / "f_reliability_all_4panels.pdf")
    plt.close(fig_rel)

    # Save Selective Prediction Figures
    fig_sel.tight_layout()
    fig_sel.savefig(REPORTS_FIGS / "f_risk_coverage_all_4panels.png")
    fig_sel.savefig(REPORTS_FIGS / "f_risk_coverage_all_4panels.pdf")
    plt.close(fig_sel)

    # Save Tables
    df_calib = pd.DataFrame(calib_records)
    df_temp = pd.DataFrame(temp_records)
    df_sel = pd.DataFrame(sel_records)

    df_calib.to_csv(REPORTS_TABLES / "t_calibration.csv", index=False)
    df_temp.to_csv(REPORTS_TABLES / "t_temperature.csv", index=False)
    df_sel.to_csv(REPORTS_TABLES / "t_selective_prediction.csv", index=False)

    print(f"\n[SAVED] {REPORTS_TABLES / 't_calibration.csv'}")
    print(f"[SAVED] {REPORTS_TABLES / 't_temperature.csv'}")
    print(f"[SAVED] {REPORTS_TABLES / 't_selective_prediction.csv'}")
    print(f"[SAVED] Figures in {REPORTS_FIGS}")
    print("\n" + "=" * 70, flush=True)
    print("E4 CALIBRATION COMPLETE!", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    run_e4()
