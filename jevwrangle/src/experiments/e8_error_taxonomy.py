"""
Experiment 8: Error Taxonomy & Slicing Analysis (RQ5)
Categorizes errors across 7 structured slices:
  1. Model/ID Mismatch
  2. Numeric Difference
  3. Variant/Version Mismatch
  4. Accessory/Bundle Difference
  5. Missing Attribute
  6. Label Noise / Ambiguous
  7. Other Semantic

Evaluates: J-H1n, J-D+LR, J-D+C+LR, GPT-6 Luna, DeepSeek V4 Flash.
Generates:
  - reports/tables/t_error_taxonomy.csv
  - reports/figures/f_error_slices.png
  - reports/figures/f_error_slices.pdf
"""

import os
import sys
import json
import re
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

ROOT_DIR = Path("E:/Desktop/jev/jevwrangle")
RUNS_DIR = ROOT_DIR / "runs"
REPORTS_TABLES = ROOT_DIR / "reports" / "tables"
REPORTS_FIGS = ROOT_DIR / "reports" / "figures"
DATA_DIR = ROOT_DIR / "data"

REPORTS_TABLES.mkdir(parents=True, exist_ok=True)
REPORTS_FIGS.mkdir(parents=True, exist_ok=True)

DATASETS = ["wa", "ag", "da", "ab"]
DATASET_NAMES = {
    "wa": "Walmart-Amazon",
    "ag": "Amazon-Google",
    "da": "DBLP-ACM",
    "ab": "Abt-Buy"
}

ACCESSORY_WORDS = {"case", "cover", "charger", "cable", "strap", "battery", "mount", "adapter", "sleeve", "protector", "bundle", "kit", "pack", "dock", "tripod", "bag", "filter"}
VERSION_WORDS = {"pro", "plus", "ultra", "mini", "max", "edition", "v1", "v2", "v3", "black", "white", "silver", "blue", "red", "gold"}

def extract_model_codes(text):
    if not text:
        return set()
    tokens = re.findall(r"\b[A-Za-z0-9]+-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?\b|\b(?=[A-Za-z]*[0-9])(?=[0-9]*[A-Za-z])[A-Za-z0-9]{4,15}\b", text)
    return {t.lower() for t in tokens}

def extract_numeric_specs(text):
    if not text:
        return set()
    return set(re.findall(r"\b\d+(?:\.\d+)?\s*(?:gb|mb|tb|mhz|ghz|inch|\"|mm|mp|w|v|mah)\b", text.lower()))

def get_tokens(text):
    return set(re.findall(r"\w+", (text or "").lower()))

def jaccard(s1, s2):
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

def classify_pair_taxonomy(ds, a, b, label):
    text_a = " ".join(str(v) for v in a.values() if v)
    text_b = " ".join(str(v) for v in b.values() if v)
    toks_a = get_tokens(text_a)
    toks_b = get_tokens(text_b)
    jac = jaccard(toks_a, toks_b)
    
    # 1. Ambiguous / Benchmark Label Noise
    if (jac >= 0.88 and label == 0) or (jac <= 0.15 and label == 1):
        return "Label Noise / Ambiguous"

    # 2. Missing Attribute
    if ds == "wa":
        m_a = str(a.get("modelno", "")).strip().lower()
        m_b = str(b.get("modelno", "")).strip().lower()
        if not m_a or not m_b or m_a in ["0.0", "nan", "none"] or m_b in ["0.0", "nan", "none"]:
            return "Missing Attribute"
    elif ds == "da":
        y_a = str(a.get("year", "")).strip()
        y_b = str(b.get("year", "")).strip()
        if not y_a or not y_b or y_a == "nan" or y_b == "nan":
            return "Missing Attribute"
    elif ds == "ag":
        m_a = str(a.get("manufacturer", "")).strip().lower()
        m_b = str(b.get("manufacturer", "")).strip().lower()
        if not m_a or not m_b or m_a == "nan":
            return "Missing Attribute"

    # 3. Model / ID Mismatch
    if ds == "wa":
        m_a = re.sub(r"[^a-z0-9]", "", str(a.get("modelno", "")).lower())
        m_b = re.sub(r"[^a-z0-9]", "", str(b.get("modelno", "")).lower())
        if m_a and m_b and m_a != m_b:
            return "Model/ID Mismatch"
    codes_a = extract_model_codes(text_a)
    codes_b = extract_model_codes(text_b)
    if codes_a and codes_b and not (codes_a & codes_b):
        return "Model/ID Mismatch"

    # 4. Numeric Difference
    if ds == "da":
        y_a = str(a.get("year", "")).strip()
        y_b = str(b.get("year", "")).strip()
        if y_a and y_b and y_a != y_b:
            return "Numeric Difference"
    else:
        try:
            p_a = float(a.get("price", 0) or 0)
            p_b = float(b.get("price", 0) or 0)
            if p_a > 1.0 and p_b > 1.0:
                ratio = max(p_a, p_b) / min(p_a, p_b)
                if ratio > 1.25:
                    return "Numeric Difference"
        except Exception:
            pass
        specs_a = extract_numeric_specs(text_a)
        specs_b = extract_numeric_specs(text_b)
        if specs_a and specs_b and not (specs_a & specs_b):
            return "Numeric Difference"

    # 5. Accessory / Bundle Difference
    acc_a = toks_a & ACCESSORY_WORDS
    acc_b = toks_b & ACCESSORY_WORDS
    if (acc_a and not acc_b) or (acc_b and not acc_a):
        return "Accessory/Bundle Difference"

    # 6. Variant / Version Mismatch
    ver_a = toks_a & VERSION_WORDS
    ver_b = toks_b & VERSION_WORDS
    if (ver_a or ver_b) and (ver_a != ver_b):
        return "Variant/Version Mismatch"

    # 7. Other Semantic
    return "Other Semantic"

def run_e8():
    print("=" * 70)
    print("E8: Error Taxonomy & Slicing Analysis (RQ5)")
    print("=" * 70)

    from src.experiments.e2_attribution import compute_code_features as compute_classical_features

    master_rows = []
    plot_rows = []

    for ds in DATASETS:
        print(f"\nProcessing {ds.upper()} ({DATASET_NAMES[ds]})...")
        # Load canonical test records
        records = []
        with open(DATA_DIR / "canonical" / ds / "test.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))

        # Load pool records for LR training
        with open(DATA_DIR / "pools" / f"{ds}_pool2000.jsonl", "r", encoding="utf-8") as f:
            pool_records = [json.loads(line) for line in f]

        # Load prediction files
        df_e1 = pd.read_csv(RUNS_DIR / f"e1_preds_{ds}.csv")
        df_d_test = pd.read_csv(RUNS_DIR / f"e2_jev_d_{ds}_test.csv")
        df_d_pool = pd.read_csv(RUNS_DIR / f"e2_jev_d_{ds}_pool.csv")

        y_test = df_e1["label"].values.astype(int)
        y_pool = np.array([p["label"] for p in pool_records], dtype=int)

        # Compute classical features
        df_c_pool = compute_classical_features(ds, pool_records)
        df_c_test = compute_classical_features(ds, records)

        # Train J-D+LR and J-D+C+LR at budget=200, seed=0
        np.random.seed(0)
        idx_b200 = np.random.choice(len(pool_records), 200, replace=False)

        # 1. J-D+LR
        scaler_d = StandardScaler()
        X_d_tr = scaler_d.fit_transform(df_d_pool.iloc[idx_b200].values)
        X_d_te = scaler_d.transform(df_d_test.values)
        lr_d = LogisticRegression(max_iter=1000, random_state=0)
        lr_d.fit(X_d_tr, y_pool[idx_b200])
        p_jd = lr_d.predict_proba(X_d_te)[:, 1]

        # 2. J-D+C+LR
        df_dc_pool = pd.concat([df_d_pool, df_c_pool], axis=1)
        df_dc_test = pd.concat([df_d_test, df_c_test], axis=1)
        scaler_dc = StandardScaler()
        X_dc_tr = scaler_dc.fit_transform(df_dc_pool.iloc[idx_b200].values)
        X_dc_te = scaler_dc.transform(df_dc_test.values)
        lr_dc = LogisticRegression(max_iter=1000, random_state=0)
        lr_dc.fit(X_dc_tr, y_pool[idx_b200])
        p_jdc = lr_dc.predict_proba(X_dc_te)[:, 1]

        # 3. Model predictions
        p_jh = df_e1["p_h1n"].values
        p_luna = df_e1["p_luna"].values
        p_ds = df_e1["p_deepseek"].values

        pred_jh = (p_jh >= 0.5).astype(int)
        pred_jd = (p_jd >= 0.5).astype(int)
        pred_jdc = (p_jdc >= 0.5).astype(int)
        pred_luna = (p_luna >= 0.5).astype(int)
        pred_ds = (p_ds >= 0.5).astype(int)

        # Classify all test pairs into taxonomy slices
        slices = [classify_pair_taxonomy(ds, r["record_a"], r["record_b"], r["label"]) for r in records]

        df_eval = pd.DataFrame({
            "pair_id": [r["pair_id"] for r in records],
            "slice": slices,
            "y_true": y_test,
            "pred_jh": pred_jh,
            "pred_jd": pred_jd,
            "pred_jdc": pred_jdc,
            "pred_luna": pred_luna,
            "pred_ds": pred_ds,
        })

        systems = [
            ("J-H1n", "pred_jh"),
            ("J-D+LR", "pred_jd"),
            ("J-D+C+LR", "pred_jdc"),
            ("GPT-6 Luna", "pred_luna"),
            ("DeepSeek V4 Flash", "pred_ds")
        ]

        for sl, grp in df_eval.groupby("slice"):
            n_slice = len(grp)
            y_sl = grp["y_true"].values
            n_pos = int(y_sl.sum())
            n_neg = n_slice - n_pos

            for sys_name, pred_col in systems:
                pred_sl = grp[pred_col].values
                fp = int(((pred_sl == 1) & (y_sl == 0)).sum())
                fn = int(((pred_sl == 0) & (y_sl == 1)).sum())
                err_count = fp + fn
                err_rate = round(err_count / n_slice * 100, 2)
                f1 = round(f1_score(y_sl, pred_sl, zero_division=0) * 100, 2)
                prec = round(precision_score(y_sl, pred_sl, zero_division=0) * 100, 2)
                rec = round(recall_score(y_sl, pred_sl, zero_division=0) * 100, 2)

                master_rows.append({
                    "dataset": ds,
                    "dataset_name": DATASET_NAMES[ds],
                    "slice": sl,
                    "slice_support": n_slice,
                    "pos_count": n_pos,
                    "neg_count": n_neg,
                    "system": sys_name,
                    "fp": fp,
                    "fn": fn,
                    "error_count": err_count,
                    "error_rate_pct": err_rate,
                    "f1": f1,
                    "precision": prec,
                    "recall": rec
                })

                plot_rows.append({
                    "dataset": ds.upper(),
                    "slice": sl,
                    "system": sys_name,
                    "error_rate": err_rate,
                    "support": n_slice
                })

    df_master = pd.DataFrame(master_rows)
    table_path = REPORTS_TABLES / "t_error_taxonomy.csv"
    df_master.to_csv(table_path, index=False)
    print(f"\n[SAVED] Master Table: {table_path} ({len(df_master)} rows)")

    # Generate Publication Figure f_error_slices.png / .pdf
    print("\nGenerating publication figure f_error_slices...")
    df_plot = pd.DataFrame(plot_rows)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
    axes = axes.flatten()

    sys_list = ["J-H1n", "J-D+LR", "J-D+C+LR", "GPT-6 Luna", "DeepSeek V4 Flash"]
    colors = ["#1f77b4", "#2ca02c", "#17becf", "#ff7f0e", "#d62728"]

    for i, ds in enumerate(["wa", "ag", "da", "ab"]):
        ax = axes[i]
        sub = df_plot[df_plot["dataset"] == ds.upper()]
        slice_order = sub.groupby("slice")["support"].mean().sort_values(ascending=False).index.tolist()
        
        n_slices = len(slice_order)
        x = np.arange(n_slices)
        width = 0.16

        for j, s_name in enumerate(sys_list):
            s_data = sub[sub["system"] == s_name].set_index("slice")
            y_vals = [s_data.loc[sl, "error_rate"] if sl in s_data.index else 0.0 for sl in slice_order]
            offset = (j - 2) * width
            ax.bar(x + offset, y_vals, width, label=s_name if i == 0 else "", color=colors[j], alpha=0.9)

        ax.set_title(f"({chr(97+i)}) {DATASET_NAMES[ds]} ({ds.upper()})", fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("")
        ax.set_ylabel("Error Rate (%)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(slice_order, rotation=28, ha="right", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        if i == 0:
            ax.legend(title="System", frameon=True, fontsize=9, loc="upper right")

    plt.tight_layout()
    fig_png = REPORTS_FIGS / "f_error_slices.png"
    fig_pdf = REPORTS_FIGS / "f_error_slices.pdf"
    plt.savefig(fig_png, dpi=300, bbox_inches="tight")
    plt.savefig(fig_pdf, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Publication Figure: {fig_png}")
    print(f"[SAVED] Publication Figure: {fig_pdf}")

    print("\n" + "=" * 70)
    print("E8 ERROR TAXONOMY EXPERIMENT COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    run_e8()
