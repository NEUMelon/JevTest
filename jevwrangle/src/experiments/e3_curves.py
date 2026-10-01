"""
E3: Annotation Efficiency Curves (RQ2)
Evaluates learning curves across budgets: 0, 50, 200, 1000, full.
Systems included:
- J-D0 (0-label point)
- J-H1n (0-label point)
- J-H1+LR
- J-D+LR
- J-D+C+LR
- C+LR (code features)
- M+LR (Magellan-style features + LR)
- M+best (Magellan-style features + Random Forest)
- L1-H1+LR (GPT-6 Luna)
- L2-H1+LR (DeepSeek V4 Flash)

Outputs:
- reports/tables/t_curves.csv
- reports/figures/f_curve_{wa,ag,da,ab}.png & .pdf
- reports/figures/f_curves_all_4panels.png & .pdf
"""

import os, sys, json, time, math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_recall_fscore_support, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt

from src.experiments.magellan_feats import compute_magellan_features

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_CANONICAL = BASE_DIR / "data" / "canonical"
DATA_POOLS     = BASE_DIR / "data" / "pools"
DATA_BUDGETS   = BASE_DIR / "data" / "budgets"
REPORTS_TABLES = BASE_DIR / "reports" / "tables"
REPORTS_FIGURES = BASE_DIR / "reports" / "figures"
RUNS_DIR       = BASE_DIR / "runs"

EM_DATASETS = ["wa", "ag", "da", "ab"]

def load_pairs(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def eval_at_threshold(y_true, y_prob, thr=0.5):
    y_pred = (np.asarray(y_prob) >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    try:
        auprc = average_precision_score(y_true, y_prob)
    except Exception:
        auprc = 0.0
    return {
        "f1":        round(f1   * 100, 2),
        "precision": round(prec * 100, 2),
        "recall":    round(rec  * 100, 2),
        "auprc":     round(auprc* 100, 2),
    }

def best_threshold_on_train(y_tr, p_tr):
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(0.1, 0.91, 0.05):
        f1 = f1_score(y_tr, (np.asarray(p_tr) >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr

def run_classifier(clf_type, train_df, test_df, feature_cols, budget, seed=42):
    Xtr = np.nan_to_num(train_df[feature_cols].values.astype(float))
    Xte = np.nan_to_num(test_df[feature_cols].values.astype(float))
    ytr = train_df["label"].values.astype(int)
    yte = test_df["label"].values.astype(int)

    if clf_type == "lr":
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        if budget >= 1000 and len(set(ytr)) > 1:
            n_pos = max(1, int(ytr.sum()))
            k = min(5, n_pos)
            best_C, best_cv = 1.0, -1.0
            if k >= 2:
                for C in [0.01, 0.1, 1.0, 10.0]:
                    lr = LogisticRegression(C=C, max_iter=2000, random_state=seed)
                    cv = cross_val_score(
                        lr, Xtr_s, ytr,
                        cv=StratifiedKFold(k, shuffle=True, random_state=seed),
                        scoring="average_precision"
                    ).mean()
                    if cv > best_cv:
                        best_cv, best_C = cv, C
            C = best_C
        else:
            C = 1.0

        model = LogisticRegression(C=C, max_iter=2000, random_state=seed)
        model.fit(Xtr_s, ytr)
        p_te = model.predict_proba(Xte_s)[:, 1]

        if len(ytr) >= 20 and ytr.sum() >= 2:
            p_tr = model.predict_proba(Xtr_s)[:, 1]
            best_thr = best_threshold_on_train(ytr, p_tr)
        else:
            best_thr = 0.5

    elif clf_type == "rf":
        model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=seed, n_jobs=-1)
        model.fit(Xtr, ytr)
        p_te = model.predict_proba(Xte)[:, 1]

        if len(ytr) >= 20 and ytr.sum() >= 2:
            p_tr = model.predict_proba(Xtr)[:, 1]
            best_thr = best_threshold_on_train(ytr, p_tr)
        else:
            best_thr = 0.5
    else:
        raise ValueError(f"Unknown classifier type: {clf_type}")

    return eval_at_threshold(yte, p_te, best_thr)

def evaluate_system(clf_type, system_name, feat_cols, pool_df, test_df, full_df, budget_files, ds):
    budgets = (50, 200, 1000)
    seeds_per_budget = (10, 10, 5)
    records = []

    for b, n_seeds in zip(budgets, seeds_per_budget):
        seed_results = []
        for seed_idx in range(n_seeds):
            key = f"b{b}_seed{seed_idx}"
            bf = budget_files.get(key)
            if bf is None:
                continue
            with open(bf) as f:
                labeled_ids = set(json.load(f)["pair_ids"])

            train_part = pool_df[pool_df["pair_id"].isin(labeled_ids)].copy()
            if len(train_part) == 0:
                continue

            m = run_classifier(clf_type, train_part, test_df, feat_cols, b, seed=seed_idx)
            seed_results.append(m)

        if seed_results:
            avg = {k: round(float(np.mean([r[k] for r in seed_results])), 2) for k in seed_results[0]}
            std_f1 = round(float(np.std([r["f1"] for r in seed_results])), 2)
            records.append({
                "dataset": ds, "system": system_name, "budget": b,
                **avg, "f1_std": std_f1, "n_seeds": len(seed_results)
            })

    # Full budget (3 seeds)
    if full_df is not None:
        seed_results = []
        for seed_idx in range(3):
            m = run_classifier(clf_type, full_df, test_df, feat_cols, len(full_df), seed=seed_idx)
            seed_results.append(m)
        avg = {k: round(float(np.mean([r[k] for r in seed_results])), 2) for k in seed_results[0]}
        std_f1 = round(float(np.std([r["f1"] for r in seed_results])), 2)
        records.append({
            "dataset": ds, "system": system_name, "budget": "full",
            **avg, "f1_std": std_f1, "n_seeds": 3
        })

    return records

def load_or_compute_magellan(ds, pairs, split):
    path = RUNS_DIR / f"e3_magellan_{ds}_{split}.csv"
    if path.exists():
        df = pd.read_csv(path)
        if len(df) == len(pairs):
            return df
    df = compute_magellan_features(ds, pairs)
    df.to_csv(path, index=False)
    return df

def generate_learning_curve_plots(df_curves):
    REPORTS_FIGURES.mkdir(parents=True, exist_ok=True)
    
    # Stylings
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
    })

    system_colors = {
        "J-D+C+LR":  "#d95f02", # Deep Orange
        "J-D+LR":    "#7570b3", # Purple
        "J-H1+LR":   "#e7298a", # Pink / Magenta
        "C+LR":      "#66a61e", # Olive Green
        "M+best":    "#1b9e77", # Teal / Green
        "M+LR":      "#a6761d", # Brown
        "L1-H1+LR":  "#1f78b4", # Blue (GPT-6 Luna)
        "L2-H1+LR":  "#e31a1c", # Red (DeepSeek)
        "J-D0":      "#ff7f00", # Orange dashed
        "J-H1n":     "#cab2d6", # Lilac
        "L1-H1":     "#a6cee3", # Light Blue
    }
    system_markers = {
        "J-D+C+LR": "s",
        "J-D+LR": "o",
        "J-H1+LR": "^",
        "C+LR": "D",
        "M+best": "v",
        "M+LR": "<",
        "L1-H1+LR": "P",
        "L2-H1+LR": "X",
    }

    # 4-panel overall figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    axes = axes.flatten()

    dataset_names = {
        "wa": "Walmart-Amazon (WA)",
        "ag": "Amazon-Google (AG)",
        "da": "DBLP-ACM (DA)",
        "ab": "Abt-Buy (AB)"
    }

    for idx, ds in enumerate(EM_DATASETS):
        ax = axes[idx]
        df_ds = df_curves[df_curves["dataset"] == ds]

        # First plot 0-budget horizontal reference lines if available
        j_d0 = df_ds[(df_ds["system"] == "J-D0") & (df_ds["budget"].astype(str) == "0")]
        if not j_d0.empty:
            val = j_d0["f1"].iloc[0]
            ax.axhline(val, color="#ff7f00", linestyle=":", linewidth=1.8, label=f"J-D0 (0-label: {val:.1f}%)")

        j_h1n = df_ds[(df_ds["system"] == "J-H1n") & (df_ds["budget"].astype(str) == "0")]
        if not j_h1n.empty:
            val = j_h1n["f1"].iloc[0]
            ax.axhline(val, color="#984ea3", linestyle="--", linewidth=1.5, alpha=0.7, label=f"J-H1n (0-label: {val:.1f}%)")

        # Map budget to numeric x-position
        # Budget points: 50, 200, 1000, full (full mapped to 4000 for visual separation)
        budget_map = {50: 50, 200: 200, 1000: 1000, "50": 50, "200": 200, "1000": 1000, "full": 4000}

        key_systems = ["J-D+C+LR", "J-D+LR", "J-H1+LR", "M+best", "C+LR", "L1-H1+LR", "L2-H1+LR"]
        for sys_name in key_systems:
            df_sys = df_ds[df_ds["system"] == sys_name].copy()
            if df_sys.empty:
                continue

            df_sys["x"] = df_sys["budget"].map(budget_map)
            df_sys = df_sys.dropna(subset=["x"]).sort_values("x")
            if df_sys.empty:
                continue

            x = df_sys["x"].values
            y = df_sys["f1"].values
            std = df_sys["f1_std"].values

            color = system_colors.get(sys_name, "#555555")
            marker = system_markers.get(sys_name, "o")

            ax.plot(x, y, label=sys_name, color=color, marker=marker, markersize=7, linewidth=2.0)
            ax.fill_between(x, y - std, y + std, color=color, alpha=0.12)

        ax.set_xscale("log")
        ax.set_xticks([50, 200, 1000, 4000])
        ax.set_xticklabels(["50", "200", "1k", "Full"], fontsize=11)
        ax.set_title(dataset_names[ds], fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Annotation Budget (b)", fontsize=12)
        ax.set_ylabel("F1 Score (%)", fontsize=12)
        ax.grid(True)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

        # Also save individual dataset figure
        fig_ind, ax_ind = plt.subplots(figsize=(8, 6), dpi=300)
        if not j_d0.empty:
            ax_ind.axhline(j_d0["f1"].iloc[0], color="#ff7f00", linestyle=":", linewidth=2.0, label=f"J-D0 (0-label: {j_d0['f1'].iloc[0]:.1f}%)")
        if not j_h1n.empty:
            ax_ind.axhline(j_h1n["f1"].iloc[0], color="#984ea3", linestyle="--", linewidth=1.6, alpha=0.7, label=f"J-H1n (0-label: {j_h1n['f1'].iloc[0]:.1f}%)")

        for sys_name in key_systems + ["M+LR"]:
            df_sys = df_ds[df_ds["system"] == sys_name].copy()
            if df_sys.empty: continue
            df_sys["x"] = df_sys["budget"].map(budget_map)
            df_sys = df_sys.dropna(subset=["x"]).sort_values("x")
            if df_sys.empty: continue

            x = df_sys["x"].values
            y = df_sys["f1"].values
            std = df_sys["f1_std"].values
            color = system_colors.get(sys_name, "#555555")
            marker = system_markers.get(sys_name, "o")

            ax_ind.plot(x, y, label=sys_name, color=color, marker=marker, markersize=8, linewidth=2.2)
            ax_ind.fill_between(x, y - std, y + std, color=color, alpha=0.15)

        ax_ind.set_xscale("log")
        ax_ind.set_xticks([50, 200, 1000, 4000])
        ax_ind.set_xticklabels(["50", "200", "1,000", "Full"], fontsize=12)
        ax_ind.set_title(f"Learning Curves: {dataset_names[ds]}", fontsize=15, fontweight="bold", pad=12)
        ax_ind.set_xlabel("Annotation Budget (Labeled Pairs)", fontsize=13)
        ax_ind.set_ylabel("F1 Score (%)", fontsize=13)
        ax_ind.grid(True)
        ax_ind.legend(loc="lower right", fontsize=10, framealpha=0.92)
        fig_ind.tight_layout()
        fig_ind.savefig(REPORTS_FIGURES / f"f_curve_{ds}.png")
        fig_ind.savefig(REPORTS_FIGURES / f"f_curve_{ds}.pdf")
        plt.close(fig_ind)

    fig.tight_layout()
    fig.savefig(REPORTS_FIGURES / "f_curves_all_4panels.png")
    fig.savefig(REPORTS_FIGURES / "f_curves_all_4panels.pdf")
    plt.close(fig)
    print(f"  [SAVED] Learning curve plots in {REPORTS_FIGURES}", flush=True)

def run_e3():
    print("=" * 70, flush=True)
    print("E3: Annotation Efficiency Curves (RQ2)", flush=True)
    print("=" * 70, flush=True)

    # 1. Load E2 results
    e2_csv = REPORTS_TABLES / "t_attribution.csv"
    if not e2_csv.exists():
        raise FileNotFoundError("t_attribution.csv not found, please run E2 first.")
    df_e2 = pd.read_csv(e2_csv)
    all_results = df_e2.to_dict("records")

    # 2. Evaluate Magellan features (M+LR and M+best / M+RF)
    magellan_records = []
    for ds in EM_DATASETS:
        print(f"\nEvaluating Magellan features on {ds.upper()}...", flush=True)
        test_pairs = load_pairs(DATA_CANONICAL / ds / "test.jsonl")
        pool_pairs = load_pairs(DATA_POOLS / f"{ds}_pool2000.jsonl")
        full_train_pairs = load_pairs(DATA_CANONICAL / ds / "train.jsonl")

        m_test = load_or_compute_magellan(ds, test_pairs, "test")
        m_pool = load_or_compute_magellan(ds, pool_pairs, "pool")
        m_full = load_or_compute_magellan(ds, full_train_pairs, "full")

        m_cols = list(m_test.columns)

        test_meta = pd.DataFrame({"pair_id": [p["pair_id"] for p in test_pairs], "label": [p["label"] for p in test_pairs]})
        pool_meta = pd.DataFrame({"pair_id": [p["pair_id"] for p in pool_pairs], "label": [p["label"] for p in pool_pairs]})
        full_meta = pd.DataFrame({"pair_id": [p["pair_id"] for p in full_train_pairs], "label": [p["label"] for p in full_train_pairs]})

        test_df = pd.concat([test_meta.reset_index(drop=True), m_test.reset_index(drop=True)], axis=1)
        pool_df = pd.concat([pool_meta.reset_index(drop=True), m_pool.reset_index(drop=True)], axis=1)
        full_df = pd.concat([full_meta.reset_index(drop=True), m_full.reset_index(drop=True)], axis=1)

        budget_files = {bf.stem: bf for bf in (DATA_BUDGETS / ds).glob("*.json")}

        # M+LR
        recs_lr = evaluate_system("lr", "M+LR", m_cols, pool_df, test_df, full_df, budget_files, ds)
        # M+best (Random Forest)
        recs_rf = evaluate_system("rf", "M+best", m_cols, pool_df, test_df, full_df, budget_files, ds)

        for r in recs_lr + recs_rf:
            print(f"  [{ds}] {r['system']} b={r['budget']}: F1={r['f1']:.2f}% ±{r['f1_std']:.2f}", flush=True)

        magellan_records.extend(recs_lr)
        magellan_records.extend(recs_rf)

    all_results.extend(magellan_records)

    # 3. Output t_curves.csv
    df_curves = pd.DataFrame(all_results)
    out_csv = REPORTS_TABLES / "t_curves.csv"
    df_curves.to_csv(out_csv, index=False)
    print(f"\n[SAVED] {out_csv} ({len(df_curves)} rows)", flush=True)

    # 4. Generate Figures
    generate_learning_curve_plots(df_curves)

    print("\n" + "=" * 70, flush=True)
    print("E3 EXPERIMENT COMPLETE!", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    run_e3()
