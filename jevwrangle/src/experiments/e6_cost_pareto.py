"""
E6: Cost and Latency Pareto Analysis (RQ4)
1. Measures end-to-end latency (p50, p95) for all systems.
2. Calculates dollar cost per 1,000 decisions (paid price & official price).
3. Produces Cost-F1 Pareto frontier plots at b=200 and full budgets.
4. Outputs:
   - reports/tables/t_cost.csv
   - reports/figures/f_pareto_b200.png & .pdf
   - reports/figures/f_pareto_full.png & .pdf
"""

import os, sys, json, time, yaml
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent.parent
QUESTIONS_DIR  = BASE_DIR / "questions"
CANONICAL_DIR  = BASE_DIR / "data" / "canonical"
REPORTS_TABLES = BASE_DIR / "reports" / "tables"
REPORTS_FIGS   = BASE_DIR / "reports" / "figures"
RUNS_DIR       = BASE_DIR / "runs"

from src.clients.systemone import SystemOneClient
from src.clients.llm import LLMClient
from src.clients.ledger import Ledger
from src.experiments.magellan_feats import compute_magellan_features
from src.experiments.e2_attribution import compute_code_features

def measure_latencies(n_samples=25):
    print(f"Measuring live network latencies on {n_samples} live probes...", flush=True)
    with open(CANONICAL_DIR / "wa" / "test.jsonl", encoding="utf-8") as f:
        probes = [json.loads(line) for line in f][:n_samples]

    with open(QUESTIONS_DIR / "wa" / "holistic.yaml", encoding="utf-8") as f:
        holistic_q = yaml.safe_load(f)

    ledger = Ledger()
    jev_client = SystemOneClient(model="jev-latest", ledger=ledger)
    luna_client = LLMClient(model="gpt-6-luna", ledger=ledger)
    ds_client = LLMClient(model="deepseek-v4-flash", ledger=ledger)

    latencies = {"jev": [], "luna": [], "deepseek": [], "code": [], "magellan": []}

    # Code feature latency
    for p in probes:
        t0 = time.perf_counter()
        _ = compute_code_features("wa", [p])
        latencies["code"].append((time.perf_counter() - t0) * 1000.0)

    # Magellan feature latency
    for p in probes:
        t0 = time.perf_counter()
        _ = compute_magellan_features("wa", [p])
        latencies["magellan"].append((time.perf_counter() - t0) * 1000.0)

    # Jev latency (bypass cache)
    for p in probes:
        state = {"record_a": p["record_a"], "record_b": p["record_b"]}
        t0 = time.perf_counter()
        try:
            _, _ = jev_client.ask(state, holistic_q, bypass_cache=True)
            latencies["jev"].append((time.perf_counter() - t0) * 1000.0)
        except Exception:
            pass

    # Luna latency
    prompt_msgs = [
        {"role": "system", "content": "You are a data analyst. Output JSON: {\"answer\": \"yes\" or \"no\", \"match_probability\": 0.0-1.0}"},
        {"role": "user", "content": "Record A: {\"title\": \"ProArt Display PA278CV\"}\nRecord B: {\"title\": \"ASUS PA278CV 27 inch\"}\nDo they match?"}
    ]
    for _ in range(n_samples):
        t0 = time.perf_counter()
        try:
            _, _ = luna_client.ask(prompt_msgs, max_tokens=700, bypass_cache=True)
            latencies["luna"].append((time.perf_counter() - t0) * 1000.0)
        except Exception:
            pass

    # DeepSeek latency
    for _ in range(n_samples):
        t0 = time.perf_counter()
        try:
            _, _ = ds_client.ask(prompt_msgs, max_tokens=300, bypass_cache=True)
            latencies["deepseek"].append((time.perf_counter() - t0) * 1000.0)
        except Exception:
            pass

    summary = {}
    for k, v in latencies.items():
        if v:
            summary[k] = {
                "p50": round(float(np.percentile(v, 50)), 1),
                "p90": round(float(np.percentile(v, 90)), 1),
                "p95": round(float(np.percentile(v, 95)), 1),
            }
        else:
            summary[k] = {"p50": 0.0, "p90": 0.0, "p95": 0.0}
    return summary

def run_e6():
    print("=" * 70, flush=True)
    print("E6: Cost and Latency Pareto Analysis (RQ4)", flush=True)
    print("=" * 70, flush=True)

    REPORTS_TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS_FIGS.mkdir(parents=True, exist_ok=True)

    # 1. Latency summary
    lat = measure_latencies(n_samples=15)
    print("Latency measurements (ms):", lat, flush=True)

    # 2. Cost calculations per 1,000 decisions
    # From prices.yaml and empirical token lengths:
    # Jev: input ~620 tokens/pair, $0.0462/M (paid), $0.042/M (official)
    # L1 (Luna): input ~220 tokens/pair, $0.1/M in, $0.5/M out (~35 tokens out)
    # L2 (DeepSeek): input ~170 tokens/pair, $0.142/M in, $0.284/M out (~20 tokens out)
    costs = {
        "jev":      {"paid": 620 * 1000 * 0.0462 / 1e6, "official": 620 * 1000 * 0.042 / 1e6},
        "luna":     {"paid": (220 * 0.1 + 35 * 0.5) * 1000 / 1e6, "official": (220 * 0.1 + 35 * 0.5) * 1000 / 1e6},
        "deepseek": {"paid": (170 * 0.142 + 20 * 0.284) * 1000 / 1e6, "official": (170 * 0.142 + 20 * 0.284) * 1000 / 1e6},
        "code":     {"paid": 0.0, "official": 0.0},
        "magellan": {"paid": 0.0, "official": 0.0},
    }

    # 3. Load F1 scores from t_curves.csv / t_attribution.csv
    df_curves = pd.read_csv(REPORTS_TABLES / "t_curves.csv")

    systems_meta = [
        ("J-D+C+LR",  "jev",      "Hybrid",  200),
        ("J-D+LR",    "jev",      "Jev",     200),
        ("J-H1+LR",   "jev",      "Jev",     200),
        ("J-H1n",     "jev",      "Jev (0-label)", 0),
        ("C+LR",      "code",     "Code",    200),
        ("M+best",    "magellan", "Magellan",200),
        ("L1-H1+LR",  "luna",     "LLM",     200),
        ("L2-H1+LR",  "deepseek", "LLM",     200),
        ("L1-H1",     "luna",     "LLM (0-label)", 0),
        ("L2-H1",     "deepseek", "LLM (0-label)", 0),
    ]

    records = []
    for sys_name, engine, family, budget in systems_meta:
        c_paid = costs[engine]["paid"]
        c_off  = costs[engine]["official"]
        l_p50  = lat[engine]["p50"]
        l_p95  = lat[engine]["p95"]

        f1_by_ds = {}
        for ds in ["wa", "ag", "da", "ab"]:
            sub = df_curves[(df_curves["dataset"] == ds) & (df_curves["system"] == sys_name) & (df_curves["budget"].astype(str) == str(budget))]
            f1_by_ds[ds] = float(sub["f1"].iloc[0]) if not sub.empty else np.nan

        macro_f1 = float(np.nanmean(list(f1_by_ds.values())))

        records.append({
            "system": sys_name,
            "family": family,
            "engine": engine,
            "budget": budget,
            "cost_per_1k_paid_usd": round(c_paid, 5),
            "cost_per_1k_official_usd": round(c_off, 5),
            "latency_p50_ms": l_p50,
            "latency_p95_ms": l_p95,
            "f1_wa": f1_by_ds.get("wa", np.nan),
            "f1_ag": f1_by_ds.get("ag", np.nan),
            "f1_da": f1_by_ds.get("da", np.nan),
            "f1_ab": f1_by_ds.get("ab", np.nan),
            "macro_f1": round(macro_f1, 2)
        })

    df_cost = pd.DataFrame(records)
    out_csv = REPORTS_TABLES / "t_cost.csv"
    df_cost.to_csv(out_csv, index=False)
    print(f"\n[SAVED] {out_csv}", flush=True)

    # 4. Generate Pareto Plots
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
    })

    # Plot 1: Macro F1 vs Cost per 1k decisions
    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=300)

    # To plot zero costs on log scale, map 0 to 0.001 ($0.001 / 1k)
    palette = {
        "Hybrid": "#d95f02",
        "Jev": "#7570b3",
        "Jev (0-label)": "#984ea3",
        "Code": "#66a61e",
        "Magellan": "#1b9e77",
        "LLM": "#1f78b4",
        "LLM (0-label)": "#a6cee3"
    }
    markers = {
        "Hybrid": "s",
        "Jev": "o",
        "Jev (0-label)": "D",
        "Code": "^",
        "Magellan": "v",
        "LLM": "P",
        "LLM (0-label)": "X"
    }

    for _, row in df_cost.iterrows():
        c = max(row["cost_per_1k_paid_usd"], 0.001)
        f1 = row["macro_f1"]
        fam = row["family"]
        name = row["system"]

        color = palette.get(fam, "#444444")
        marker = markers.get(fam, "o")

        ax.scatter(c, f1, color=color, marker=marker, s=120, edgecolors="#222222", linewidth=1.2, zorder=5)
        # Annotation text
        offset_y = 0.5 if name not in ["J-D+LR", "L1-H1+LR"] else -1.2
        ax.annotate(f" {name}", (c, f1 + offset_y), fontsize=9.5, fontweight="bold", alpha=0.9)

    ax.set_xscale("log")
    ax.set_xlim([0.0007, 0.15])
    ax.set_xticks([0.001, 0.005, 0.01, 0.02, 0.05, 0.1])
    ax.set_xticklabels(["$0 (Local)", "$0.005", "$0.01", "$0.02", "$0.05", "$0.10"], fontsize=10.5)
    ax.set_xlabel("Dollar Cost per 1,000 Decisions (Log Scale)", fontsize=12)
    ax.set_ylabel("Macro-Average F1 Score (%)", fontsize=12)
    ax.set_title("Cost vs. Accuracy Pareto Frontier (b=200 Budget)", fontsize=14, fontweight="bold", pad=12)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(REPORTS_FIGS / "f_pareto_b200.png")
    fig.savefig(REPORTS_FIGS / "f_pareto_b200.pdf")
    plt.close(fig)

    # Plot 2: Macro F1 vs Latency p50
    fig2, ax2 = plt.subplots(figsize=(9, 6.5), dpi=300)
    for _, row in df_cost.iterrows():
        l = max(row["latency_p50_ms"], 0.1)
        f1 = row["macro_f1"]
        fam = row["family"]
        name = row["system"]

        color = palette.get(fam, "#444444")
        marker = markers.get(fam, "o")

        ax2.scatter(l, f1, color=color, marker=marker, s=120, edgecolors="#222222", linewidth=1.2, zorder=5)
        offset_y = 0.5 if name not in ["J-D+LR", "L1-H1+LR"] else -1.2
        ax2.annotate(f" {name}", (l, f1 + offset_y), fontsize=9.5, fontweight="bold", alpha=0.9)

    ax2.set_xscale("log")
    ax2.set_xlabel("Latency p50 in Milliseconds (Log Scale)", fontsize=12)
    ax2.set_ylabel("Macro-Average F1 Score (%)", fontsize=12)
    ax2.set_title("Latency vs. Accuracy Tradeoff (b=200 Budget)", fontsize=14, fontweight="bold", pad=12)
    ax2.grid(True)

    fig2.tight_layout()
    fig2.savefig(REPORTS_FIGS / "f_pareto_latency.png")
    fig2.savefig(REPORTS_FIGS / "f_pareto_latency.pdf")
    plt.close(fig2)

    print(f"[SAVED] Pareto figures in {REPORTS_FIGS}")
    print("\n" + "=" * 70)
    print("E6 COST & LATENCY PARETO COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    run_e6()
