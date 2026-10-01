"""
Experiment 9: Numerical & Identifier Probing (H5)
Diagnostic probes testing whether systems reliably evaluate explicit mathematical,
code-equivalence, and temporal logic on isolated record attributes:
  1. Price within 10% (Walmart-Amazon)
  2. Price within 10% (Abt-Buy)
  3. Model/SKU normalization equivalence (Walmart-Amazon)
  4. Publication Year matching (DBLP-ACM)
  5. Version number matching (Amazon-Google)

Evaluates: Jev (jev-1.13), GPT-6 Luna, DeepSeek V4 Flash.
Metrics: Accuracy, PR-AUC, ROC-AUC, Brier Score, ECE.
Generates:
  - reports/tables/t_probe.csv
  - reports/figures/f_probe.png / .pdf
"""

import os
import sys
import json
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, brier_score_loss

from src.clients.systemone import SystemOneClient
from src.clients.llm import LLMClient
from src.experiments.e2_attribution import extract_version

ROOT_DIR = Path("E:/Desktop/jev/jevwrangle")
RUNS_DIR = ROOT_DIR / "runs"
REPORTS_TABLES = ROOT_DIR / "reports" / "tables"
REPORTS_FIGS = ROOT_DIR / "reports" / "figures"
DATA_DIR = ROOT_DIR / "data"

REPORTS_TABLES.mkdir(parents=True, exist_ok=True)
REPORTS_FIGS.mkdir(parents=True, exist_ok=True)

PROBE_CONFIGS = [
    {
        "task_id": "probe_price_wa",
        "dataset": "wa",
        "name": "Price ±10% (WA)",
        "question": "Is the price of record_a within 10% of the price of record_b?",
        "n_pos": 250,
        "n_neg": 250,
    },
    {
        "task_id": "probe_price_ab",
        "dataset": "ab",
        "name": "Price ±10% (AB)",
        "question": "Is the price of record_a within 10% of the price of record_b?",
        "n_pos": 200,
        "n_neg": 200,
    },
    {
        "task_id": "probe_model_wa",
        "dataset": "wa",
        "name": "Model Code Match (WA)",
        "question": "Are record_a.modelno and record_b.modelno the same code after ignoring case, spaces, hyphens, and slashes?",
        "n_pos": 250,
        "n_neg": 250,
    },
    {
        "task_id": "probe_year_da",
        "dataset": "da",
        "name": "Year Match (DA)",
        "question": "Do record_a.year and record_b.year state the same year?",
        "n_pos": 250,
        "n_neg": 250,
    },
    {
        "task_id": "probe_version_ag",
        "dataset": "ag",
        "name": "Version Match (AG)",
        "question": "Do record_a.title and record_b.title state the same version number?",
        "n_pos": 200,
        "n_neg": 200,
    },
]

def load_canonical_pairs(ds):
    pairs = []
    for split in ["train", "valid", "test"]:
        p_path = DATA_DIR / "canonical" / ds / f"{split}.jsonl"
        if p_path.exists():
            with open(p_path, "r", encoding="utf-8") as f:
                for line in f:
                    pairs.append(json.loads(line))
    return pairs

def build_probe_pairs(config):
    ds = config["dataset"]
    task_id = config["task_id"]
    pairs = load_canonical_pairs(ds)

    pos_pairs, neg_pairs = [], []

    if "price" in task_id:
        for p in pairs:
            try:
                pa = float(p["record_a"].get("price", 0) or 0)
                pb = float(p["record_b"].get("price", 0) or 0)
                if pa > 1.0 and pb > 1.0:
                    rel = abs(pa - pb) / max(pa, pb)
                    item = {"record_a": p["record_a"], "record_b": p["record_b"]}
                    if rel <= 0.10:
                        item["probe_label"] = 1
                        pos_pairs.append(item)
                    elif rel > 0.15:
                        item["probe_label"] = 0
                        neg_pairs.append(item)
            except Exception:
                pass

    elif "model" in task_id:
        for p in pairs:
            ma = p["record_a"].get("modelno")
            mb = p["record_b"].get("modelno")
            if ma and mb and str(ma).lower() not in ["0.0", "nan", "none"] and str(mb).lower() not in ["0.0", "nan", "none"]:
                ca = re.sub(r"[^a-zA-Z0-9]", "", str(ma)).lower()
                cb = re.sub(r"[^a-zA-Z0-9]", "", str(mb)).lower()
                if ca and cb:
                    item = {"record_a": p["record_a"], "record_b": p["record_b"]}
                    if ca == cb:
                        item["probe_label"] = 1
                        pos_pairs.append(item)
                    else:
                        item["probe_label"] = 0
                        neg_pairs.append(item)

    elif "year" in task_id:
        for p in pairs:
            ya = str(p["record_a"].get("year", "")).strip()
            yb = str(p["record_b"].get("year", "")).strip()
            if len(ya) == 4 and len(yb) == 4 and ya.isdigit() and yb.isdigit():
                item = {"record_a": p["record_a"], "record_b": p["record_b"]}
                if int(ya) == int(yb):
                    item["probe_label"] = 1
                    pos_pairs.append(item)
                else:
                    item["probe_label"] = 0
                    neg_pairs.append(item)

    elif "version" in task_id:
        for p in pairs:
            va = extract_version(p["record_a"].get("title"))
            vb = extract_version(p["record_b"].get("title"))
            if va is not None and vb is not None:
                item = {"record_a": p["record_a"], "record_b": p["record_b"]}
                if va == vb:
                    item["probe_label"] = 1
                    pos_pairs.append(item)
                else:
                    item["probe_label"] = 0
                    neg_pairs.append(item)

    # Deterministic sampling with seed 42
    np.random.seed(42)
    n_pos = min(config["n_pos"], len(pos_pairs))
    n_neg = min(config["n_neg"], len(neg_pairs))

    idx_pos = np.random.choice(len(pos_pairs), n_pos, replace=False)
    idx_neg = np.random.choice(len(neg_pairs), n_neg, replace=False)

    sampled = [pos_pairs[i] for i in idx_pos] + [neg_pairs[i] for i in idx_neg]
    np.random.shuffle(sampled)
    for i, item in enumerate(sampled):
        item["probe_id"] = f"{task_id}_{i:04d}"
    return sampled

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        idx = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if idx.sum() > 0:
            bin_acc = y_true[idx].mean()
            bin_conf = y_prob[idx].mean()
            ece += (idx.sum() / len(y_true)) * abs(bin_acc - bin_conf)
    return round(ece * 100, 2)

def run_e9():
    print("=" * 70)
    print("E9: Numerical & Identifier Probing (H5)")
    print("=" * 70)

    jev_client = SystemOneClient(model="jev-1.13")
    luna_client = LLMClient(model="gpt-6-luna")
    ds_client = LLMClient(model="deepseek-v4-flash")

    summary_rows = []
    plot_rows = []

    for cfg in PROBE_CONFIGS:
        task_id = cfg["task_id"]
        task_name = cfg["name"]
        q_text = cfg["question"]
        print(f"\n[{task_id.upper()}] {task_name}")

        probe_pairs = build_probe_pairs(cfg)
        y_true = np.array([p["probe_label"] for p in probe_pairs])
        print(f"  Sampled {len(probe_pairs)} pairs (Pos={int(y_true.sum())}, Neg={len(y_true)-int(y_true.sum())})")

        # 1. Jev Evaluation
        jev_file = RUNS_DIR / f"e9_{task_id}_jev.csv"
        if jev_file.exists():
            df_jev = pd.read_csv(jev_file)
            p_jev = df_jev["prob"].values
            print(f"  [cache hit] Jev: {len(df_jev)} probes loaded")
        else:
            print(f"  Querying Jev on {len(probe_pairs)} probes...")
            p_jev = [0.5] * len(probe_pairs)

            def _query_jev(idx, item):
                clean_a = {k: ("" if v is None else str(v)) for k, v in item["record_a"].items()}
                clean_b = {k: ("" if v is None else str(v)) for k, v in item["record_b"].items()}
                state = {"record_a": clean_a, "record_b": clean_b}
                q_dict = {"probe": {"instructions": q_text, "type": "noul"}}
                for attempt in range(5):
                    try:
                        resp, _ = jev_client.ask(state, q_dict)
                        ans = resp.get("answers", {}).get("probe", {})
                        prob = float(ans.get("noul", 0.5))
                        return idx, prob
                    except Exception as e:
                        if attempt == 4:
                            print(f"    [WARN] Jev {task_id} pair {idx} failed: {e}", flush=True)
                            return idx, 0.5
                        time.sleep(1.0 * (attempt + 1))

            completed = 0
            total = len(probe_pairs)
            with ThreadPoolExecutor(max_workers=8) as ex:
                futs = [ex.submit(_query_jev, i, p) for i, p in enumerate(probe_pairs)]
                for fut in as_completed(futs):
                    i, prob = fut.result()
                    p_jev[i] = prob
                    completed += 1
                    if completed % 100 == 0 or completed == total:
                        print(f"    Jev {task_id}: {completed}/{total} probes done", flush=True)

            df_jev = pd.DataFrame({
                "probe_id": [p["probe_id"] for p in probe_pairs],
                "label": y_true,
                "prob": p_jev
            })
            df_jev.to_csv(jev_file, index=False)
            print(f"  [SAVED] {jev_file}")

        # 2. LLM Evaluations (Luna & DeepSeek)
        for m_name, client in [("gpt-6-luna", luna_client), ("deepseek-v4-flash", ds_client)]:
            llm_disp = "GPT-6 Luna" if "luna" in m_name else "DeepSeek V4 Flash"
            llm_file = RUNS_DIR / f"e9_{task_id}_{m_name}.csv"

            if llm_file.exists():
                df_llm = pd.read_csv(llm_file)
                p_llm = df_llm["prob"].values
                print(f"  [cache hit] {llm_disp}: {len(df_llm)} probes loaded")
            else:
                print(f"  Querying {llm_disp} on {len(probe_pairs)} probes...")
                p_llm = [0.5] * len(probe_pairs)

                def _query_llm(idx, item):
                    clean_a = {k: ("" if v is None else str(v)) for k, v in item["record_a"].items()}
                    clean_b = {k: ("" if v is None else str(v)) for k, v in item["record_b"].items()}
                    prompt = (
                        f"Record A: {json.dumps(clean_a, ensure_ascii=False)}\n"
                        f"Record B: {json.dumps(clean_b, ensure_ascii=False)}\n\n"
                        f"Question: {q_text}\n"
                        'Output strictly as JSON: {"answer": "yes" or "no", "probability": <0.0-1.0>}'
                    )
                    messages = [{"role": "user", "content": prompt}]
                    max_tok = 600 if "luna" in m_name else 300
                    for attempt in range(5):
                        try:
                            resp, _ = client.ask(messages, max_tokens=max_tok, reasoning_effort="low" if "luna" in m_name else None)
                            choice = resp.get("choices", [{}])[0]
                            content = choice.get("message", {}).get("content", "") or "{}"
                            prob = 0.5
                            m = re.search(r"\{.*\}", content, re.DOTALL)
                            if m:
                                parsed = json.loads(m.group(0))
                                prob = float(parsed.get("probability", 0.5))
                                ans_str = str(parsed.get("answer", "")).lower()
                                if "yes" in ans_str and prob < 0.5:
                                    prob = 0.9
                                elif "no" in ans_str and prob > 0.5:
                                    prob = 0.1
                            return idx, np.clip(prob, 0.0, 1.0)
                        except Exception as e:
                            if attempt == 4:
                                print(f"    [WARN] {llm_disp} {task_id} pair {idx} failed: {e}", flush=True)
                                return idx, 0.5
                            time.sleep(1.0 * (attempt + 1))

                completed = 0
                total = len(probe_pairs)
                with ThreadPoolExecutor(max_workers=8) as ex:
                    futs = [ex.submit(_query_llm, i, p) for i, p in enumerate(probe_pairs)]
                    for fut in as_completed(futs):
                        i, prob = fut.result()
                        p_llm[i] = prob
                        completed += 1
                        if completed % 100 == 0 or completed == total:
                            print(f"    {llm_disp} {task_id}: {completed}/{total} probes done", flush=True)

                df_llm = pd.DataFrame({
                    "probe_id": [p["probe_id"] for p in probe_pairs],
                    "label": y_true,
                    "prob": p_llm
                })
                df_llm.to_csv(llm_file, index=False)
                print(f"  [SAVED] {llm_file}")

        # Compute metrics across 3 systems
        for s_name, p_vals in [("Jev (jev-1.13)", p_jev), ("GPT-6 Luna", p_llm if "luna" in m_name else df_jev["prob"].values)]:
            pass

        # Load all 3 for reporting
        sys_dict = {
            "Jev (jev-1.13)": pd.read_csv(RUNS_DIR / f"e9_{task_id}_jev.csv")["prob"].values,
            "GPT-6 Luna": pd.read_csv(RUNS_DIR / f"e9_{task_id}_gpt-6-luna.csv")["prob"].values,
            "DeepSeek V4 Flash": pd.read_csv(RUNS_DIR / f"e9_{task_id}_deepseek-v4-flash.csv")["prob"].values,
        }

        for sys_name, p_vals in sys_dict.items():
            pred = (p_vals >= 0.5).astype(int)
            acc = round(accuracy_score(y_true, pred) * 100, 2)
            f1 = round(f1_score(y_true, pred, zero_division=0) * 100, 2)
            prec = round(precision_score(y_true, pred, zero_division=0) * 100, 2)
            rec = round(recall_score(y_true, pred, zero_division=0) * 100, 2)
            brier = round(brier_score_loss(y_true, p_vals), 4)
            ece = compute_ece(y_true, p_vals)
            try:
                roc = round(roc_auc_score(y_true, p_vals) * 100, 2)
                pr = round(average_precision_score(y_true, p_vals) * 100, 2)
            except Exception:
                roc, pr = 50.0, 50.0

            summary_rows.append({
                "task_id": task_id,
                "task_name": task_name,
                "dataset": cfg["dataset"].upper(),
                "system": sys_name,
                "accuracy": acc,
                "f1": f1,
                "precision": prec,
                "recall": rec,
                "roc_auc": roc,
                "pr_auc": pr,
                "brier_score": brier,
                "ece": ece
            })

            plot_rows.append({
                "task": task_name,
                "system": sys_name,
                "accuracy": acc,
                "ece": ece
            })

    df_master = pd.DataFrame(summary_rows)
    master_csv = REPORTS_TABLES / "t_probe.csv"
    df_master.to_csv(master_csv, index=False)
    print(f"\n[SAVED] Master Table: {master_csv} ({len(df_master)} rows)")

    # Generate Publication Figure
    print("\nGenerating publication figure f_probe...")
    df_plot = pd.DataFrame(plot_rows)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    tasks = [cfg["name"] for cfg in PROBE_CONFIGS]
    systems = ["Jev (jev-1.13)", "GPT-6 Luna", "DeepSeek V4 Flash"]
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]

    x = np.arange(len(tasks))
    width = 0.24

    for j, sys_name in enumerate(systems):
        sub = df_plot[df_plot["system"] == sys_name].set_index("task")
        accs = [sub.loc[t, "accuracy"] for t in tasks]
        eces = [sub.loc[t, "ece"] for t in tasks]

        ax1.bar(x + (j - 1) * width, accs, width, label=sys_name, color=colors[j], alpha=0.9)
        ax2.bar(x + (j - 1) * width, eces, width, label=sys_name, color=colors[j], alpha=0.9)

    ax1.set_title("(a) Diagnostic Probe Accuracy (%)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks, rotation=22, ha="right", fontsize=9.5)
    ax1.set_ylim([40, 102])
    ax1.legend(frameon=True, fontsize=9.5, loc="lower right")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    ax2.set_title("(b) Expected Calibration Error (ECE %)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_ylabel("ECE (%) [Lower is Better]", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tasks, rotation=22, ha="right", fontsize=9.5)
    ax2.legend(frameon=True, fontsize=9.5, loc="upper right")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_png = REPORTS_FIGS / "f_probe.png"
    fig_pdf = REPORTS_FIGS / "f_probe.pdf"
    plt.savefig(fig_png, dpi=300, bbox_inches="tight")
    plt.savefig(fig_pdf, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Publication Figure: {fig_png}")
    print(f"[SAVED] Publication Figure: {fig_pdf}")

    print("\n" + "=" * 70)
    print("E9 NUMERICAL & ID PROBING COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    run_e9()
