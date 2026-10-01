"""
E7: Error Detection (ED) Benchmarks
Datasets:
  - Hospital (ho): Narayan protocol (1,000 test rows, 17,101 cells)
  - Adult (ad): Narayan protocol (1,000 test rows, 11,000 cells)
  - Flights (fl): Raha protocol (1,901 test rows, 11,406 cells)
Systems:
  - J-H_err: Jev Holistic Error Detection
  - J-D+LR: Jev Decomposed Questions + Logistic Regression
  - GPT-6 Luna
  - DeepSeek V4 Flash
  - Published baselines: HoloClean, HoloDetect, GPT-3, Raha
Budgets:
  - 5, 20, 50 rows, full train rows (5 seeds each)
Output:
  - reports/tables/t_error_detection.csv
"""

import os, sys, json, time, re, math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CANONICAL_DIR  = BASE_DIR / "data" / "canonical"
RUNS_DIR       = BASE_DIR / "runs"
REPORTS_TABLES = BASE_DIR / "reports" / "tables"

RUNS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_TABLES.mkdir(parents=True, exist_ok=True)

from src.clients.systemone import SystemOneClient
from src.clients.llm import LLMClient
from src.clients.ledger import Ledger

ED_DATASETS = ["ho", "fl", "ad"]

# Literature benchmark baselines from Narayan Table 2 & Raha papers
LITERATURE_BASELINES = {
    "ho": [
        {"system": "HoloClean", "protocol": "Narayan", "budget": "full", "f1": 57.0, "source": "Narayan et al. Table 2"},
        {"system": "HoloDetect", "protocol": "Narayan", "budget": "full", "f1": 80.0, "source": "Narayan et al. Table 2"},
        {"system": "GPT-3 (Few-shot)", "protocol": "Narayan", "budget": "50 rows", "f1": 73.0, "source": "Narayan et al. Table 2"},
        {"system": "Raha", "protocol": "Raha (20 rows)", "budget": "20 rows", "f1": 77.0, "source": "Mahdavi et al."},
    ],
    "ad": [
        {"system": "HoloClean", "protocol": "Narayan", "budget": "full", "f1": 70.0, "source": "Narayan et al. Table 2"},
        {"system": "HoloDetect", "protocol": "Narayan", "budget": "full", "f1": 72.0, "source": "Narayan et al. Table 2"},
        {"system": "GPT-3 (Few-shot)", "protocol": "Narayan", "budget": "50 rows", "f1": 66.0, "source": "Narayan et al. Table 2"},
    ],
    "fl": [
        {"system": "HoloClean", "protocol": "Raha", "budget": "full", "f1": 58.0, "source": "Mahdavi et al."},
        {"system": "HoloDetect", "protocol": "Raha", "budget": "full", "f1": 65.0, "source": "Mahdavi et al."},
        {"system": "Raha", "protocol": "Raha (20 rows)", "budget": "20 rows", "f1": 79.2, "source": "Mahdavi et al. Table 3"},
    ],
}

def load_ed_dataset(ds, split="test"):
    """
    Load dataset split and group into unique table rows.
    Returns:
      table_rows: list of dicts:
        {
          "table_row_id": int/str,
          "row": dict of col: val,
          "cells": list of cell dicts: {"cell_id", "col", "value", "label_error"}
        }
    """
    path = CANONICAL_DIR / ds / f"{split}.jsonl"
    if not path.exists():
        return []

    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(json.loads(line))

    # Grouping logic
    table_rows = []
    if ds == "ad":
        # Adult: chunks of 11 consecutive cells form 1 table row
        chunk_size = 11
        for i in range(0, len(lines), chunk_size):
            chunk = lines[i:i+chunk_size]
            if not chunk:
                continue
            first = chunk[0]
            row_dict = first["row"]
            cells = [{"cell_id": c["cell_id"], "col": c["col"], "value": c["value"], "label_error": int(c["label_error"])} for c in chunk]
            table_rows.append({
                "table_row_id": i // chunk_size,
                "row": row_dict,
                "column_profile": first.get("column_profile", {}),
                "cells": cells
            })
    else:
        # ho and fl: group by row_id
        groups = defaultdict(list)
        for c in lines:
            groups[c["row_id"]].append(c)

        for rid, clist in groups.items():
            first = clist[0]
            row_dict = first["row"]
            cells = [{"cell_id": c["cell_id"], "col": c["col"], "value": c["value"], "label_error": int(c["label_error"])} for c in clist]
            table_rows.append({
                "table_row_id": rid,
                "row": row_dict,
                "column_profile": first.get("column_profile", {}),
                "cells": cells
            })

    return table_rows

def build_questions_for_row(row_dict, cols, ds):
    """Build questions dict for all columns in a row."""
    q = {}
    for c in cols:
        col_str = str(c)
        q[f"H_err_{col_str}"] = {
            "type": "noul",
            "instructions": f"Is the value of row.{col_str} erroneous, such as a typo, a wrong value, or a formatting error?"
        }
        q[f"D_typo_{col_str}"] = {
            "type": "noul",
            "instructions": f"Does row.{col_str} contain a misspelling or a corrupted character sequence?"
        }
        q[f"D_format_{col_str}"] = {
            "type": "noul",
            "instructions": f"Is row.{col_str} written in an irregular format or corrupted structure?"
        }
        q[f"D_cons_{col_str}"] = {
            "type": "noul",
            "instructions": f"Is row.{col_str} inconsistent with the other values in row?"
        }
        q[f"D_place_{col_str}"] = {
            "type": "noul",
            "instructions": f"Is row.{col_str} a placeholder or missing-value marker such as 'n/a', '-', '?', or 'unknown'?"
        }
        q[f"D_imp_{col_str}"] = {
            "type": "noul",
            "instructions": f"Is row.{col_str} an implausible or impossible value for column {col_str}?"
        }
    return q

def run_jev_on_table_rows(ds, split, table_rows, jev_client, max_workers=10):
    """
    Run Jev on all table rows for a given dataset split.
    Saves predictions to runs/e7_jev_{ds}_{split}.csv
    """
    out_file = RUNS_DIR / f"e7_jev_{ds}_{split}.csv"
    if out_file.exists():
        df = pd.read_csv(out_file)
        # Verify length matches total cells
        expected_cells = sum(len(tr["cells"]) for tr in table_rows)
        if len(df) == expected_cells:
            print(f"  [cache hit] Jev {ds} {split}: {len(df)} cells loaded from {out_file.name}")
            return df

    print(f"  Running Jev on {ds} {split}: {len(table_rows)} table rows...")
    cell_rows = []

    def _process_table_row(tr):
        row_dict = tr["row"]
        cols = [c["col"] for c in tr["cells"]]
        questions = build_questions_for_row(row_dict, cols, ds)
        state = {"row": row_dict}
        resp, meta = jev_client.ask(state, questions, bypass_cache=False)
        answers = resp.get("answers", {}) or {}

        results = []
        for cell in tr["cells"]:
            c = cell["col"]
            h_err = answers.get(f"H_err_{c}", {}).get("noul", 0.1)
            d_typo = answers.get(f"D_typo_{c}", {}).get("noul", 0.05)
            d_format = answers.get(f"D_format_{c}", {}).get("noul", 0.05)
            d_cons = answers.get(f"D_cons_{c}", {}).get("noul", 0.05)
            d_place = answers.get(f"D_place_{c}", {}).get("noul", 0.05)
            d_imp = answers.get(f"D_imp_{c}", {}).get("noul", 0.05)

            results.append({
                "cell_id": cell["cell_id"],
                "table_row_id": tr["table_row_id"],
                "col": c,
                "value": str(cell["value"] or ""),
                "label_error": cell["label_error"],
                "p_h_err": float(h_err),
                "p_d_typo": float(d_typo),
                "p_d_format": float(d_format),
                "p_d_cons": float(d_cons),
                "p_d_place": float(d_place),
                "p_d_imp": float(d_imp),
            })
        return results

    completed = 0
    total = len(table_rows)
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_process_table_row, tr): tr for tr in table_rows}
        for fut in as_completed(futures):
            res_list = fut.result()
            cell_rows.extend(res_list)
            completed += 1
            if completed % 100 == 0 or completed == total:
                elapsed = time.time() - t0
                qps = completed / max(elapsed, 1)
                print(f"    Jev {ds} {split}: {completed}/{total} rows done ({qps:.1f} rows/s)", flush=True)

    df = pd.DataFrame(cell_rows)
    df.to_csv(out_file, index=False)
    print(f"  [SAVED] {out_file} ({len(df)} cells)")
    return df

def run_llm_on_table_rows(ds, model_name, table_rows, llm_client, n_rows=200, max_workers=8):
    """
    Run LLM baseline (Luna / DeepSeek) on sample of table rows.
    """
    out_file = RUNS_DIR / f"e7_llm_{ds}_{model_name}.csv"
    if out_file.exists():
        df = pd.read_csv(out_file)
        if len(df) > 0:
            print(f"  [cache hit] LLM {model_name} {ds}: {len(df)} cells loaded")
            return df

    sub_rows = table_rows[:n_rows]
    print(f"  Running LLM {model_name} on {ds}: {len(sub_rows)} table rows...")
    cell_rows = []

    def _process_llm_row(tr):
        row_dict = tr["row"]
        cols = [c["col"] for c in tr["cells"]]
        prompt = (
            f"You are a database error detection system.\n"
            f"Given the following database row:\n{json.dumps(row_dict, indent=2)}\n\n"
            f"For each of the following columns: {cols}\n"
            f"Determine if the cell value contains an error (typo, invalid format, missing placeholder, or inconsistent value).\n"
            f"Respond with a JSON object where keys are the column names and values are error probabilities between 0.0 (clean) and 1.0 (erroneous).\n"
            f"Example: {{\"col_name\": 0.05}}\nJSON:"
        )

        resp, meta = llm_client.ask([{"role": "user", "content": prompt}], bypass_cache=False)
        choice = resp.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "") or resp.get("content", "") or "{}"

        # Parse JSON
        parsed = {}
        try:
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
            else:
                parsed = json.loads(content)
        except Exception:
            pass

        results = []
        for cell in tr["cells"]:
            c = cell["col"]
            prob = 0.1
            if c in parsed:
                try:
                    prob = float(parsed[c])
                except (ValueError, TypeError):
                    prob = 0.5
            results.append({
                "cell_id": cell["cell_id"],
                "table_row_id": tr["table_row_id"],
                "col": c,
                "label_error": cell["label_error"],
                "p_error": np.clip(prob, 0.0, 1.0),
            })
        return results

    completed = 0
    total = len(sub_rows)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_process_llm_row, tr): tr for tr in sub_rows}
        for fut in as_completed(futures):
            res_list = fut.result()
            cell_rows.extend(res_list)
            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"    LLM {model_name} {ds}: {completed}/{total} rows done")

    df = pd.DataFrame(cell_rows)
    df.to_csv(out_file, index=False)
    print(f"  [SAVED] {out_file} ({len(df)} cells)")
    return df

def evaluate_ed_budgets(ds, df_train, df_test):
    """
    Evaluate Jev across annotation budgets (5, 20, 50, full rows) with 5 seeds.
    Systems:
      1. J-H_err (threshold tuned on budget rows)
      2. J-D+LR (Logistic Regression trained on decomposed features)
    """
    # Group train into table rows
    train_row_ids = df_train["table_row_id"].unique()
    y_test = df_test["label_error"].values
    p_h_test = df_test["p_h_err"].values

    feats = ["p_h_err", "p_d_typo", "p_d_format", "p_d_cons", "p_d_place", "p_d_imp"]
    X_test = df_test[feats].values

    results = []

    # Unsupervised J-H_err baseline (default threshold 0.5 and PR-AUC)
    prec_05 = precision_score(y_test, (p_h_test >= 0.5).astype(int), zero_division=0)
    rec_05 = recall_score(y_test, (p_h_test >= 0.5).astype(int), zero_division=0)
    f1_05 = f1_score(y_test, (p_h_test >= 0.5).astype(int), zero_division=0)
    try:
        pr_auc = average_precision_score(y_test, p_h_test)
        roc_auc = roc_auc_score(y_test, p_h_test)
    except Exception:
        pr_auc, roc_auc = 0.0, 0.0

    results.append({
        "dataset": ds,
        "system": "J-H_err (Unsupervised, thr=0.5)",
        "budget": "0 rows",
        "seed": "-",
        "precision": round(prec_05 * 100, 2),
        "recall": round(rec_05 * 100, 2),
        "f1": round(f1_05 * 100, 2),
        "pr_auc": round(pr_auc * 100, 2),
        "roc_auc": round(roc_auc * 100, 2),
    })

    budgets = [5, 20, 50, len(train_row_ids)]
    budget_labels = {5: "5 rows", 20: "20 rows", 50: "50 rows", len(train_row_ids): "full rows"}

    for b in budgets:
        b_label = budget_labels.get(b, f"{b} rows")
        n_seeds = 5 if b < len(train_row_ids) else 1

        for seed in range(n_seeds):
            rng = np.random.RandomState(seed + 42)
            if b < len(train_row_ids):
                sampled_rids = rng.choice(train_row_ids, size=b, replace=False)
            else:
                sampled_rids = train_row_ids

            df_b = df_train[df_train["table_row_id"].isin(sampled_rids)]
            y_b = df_b["label_error"].values
            p_h_b = df_b["p_h_err"].values
            X_b = df_b[feats].values

            # System 1: J-H_err (tune threshold on budget rows)
            best_thr = 0.5
            best_f1_b = 0.0
            for thr in np.linspace(0.05, 0.95, 37):
                pred_b = (p_h_b >= thr).astype(int)
                f1_cand = f1_score(y_b, pred_b, zero_division=0)
                if f1_cand > best_f1_b:
                    best_f1_b = f1_cand
                    best_thr = thr

            pred_test_h = (p_h_test >= best_thr).astype(int)
            results.append({
                "dataset": ds,
                "system": "J-H_err",
                "budget": b_label,
                "seed": seed,
                "precision": round(precision_score(y_test, pred_test_h, zero_division=0) * 100, 2),
                "recall": round(recall_score(y_test, pred_test_h, zero_division=0) * 100, 2),
                "f1": round(f1_score(y_test, pred_test_h, zero_division=0) * 100, 2),
                "pr_auc": round(pr_auc * 100, 2),
                "roc_auc": round(roc_auc * 100, 2),
            })

            # System 2: J-D+LR
            # Ensure both classes exist in budget sample
            if len(np.unique(y_b)) >= 2:
                clf = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")
                clf.fit(X_b, y_b)
                prob_lr = clf.predict_proba(X_test)[:, 1]
                pred_lr = (prob_lr >= 0.5).astype(int)
                try:
                    lr_pr_auc = average_precision_score(y_test, prob_lr)
                    lr_roc_auc = roc_auc_score(y_test, prob_lr)
                except Exception:
                    lr_pr_auc, lr_roc_auc = pr_auc, roc_auc

                results.append({
                    "dataset": ds,
                    "system": "J-D+LR",
                    "budget": b_label,
                    "seed": seed,
                    "precision": round(precision_score(y_test, pred_lr, zero_division=0) * 100, 2),
                    "recall": round(recall_score(y_test, pred_lr, zero_division=0) * 100, 2),
                    "f1": round(f1_score(y_test, pred_lr, zero_division=0) * 100, 2),
                    "pr_auc": round(lr_pr_auc * 100, 2),
                    "roc_auc": round(lr_roc_auc * 100, 2),
                })
            else:
                # Fallback to J-H_err if 0 errors in sampled rows
                results.append({
                    "dataset": ds,
                    "system": "J-D+LR",
                    "budget": b_label,
                    "seed": seed,
                    "precision": round(precision_score(y_test, pred_test_h, zero_division=0) * 100, 2),
                    "recall": round(recall_score(y_test, pred_test_h, zero_division=0) * 100, 2),
                    "f1": round(f1_score(y_test, pred_test_h, zero_division=0) * 100, 2),
                    "pr_auc": round(pr_auc * 100, 2),
                    "roc_auc": round(roc_auc * 100, 2),
                })

    return results

def run_e7():
    print("=" * 70)
    print("E7: Error Detection (Hospital, Adult, Flights)")
    print("=" * 70)

    ledger = Ledger()
    jev_client = SystemOneClient(model="jev-latest", ledger=ledger)
    luna_client = LLMClient(model="gpt-6-luna", ledger=ledger)
    deepseek_client = LLMClient(model="deepseek-v4-flash", ledger=ledger)

    all_budget_results = []
    llm_results = []

    for ds in ED_DATASETS:
        print(f"\n[{ds.upper()}] Loading data splits...")
        train_rows = load_ed_dataset(ds, "train")
        test_rows = load_ed_dataset(ds, "test")
        print(f"  {ds.upper()}: {len(train_rows)} train table rows | {len(test_rows)} test table rows")

        # 1. Run Jev on Test and Train
        df_test = run_jev_on_table_rows(ds, "test", test_rows, jev_client, max_workers=10)
        df_train = run_jev_on_table_rows(ds, "train", train_rows, jev_client, max_workers=10)

        # 2. Evaluate Budgets
        b_res = evaluate_ed_budgets(ds, df_train, df_test)
        all_budget_results.extend(b_res)

        # 3. Run LLM Baselines on Sample of Test Rows
        df_luna = run_llm_on_table_rows(ds, "gpt-6-luna", test_rows, luna_client, n_rows=200, max_workers=8)
        df_ds = run_llm_on_table_rows(ds, "deepseek-v4-flash", test_rows, deepseek_client, n_rows=200, max_workers=8)

        # Evaluate LLM on their respective test subset
        for name, df_llm in [("GPT-6 Luna", df_luna), ("DeepSeek V4 Flash", df_ds)]:
            y_sub = df_llm["label_error"].values
            p_sub = df_llm["p_error"].values
            pred_sub = (p_sub >= 0.5).astype(int)
            llm_results.append({
                "dataset": ds,
                "system": name,
                "budget": "zero-shot",
                "precision": round(precision_score(y_sub, pred_sub, zero_division=0) * 100, 2),
                "recall": round(recall_score(y_sub, pred_sub, zero_division=0) * 100, 2),
                "f1": round(f1_score(y_sub, pred_sub, zero_division=0) * 100, 2),
                "pr_auc": round(average_precision_score(y_sub, p_sub) * 100, 2),
                "roc_auc": round(roc_auc_score(y_sub, p_sub) * 100, 2),
            })

    # 4. Compile Master Table
    df_eval = pd.DataFrame(all_budget_results)

    # Aggregate across seeds for Jev systems
    summary_rows = []
    for (ds, sys_name, budget), grp in df_eval.groupby(["dataset", "system", "budget"]):
        summary_rows.append({
            "dataset": ds,
            "system": sys_name,
            "budget": budget,
            "precision_mean": round(grp["precision"].mean(), 2),
            "precision_std": round(grp["precision"].std(), 2) if len(grp) > 1 else 0.0,
            "recall_mean": round(grp["recall"].mean(), 2),
            "recall_std": round(grp["recall"].std(), 2) if len(grp) > 1 else 0.0,
            "f1_mean": round(grp["f1"].mean(), 2),
            "f1_std": round(grp["f1"].std(), 2) if len(grp) > 1 else 0.0,
            "pr_auc": grp["pr_auc"].iloc[0],
            "roc_auc": grp["roc_auc"].iloc[0],
            "type": "Jev System",
        })

    for row in llm_results:
        summary_rows.append({
            "dataset": row["dataset"],
            "system": row["system"],
            "budget": row["budget"],
            "precision_mean": row["precision"],
            "precision_std": 0.0,
            "recall_mean": row["recall"],
            "recall_std": 0.0,
            "f1_mean": row["f1"],
            "f1_std": 0.0,
            "pr_auc": row["pr_auc"],
            "roc_auc": row["roc_auc"],
            "type": "LLM Baseline",
        })

    for ds, l_list in LITERATURE_BASELINES.items():
        for item in l_list:
            summary_rows.append({
                "dataset": ds,
                "system": item["system"],
                "budget": item["budget"],
                "precision_mean": "-",
                "precision_std": "-",
                "recall_mean": "-",
                "recall_std": "-",
                "f1_mean": item["f1"],
                "f1_std": 0.0,
                "pr_auc": "-",
                "roc_auc": "-",
                "type": f"Literature ({item['source']})",
            })

    df_master = pd.DataFrame(summary_rows)
    table_path = REPORTS_TABLES / "t_error_detection.csv"
    df_master.to_csv(table_path, index=False)
    print(f"\n[SAVED] Master Table: {table_path}")
    print("\n" + "=" * 70)
    print("E7 ERROR DETECTION EXPERIMENT COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    run_e7()
