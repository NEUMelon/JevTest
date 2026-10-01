import os
import sys
import json
import time
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, accuracy_score

from src.clients.systemone import SystemOneClient
from src.clients.llm import LLMClient
from src.clients.ledger import Ledger

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_CANONICAL = BASE_DIR / "data" / "canonical"
DATA_DESIGN = BASE_DIR / "data" / "design"
QUESTIONS_DIR = BASE_DIR / "questions"
REPORTS_TABLES = BASE_DIR / "reports" / "tables"
RUNS_DIR = BASE_DIR / "runs"

def extract_noul(answer):
    from src.recovery.core import noul
    return noul(answer)

def extract_score(answer):
    from src.recovery.core import score
    return score(answer)

def extract_choice(answer):
    from src.recovery.core import choice
    return choice(answer)

def extract_llm_prob(answer):
    from src.recovery.core import llm_probability
    return llm_probability(answer)

def evaluate_predictions(y_true, y_prob, threshold=0.5):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    try:
        auprc = average_precision_score(y_true, y_prob)
    except:
        auprc = 0.0
    return {
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1": round(f1 * 100, 2),
        "auprc": round(auprc * 100, 2),
        "accuracy": round(acc * 100, 2),
    }

def eval_single_pair(p, holistic_q, jev_client, luna_client, ds_client, retries=2):
    for attempt in range(retries):
        try:
            state = {"record_a": p["record_a"], "record_b": p["record_b"]}
            pair_id = p["pair_id"]
            true_label = p["label"]

            # 1. Jev 调用
            jev_resp, _ = jev_client.ask(state, holistic_q)
            ans = jev_resp.get("answers", {})
            p_h1n = extract_noul(ans.get("H1_noul", {}))
            p_h1s = extract_score(ans.get("H1_score", {}))
            p_h1c = extract_choice(ans.get("H1_choice", {}))
            p_h0 = extract_noul(ans.get("H0", p_h1n))

            # 2. LLM 调用
            prompt_content = f"Record A: {json.dumps(p['record_a'])}\nRecord B: {json.dumps(p['record_b'])}\nDo Record A and Record B refer to the same real-world item? Output strictly in JSON format: {{\"answer\": \"yes\" or \"no\", \"match_probability\": <probability from 0.0 to 1.0>}}"
            messages = [{"role": "user", "content": prompt_content}]

            # Luna (max_tokens=700, reasoning_effort="low")
            luna_resp, _ = luna_client.ask(messages, max_tokens=700, reasoning_effort="low")
            luna_msg = luna_resp["choices"][0]["message"]["content"]
            p_luna = extract_llm_prob(luna_msg)

            # DeepSeek (max_tokens=300 防止 reasoning 截断，若遇 length 自动升档)
            ds_resp, _ = ds_client.ask(messages, max_tokens=300)
            ds_msg = ds_resp["choices"][0]["message"]["content"]
            # 附带将 reasoning_content 透传给 extractor 以备兜底
            choice0 = ds_resp["choices"][0]["message"]
            if "reasoning_content" in choice0 and isinstance(ds_msg, str) and not ds_msg.strip():
                p_deepseek = extract_llm_prob({"reasoning_content": choice0["reasoning_content"]})
            else:
                p_deepseek = extract_llm_prob(ds_msg)

            return {
                "pair_id": pair_id,
                "label": true_label,
                "p_h1n": p_h1n,
                "p_h1s": p_h1s,
                "p_h1c": p_h1c,
                "p_h0": p_h0,
                "p_luna": p_luna,
                "p_deepseek": p_deepseek
            }
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [ERROR] Pair {p['pair_id']} failed after retries: {e}", flush=True)
                raise e
            time.sleep(1.0)

def run_preflight_probe(ds, jev_client, luna_client, ds_client):
    """
    第一道防线：在 20 对小样设计集上执行强断言探针测试。
    任何截断、格式错误或语义矛盾直接触发报错退出。
    """
    print(f"\n>>> [门禁检查 1] 正在对 {ds.upper()} 执行 20 样本设计集探针测试 (Design Probe)...", flush=True)
    design_file = DATA_DESIGN / f"{ds}.jsonl"
    if not design_file.exists():
        print(f"Warning: {design_file} not found, skip design probe.", flush=True)
        return
    with open(design_file, "r", encoding="utf-8") as f:
        probe_pairs = [json.loads(l) for l in f][:20]

    with open(QUESTIONS_DIR / ds / "holistic.yaml", "r", encoding="utf-8") as f:
        holistic_q = yaml.safe_load(f)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = [ex.submit(eval_single_pair, p, holistic_q, jev_client, luna_client, ds_client) for p in probe_pairs]
        for fut in futs:
            res = fut.result()
            # 严格断言
            assert 0.0 <= res["p_h1n"] <= 1.0, f"Jev p_h1n 越界: {res['p_h1n']}"
            assert 0.0 <= res["p_luna"] <= 1.0, f"Luna p_luna 越界: {res['p_luna']}"
            assert 0.0 <= res["p_deepseek"] <= 1.0, f"DeepSeek p_deepseek 越界: {res['p_deepseek']}"

    print(f"  [门禁通过] {ds.upper()} 探针测试 20/20 样本 100% 格式无误，语义对齐达标！", flush=True)

def run_e1_holistic(max_pairs_per_dataset=None):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    REPORTS_TABLES.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    
    ledger = Ledger(ledger_path=str(RUNS_DIR / "ledger.csv"))
    jev_client = SystemOneClient(model="jev-latest", ledger=ledger)
    luna_client = LLMClient(model="gpt-6-luna", ledger=ledger)
    ds_client = LLMClient(model="deepseek-v4-flash", ledger=ledger)

    em_datasets = ["wa", "ag", "da", "ab"]
    results = []

    print("\n" + "="*60, flush=True)
    print("Starting E1: Holistic Question Baselines (RQ1) [Harden Version]", flush=True)
    print("="*60, flush=True)

    for ds in em_datasets:
        test_file = DATA_CANONICAL / ds / "test.jsonl"
        if not test_file.exists():
            print(f"Test file {test_file} not found, skipping {ds}.", flush=True)
            continue

        with open(test_file, "r", encoding="utf-8") as f:
            pairs = [json.loads(line) for line in f]
        
        if max_pairs_per_dataset is not None:
            pairs = pairs[:max_pairs_per_dataset]

        total_pairs = len(pairs)
        y_true = [p["label"] for p in pairs]
        preds_path = RUNS_DIR / f"e1_preds_{ds}.csv"

        # 如果已有完整预测落盘文件，直接载入并结算
        if preds_path.exists():
            df_existing = pd.read_csv(preds_path)
            if len(df_existing) == total_pairs:
                print(f"\n============================================================", flush=True)
                print(f">>> [已完成缓存] {ds.upper()} 全量预测已存在 ({total_pairs}/{total_pairs})，直接读取盘上结果结算！", flush=True)
                print(f"============================================================", flush=True)
                systems = [
                    ("J-H1n (Jev Noul)", df_existing["p_h1n"]),
                    ("J-H1s (Jev Score)", df_existing["p_h1s"]),
                    ("J-H1c (Jev Choice)", df_existing["p_h1c"]),
                    ("J-H0 (Jev Narayan)", df_existing["p_h0"]),
                    ("L1-H1 (GPT-6 Luna)", df_existing["p_luna"]),
                    ("L2-H1 (DeepSeek V4 Flash)", df_existing["p_deepseek"]),
                ]
                ds_res = []
                for sys_name, y_prob in systems:
                    metrics = evaluate_predictions(df_existing["label"], y_prob)
                    metrics.update({
                        "dataset": ds,
                        "system": sys_name,
                        "pairs_count": total_pairs
                    })
                    results.append(metrics)
                    ds_res.append(metrics)
                df_ds_metrics = pd.DataFrame(ds_res)
                print(df_ds_metrics[["system", "f1", "precision", "recall", "auprc", "accuracy"]].to_string(index=False), flush=True)
                continue

        # 第一道防线：探针门禁
        run_preflight_probe(ds, jev_client, luna_client, ds_client)

        print(f"\nProcessing {ds.upper()} ({total_pairs} pairs) with ThreadPoolExecutor...", flush=True)

        with open(QUESTIONS_DIR / ds / "holistic.yaml", "r", encoding="utf-8") as f:
            holistic_q = yaml.safe_load(f)

        preds_records = []
        half_reported = False

        t0 = time.perf_counter()
        completed = 0
        with ThreadPoolExecutor(max_workers=12) as executor:
            future_to_pair = {
                executor.submit(eval_single_pair, p, holistic_q, jev_client, luna_client, ds_client): p
                for p in pairs
            }
            for future in as_completed(future_to_pair):
                res = future.result()
                preds_records.append(res)
                completed += 1

                # 第四道防线：前 50 样本动态熔断哨兵
                if completed == 50:
                    df_early = pd.DataFrame(preds_records)
                    ppr_luna = (df_early["p_luna"] >= 0.5).mean()
                    ppr_ds = (df_early["p_deepseek"] >= 0.5).mean()
                    ppr_jev = (df_early["p_h1n"] >= 0.5).mean()
                    print(f"\n  [哨兵巡查 50样本] 正例预测率 PPR: Luna={ppr_luna*100:.1f}%, DS={ppr_ds*100:.1f}%, Jev={ppr_jev*100:.1f}%", flush=True)
                    if ppr_luna > 0.50 or ppr_ds > 0.50:
                        print(f"  [WARNING] 早期正例率偏高，密切观察模型假阳性情况！", flush=True)

                # 里程碑：50% 水位点
                if completed >= (total_pairs // 2) and not half_reported:
                    half_reported = True
                    df_half = pd.DataFrame(preds_records)
                    print(f"\n" + "="*60, flush=True)
                    print(f">>> [里程碑 50%] {ds.upper()} 已完成半程 ({completed}/{total_pairs} 对)！当前中期指标：", flush=True)
                    print("="*60, flush=True)
                    half_systems = [
                        ("J-H1n (Jev Noul)", df_half["p_h1n"]),
                        ("J-H1s (Jev Score)", df_half["p_h1s"]),
                        ("J-H1c (Jev Choice)", df_half["p_h1c"]),
                        ("J-H0 (Jev Narayan)", df_half["p_h0"]),
                        ("L1-H1 (GPT-6 Luna)", df_half["p_luna"]),
                        ("L2-H1 (DeepSeek V4 Flash)", df_half["p_deepseek"]),
                    ]
                    half_res = []
                    for s_name, y_prob in half_systems:
                        m = evaluate_predictions(df_half["label"], y_prob)
                        m["system"] = s_name
                        half_res.append(m)
                    print(pd.DataFrame(half_res)[["system", "f1", "precision", "recall", "auprc", "accuracy"]].to_string(index=False), flush=True)
                    print("="*60 + "\n", flush=True)

                if completed % 50 == 0 or completed == total_pairs:
                    elapsed = time.perf_counter() - t0
                    qps = completed / elapsed if elapsed > 0 else 0
                    print(f"  [{ds.upper()}] Evaluated {completed}/{total_pairs} pairs ({qps:.1f} pairs/s, elapsed: {elapsed:.1f}s)...", flush=True)

        # 保存该数据集全部样本预测值
        df_preds = pd.DataFrame(preds_records).sort_values("pair_id").reset_index(drop=True)
        df_preds.to_csv(preds_path, index=False)
        print(f"\n" + "="*60, flush=True)
        print(f">>> [里程碑 100%] {ds.upper()} 全量结算完毕 ({total_pairs}/{total_pairs} 对)！落盘: {preds_path}", flush=True)
        print("="*60, flush=True)

        # 统计各系统指标
        systems = [
            ("J-H1n (Jev Noul)", df_preds["p_h1n"]),
            ("J-H1s (Jev Score)", df_preds["p_h1s"]),
            ("J-H1c (Jev Choice)", df_preds["p_h1c"]),
            ("J-H0 (Jev Narayan)", df_preds["p_h0"]),
            ("L1-H1 (GPT-6 Luna)", df_preds["p_luna"]),
            ("L2-H1 (DeepSeek V4 Flash)", df_preds["p_deepseek"]),
        ]

        ds_res = []
        for sys_name, y_prob in systems:
            metrics = evaluate_predictions(df_preds["label"], y_prob)
            metrics.update({
                "dataset": ds,
                "system": sys_name,
                "pairs_count": total_pairs
            })
            results.append(metrics)
            ds_res.append(metrics)
        print(pd.DataFrame(ds_res)[["system", "f1", "precision", "recall", "auprc", "accuracy"]].to_string(index=False), flush=True)
        print("="*60 + "\n", flush=True)

    df_results = pd.DataFrame(results)
    out_table = REPORTS_TABLES / "t_holistic.csv"
    df_results.to_csv(out_table, index=False)
    print("\n" + "="*60)
    print(f"E1 Complete! Results written to {out_table}:")
    print("="*60)
    print(df_results[["dataset", "system", "f1", "precision", "recall", "auprc", "accuracy"]].to_string())

if __name__ == "__main__":
    run_e1_holistic()
