"""
E5: Stability and Robustness under Perturbations (RQ3)
1. Determinism Test:
   - 500 pairs x 5 repetitions with bypass_cache=True on Jev
   - Reports average probability std, max std, and flip rate
2. Perturbations P1-P7:
   - P1: Symmetric swap (record_a <-> record_b)
   - P2: Attribute key order shuffle
   - P3: Paraphrase 1
   - P4: Paraphrase 2
   - P5: Choice options reverse order
   - P6: Neutral keys (left_record / right_record)
   - P7: Plain-text string serialization
3. Models evaluated:
   - Jev: full test set on P1-P7
   - L1 (GPT-6 Luna) & L2 (DeepSeek): 500-pair subset on P1, P3, P6
4. Outputs:
   - reports/tables/t_determinism.csv
   - reports/tables/t_stability.csv
   - reports/tables/t_ensemble.csv
   - reports/figures/f_stability_heatmap.png & .pdf
"""

import os, sys, json, time, yaml, random, copy
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import f1_score, cohen_kappa_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent.parent
QUESTIONS_DIR  = BASE_DIR / "questions"
CANONICAL_DIR  = BASE_DIR / "data" / "canonical"
RUNS_DIR       = BASE_DIR / "runs"
REPORTS_TABLES = BASE_DIR / "reports" / "tables"
REPORTS_FIGS   = BASE_DIR / "reports" / "figures"

from src.clients.systemone import SystemOneClient
from src.clients.llm import LLMClient
from src.clients.ledger import Ledger
from src.experiments.e1_holistic import extract_noul, extract_llm_prob

EM_DATASETS = ["wa", "ag", "da", "ab"]

def get_questions(ds):
    with open(QUESTIONS_DIR / ds / "holistic.yaml", encoding="utf-8") as f:
        holistic_q = yaml.safe_load(f)
    with open(QUESTIONS_DIR / ds / "decomposed.yaml", encoding="utf-8") as f:
        decomposed_q = yaml.safe_load(f)
    with open(QUESTIONS_DIR / ds / "paraphrase.yaml", encoding="utf-8") as f:
        paraphrase_q = yaml.safe_load(f)
    return holistic_q, decomposed_q, paraphrase_q

def to_plain_text(rec):
    return "\n".join(f"{k}: {v}" for k, v in rec.items() if v is not None)

def make_perturbed_requests(p, holistic_q, decomposed_q, paraphrase_q, ds):
    rec_a, rec_b = p["record_a"], p["record_b"]
    clean_state = {"record_a": rec_a, "record_b": rec_b}
    clean_q = {**holistic_q, **decomposed_q}

    # P1: Swap
    s_p1 = {"record_a": rec_b, "record_b": rec_a}
    q_p1 = copy.deepcopy(clean_q)

    # P2: Shuffle key order
    rng = random.Random(42)
    ka = list(rec_a.keys()); rng.shuffle(ka)
    kb = list(rec_b.keys()); rng.shuffle(kb)
    s_p2 = {"record_a": {k: rec_a[k] for k in ka}, "record_b": {k: rec_b[k] for k in kb}}
    q_p2 = copy.deepcopy(clean_q)

    # P3: Paraphrase 1
    s_p3 = copy.deepcopy(clean_state)
    q_p3 = copy.deepcopy(clean_q)
    for qid, renames in paraphrase_q.items():
        if qid in q_p3 and "p1" in renames:
            q_p3[qid]["instructions"] = renames["p1"]
    entity = "publication" if ds == "da" else "product"
    q_p3["H1_noul"]["instructions"] = (
        f"Do `record_a` and `record_b` refer to the identical real-world {entity}? "
        f"Any variance in color, size, capacity, edition, or pack count means they are different. "
        f"Accessories or bundled items do not match."
    )

    # P4: Paraphrase 2
    s_p4 = copy.deepcopy(clean_state)
    q_p4 = copy.deepcopy(clean_q)
    for qid, renames in paraphrase_q.items():
        if qid in q_p4 and "p2" in renames:
            q_p4[qid]["instructions"] = renames["p2"]
    q_p4["H1_noul"]["instructions"] = (
        f"Are `record_a` and `record_b` listings for the very same {entity}? "
        f"Note that variations in edition, version, packaging, capacity, or color represent distinct entities, "
        f"as do accessories."
    )

    # P5: Reverse choice
    s_p5 = copy.deepcopy(clean_state)
    q_p5 = copy.deepcopy(clean_q)
    if "H1_choice" in q_p5 and "criteria" in q_p5["H1_choice"]:
        crit = q_p5["H1_choice"]["criteria"]
        q_p5["H1_choice"]["criteria"] = {
            "different": crit.get("different", ""),
            "cannot_tell": crit.get("cannot_tell", ""),
            "same": crit.get("same", "")
        }

    # P6: Neutral keys
    s_p6 = {"left_record": rec_a, "right_record": rec_b}
    q_p6 = copy.deepcopy(clean_q)
    for qid, qdef in q_p6.items():
        if "instructions" in qdef:
            inst = qdef["instructions"].replace("`record_a`", "`left_record`").replace("`record_b`", "`right_record`")
            inst = inst.replace("record_a", "left_record").replace("record_b", "right_record")
            qdef["instructions"] = inst

    # P7: Plain text
    s_p7 = {"record_a": to_plain_text(rec_a), "record_b": to_plain_text(rec_b)}
    q_p7 = copy.deepcopy(clean_q)

    return {
        "P1": (s_p1, q_p1),
        "P2": (s_p2, q_p2),
        "P3": (s_p3, q_p3),
        "P4": (s_p4, q_p4),
        "P5": (s_p5, q_p5),
        "P6": (s_p6, q_p6),
        "P7": (s_p7, q_p7),
    }

def run_determinism_test(ds, pairs, jev_client, n_samples=500, n_repeats=5):
    cache_file = RUNS_DIR / f"e5_determinism_{ds}.csv"
    if cache_file.exists():
        print(f"  [cache hit] Determinism {ds} ({cache_file.name})", flush=True)
        return pd.read_csv(cache_file)

    sample_pairs = pairs[:n_samples]
    with open(QUESTIONS_DIR / ds / "holistic.yaml", encoding="utf-8") as f:
        holistic_q = yaml.safe_load(f)

    print(f"  Running Determinism on {ds}: {len(sample_pairs)} pairs x {n_repeats} repeats...", flush=True)
    all_reps = [[] for _ in range(len(sample_pairs))]

    for rep in range(n_repeats):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=12) as ex:
            def _probe(idx, p):
                state = {"record_a": p["record_a"], "record_b": p["record_b"]}
                resp, _ = jev_client.ask(state, holistic_q, bypass_cache=True)
                ans = resp.get("answers", {})
                return idx, extract_noul(ans.get("H1_noul", 0.5))

            futs = [ex.submit(_probe, i, p) for i, p in enumerate(sample_pairs)]
            for fut in as_completed(futs):
                idx, val = fut.result()
                all_reps[idx].append(val)
        print(f"    rep {rep+1}/{n_repeats} done ({time.perf_counter() - t0:.1f}s)", flush=True)

    rows = []
    for i, p in enumerate(sample_pairs):
        vals = all_reps[i]
        std_p = float(np.std(vals))
        mean_p = float(np.mean(vals))
        binary_decisions = [int(v >= 0.5) for v in vals]
        flipped = int(len(set(binary_decisions)) > 1)
        rows.append({
            "pair_id": p["pair_id"], "label": p["label"],
            "std_p": std_p, "mean_p": mean_p, "flipped": flipped,
            **{f"rep_{r}": vals[r] for r in range(n_repeats)}
        })
    df = pd.DataFrame(rows)
    df.to_csv(cache_file, index=False)
    return df

def run_jev_perturbation_split(ds, pairs, pert_name, state_q_pairs, jev_client):
    cache_file = RUNS_DIR / f"e5_jev_{ds}_{pert_name}.csv"
    if cache_file.exists():
        df = pd.read_csv(cache_file)
        if len(df) == len(pairs):
            return df

    results = [None] * len(pairs)
    with ThreadPoolExecutor(max_workers=10) as ex:
        def _call_one(idx, sq):
            state, q = sq
            for attempt in range(8):
                try:
                    resp, _ = jev_client.ask(state, q)
                    ans = resp.get("answers", {})
                    p_noul = extract_noul(ans.get("H1_noul", 0.5))
                    return idx, p_noul
                except Exception as e:
                    if attempt == 7:
                        print(f"  [WARN] pair {idx} failed after retries: {e}", flush=True)
                        return idx, 0.5
                    time.sleep(1.0 + attempt * 1.5)

        futs = [ex.submit(_call_one, i, sq) for i, sq in enumerate(state_q_pairs)]
        done = 0
        total = len(pairs)
        for fut in as_completed(futs):
            idx, p_noul = fut.result()
            results[idx] = p_noul
            done += 1
            if done % 500 == 0 or done == total:
                print(f"    Jev {ds} {pert_name}: {done}/{total}", flush=True)

    df = pd.DataFrame({
        "pair_id": [p["pair_id"] for p in pairs],
        "label": [p["label"] for p in pairs],
        "p_h1n": results
    })
    df.to_csv(cache_file, index=False)
    return df

def run_llm_perturbation_split(ds, pairs, pert_name, model_name, client, prompt_fn):
    cache_file = RUNS_DIR / f"e5_llm_{ds}_{model_name}_{pert_name}.csv"
    if cache_file.exists():
        df = pd.read_csv(cache_file)
        if len(df) == len(pairs):
            return df

    max_tok = 700 if "luna" in model_name else 300
    results = [None] * len(pairs)
    with ThreadPoolExecutor(max_workers=8) as ex:
        def _call_llm(idx, p):
            msgs = prompt_fn(p)
            for attempt in range(8):
                try:
                    resp, _ = client.ask(msgs, max_tokens=max_tok)
                    content = resp["choices"][0]["message"]["content"]
                    prob = extract_llm_prob(content)
                    return idx, prob
                except Exception as e:
                    if attempt == 7:
                        print(f"  [WARN] LLM pair {idx} failed after retries: {e}", flush=True)
                        return idx, 0.5
                    time.sleep(1.0 + attempt * 1.5)

        futs = [ex.submit(_call_llm, i, p) for i, p in enumerate(pairs)]
        done = 0
        total = len(pairs)
        for fut in as_completed(futs):
            idx, prob = fut.result()
            results[idx] = prob
            done += 1
            if done % 100 == 0 or done == total:
                print(f"    LLM {model_name} {ds} {pert_name}: {done}/{total}", flush=True)

    df = pd.DataFrame({
        "pair_id": [p["pair_id"] for p in pairs],
        "label": [p["label"] for p in pairs],
        "prob": results
    })
    df.to_csv(cache_file, index=False)
    return df

def run_e5():
    print("=" * 70, flush=True)
    print("E5: Stability and Robustness under Perturbations (RQ3)", flush=True)
    print("=" * 70, flush=True)

    REPORTS_TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS_FIGS.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    ledger = Ledger(ledger_path=str(RUNS_DIR / "ledger.csv"))
    jev_client = SystemOneClient(model="jev-latest", ledger=ledger)
    luna_client = LLMClient(model="gpt-6-luna", ledger=ledger)
    ds_client = LLMClient(model="deepseek-v4-flash", ledger=ledger)

    stability_rows = []
    determinism_rows = []
    ensemble_rows = []

    for ds in EM_DATASETS:
        print(f"\n{'='*70}\n[DATASET] {ds.upper()}\n{'='*70}", flush=True)
        test_pairs = [json.loads(line) for line in open(CANONICAL_DIR / ds / "test.jsonl", encoding="utf-8")]
        holistic_q, decomposed_q, paraphrase_q = get_questions(ds)

        # Clean test baseline predictions
        # Clean test baseline predictions
        e1_test = pd.read_csv(RUNS_DIR / f"e1_preds_{ds}.csv")
        clean_df = e1_test[["pair_id", "label", "p_h1n", "p_luna", "p_deepseek"]].copy()

        # Part 1: Determinism (500 pairs x 5 repeats)
        print(f"\n[Part 1] Determinism Test for {ds.upper()}...", flush=True)
        df_det = run_determinism_test(ds, test_pairs, jev_client, n_samples=500, n_repeats=5)
        avg_std = float(df_det["std_p"].mean())
        max_std = float(df_det["std_p"].max())
        flip_rate_det = float(df_det["flipped"].mean() * 100.0)

        determinism_rows.append({
            "dataset": ds, "n_samples": len(df_det), "n_repeats": 5,
            "avg_std": round(avg_std, 5), "max_std": round(max_std, 5),
            "flip_rate_pct": round(flip_rate_det, 2)
        })
        print(f"  [{ds}] Determinism: avg_std={avg_std:.5f}, flip_rate={flip_rate_det:.2f}%", flush=True)

        # Part 2: Jev Perturbations (P1-P7)
        print(f"\n[Part 2] Jev Perturbations P1-P7 on full test ({len(test_pairs)} pairs)...", flush=True)
        pert_requests = {k: [] for k in ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]}
        for p in test_pairs:
            p_dict = make_perturbed_requests(p, holistic_q, decomposed_q, paraphrase_q, ds)
            for k in pert_requests:
                pert_requests[k].append(p_dict[k])

        pert_preds = {}
        for p_name in ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]:
            df_p = run_jev_perturbation_split(ds, test_pairs, p_name, pert_requests[p_name], jev_client)
            merged = df_p.merge(clean_df, on="pair_id", suffixes=("_pert", "_clean"))

            p_vals = merged["p_h1n_pert"].values
            clean_p_jev = merged["p_h1n_clean"].values
            clean_y = merged["label_clean"].values
            pert_preds[p_name] = p_vals

            mean_abs_dp = float(np.mean(np.abs(p_vals - clean_p_jev)))
            flips = float(np.mean((p_vals >= 0.5) != (clean_p_jev >= 0.5)) * 100.0)
            f1_pert = f1_score(clean_y, (p_vals >= 0.5).astype(int)) * 100.0
            f1_clean = f1_score(clean_y, (clean_p_jev >= 0.5).astype(int)) * 100.0
            delta_f1 = f1_pert - f1_clean
            kappa = float(cohen_kappa_score((clean_p_jev >= 0.5).astype(int), (p_vals >= 0.5).astype(int)))

            stability_rows.append({
                "dataset": ds, "model": "Jev (H1n)", "perturbation": p_name,
                "f1": round(f1_pert, 2), "delta_f1": round(delta_f1, 2),
                "mean_abs_delta_p": round(mean_abs_dp, 4),
                "flip_rate_pct": round(flips, 2),
                "cohen_kappa": round(kappa, 4)
            })
            print(f"  [{ds}] Jev {p_name}: F1={f1_pert:.2f}% (Δ={delta_f1:+.2f}%), flips={flips:.2f}%, κ={kappa:.4f}", flush=True)

        # Part 3: Perturbation Ensemble on Jev {Clean, P1, P3, P4}
        eps = 1e-4
        z_clean = np.log(np.clip(clean_p_jev, eps, 1-eps) / (1 - np.clip(clean_p_jev, eps, 1-eps)))
        z_p1 = np.log(np.clip(pert_preds["P1"], eps, 1-eps) / (1 - np.clip(pert_preds["P1"], eps, 1-eps)))
        z_p3 = np.log(np.clip(pert_preds["P3"], eps, 1-eps) / (1 - np.clip(pert_preds["P3"], eps, 1-eps)))
        z_p4 = np.log(np.clip(pert_preds["P4"], eps, 1-eps) / (1 - np.clip(pert_preds["P4"], eps, 1-eps)))
        z_ens = (z_clean + z_p1 + z_p3 + z_p4) / 4.0
        p_ens = 1.0 / (1.0 + np.exp(-z_ens))
        f1_ens = f1_score(clean_y, (p_ens >= 0.5).astype(int)) * 100.0

        ensemble_rows.append({
            "dataset": ds, "model": "Jev (H1n)",
            "f1_clean": round(f1_clean, 2), "f1_ensemble": round(f1_ens, 2),
            "delta_f1": round(f1_ens - f1_clean, 2)
        })
        print(f"  [{ds}] Jev Ensemble {{Clean, P1, P3, P4}}: F1={f1_ens:.2f}% (Δ={f1_ens - f1_clean:+.2f}%)", flush=True)

        # Part 4: LLM Perturbations on 500-pair subset (P1, P3, P6)
        print(f"\n[Part 4] LLM Perturbations (P1, P3, P6) on 500-pair subset...", flush=True)
        sub_pairs = test_pairs[:500]
        sub_ids = [p["pair_id"] for p in sub_pairs]
        sub_clean = clean_df[clean_df["pair_id"].isin(sub_ids)].copy()

        # LLM prompt builders
        def prompt_clean(p):
            return [{"role": "system", "content": "You are a careful data analyst. Output strictly JSON: {\"answer\": \"yes\" or \"no\", \"match_probability\": <0.0-1.0>}"},
                    {"role": "user", "content": f"Record A: {json.dumps(p['record_a'], ensure_ascii=False)}\nRecord B: {json.dumps(p['record_b'], ensure_ascii=False)}\nDo they refer to the same real-world entity?"}]

        def prompt_p1(p): # Swap
            return [{"role": "system", "content": "You are a careful data analyst. Output strictly JSON: {\"answer\": \"yes\" or \"no\", \"match_probability\": <0.0-1.0>}"},
                    {"role": "user", "content": f"Record A: {json.dumps(p['record_b'], ensure_ascii=False)}\nRecord B: {json.dumps(p['record_a'], ensure_ascii=False)}\nDo they refer to the same real-world entity?"}]

        def prompt_p3(p): # Paraphrase
            return [{"role": "system", "content": "You are a precise data auditor. Respond with JSON: {\"answer\": \"yes\" or \"no\", \"match_probability\": <0.0-1.0>}"},
                    {"role": "user", "content": f"Item 1: {json.dumps(p['record_a'], ensure_ascii=False)}\nItem 2: {json.dumps(p['record_b'], ensure_ascii=False)}\nAre Item 1 and Item 2 identical entities?"}]

        def prompt_p6(p): # Neutral keys
            return [{"role": "system", "content": "You are a careful data analyst. Output strictly JSON: {\"answer\": \"yes\" or \"no\", \"match_probability\": <0.0-1.0>}"},
                    {"role": "user", "content": f"Left Record: {json.dumps(p['record_a'], ensure_ascii=False)}\nRight Record: {json.dumps(p['record_b'], ensure_ascii=False)}\nDo Left Record and Right Record refer to the same entity?"}]

        llm_pert_map = {"P1": prompt_p1, "P3": prompt_p3, "P6": prompt_p6}

        for p_name, p_fn in llm_pert_map.items():
            # Luna
            df_luna = run_llm_perturbation_split(ds, sub_pairs, p_name, "gpt-6-luna", luna_client, p_fn)
            m_luna = df_luna.merge(sub_clean, on="pair_id", suffixes=("_pert", "_clean"))
            p_luna = m_luna["prob"].values
            clean_luna = m_luna["p_luna"].values
            y_luna = m_luna["label_clean"].values

            flips_luna = float(np.mean((p_luna >= 0.5) != (clean_luna >= 0.5)) * 100.0)
            f1_luna = f1_score(y_luna, (p_luna >= 0.5).astype(int)) * 100.0
            f1_clean_luna = f1_score(y_luna, (clean_luna >= 0.5).astype(int)) * 100.0
            delta_luna = f1_luna - f1_clean_luna
            kappa_luna = float(cohen_kappa_score((clean_luna >= 0.5).astype(int), (p_luna >= 0.5).astype(int)))
            mean_dp_luna = float(np.mean(np.abs(p_luna - clean_luna)))

            stability_rows.append({
                "dataset": ds, "model": "GPT-6 Luna", "perturbation": p_name,
                "f1": round(f1_luna, 2), "delta_f1": round(delta_luna, 2),
                "mean_abs_delta_p": round(mean_dp_luna, 4),
                "flip_rate_pct": round(flips_luna, 2),
                "cohen_kappa": round(kappa_luna, 4)
            })

            # DeepSeek
            df_ds = run_llm_perturbation_split(ds, sub_pairs, p_name, "deepseek-v4-flash", ds_client, p_fn)
            m_ds = df_ds.merge(sub_clean, on="pair_id", suffixes=("_pert", "_clean"))
            p_ds = m_ds["prob"].values
            clean_ds = m_ds["p_deepseek"].values
            y_ds = m_ds["label_clean"].values

            flips_ds = float(np.mean((p_ds >= 0.5) != (clean_ds >= 0.5)) * 100.0)
            f1_ds = f1_score(y_ds, (p_ds >= 0.5).astype(int)) * 100.0
            f1_clean_ds = f1_score(y_ds, (clean_ds >= 0.5).astype(int)) * 100.0
            delta_ds = f1_ds - f1_clean_ds
            kappa_ds = float(cohen_kappa_score((clean_ds >= 0.5).astype(int), (p_ds >= 0.5).astype(int)))
            mean_dp_ds = float(np.mean(np.abs(p_ds - clean_ds)))

            stability_rows.append({
                "dataset": ds, "model": "DeepSeek V4 Flash", "perturbation": p_name,
                "f1": round(f1_ds, 2), "delta_f1": round(delta_ds, 2),
                "mean_abs_delta_p": round(mean_dp_ds, 4),
                "flip_rate_pct": round(flips_ds, 2),
                "cohen_kappa": round(kappa_ds, 4)
            })

            print(f"  [{ds}] LLM {p_name} | Luna: flips={flips_luna:.2f}%, κ={kappa_luna:.4f} | DS: flips={flips_ds:.2f}%, κ={kappa_ds:.4f}", flush=True)

    # Save Tables
    df_det_out = pd.DataFrame(determinism_rows)
    df_stab_out = pd.DataFrame(stability_rows)
    df_ens_out = pd.DataFrame(ensemble_rows)

    df_det_out.to_csv(REPORTS_TABLES / "t_determinism.csv", index=False)
    df_stab_out.to_csv(REPORTS_TABLES / "t_stability.csv", index=False)
    df_ens_out.to_csv(REPORTS_TABLES / "t_ensemble.csv", index=False)

    print(f"\n[SAVED] {REPORTS_TABLES / 't_determinism.csv'}")
    print(f"[SAVED] {REPORTS_TABLES / 't_stability.csv'}")
    print(f"[SAVED] {REPORTS_TABLES / 't_ensemble.csv'}")

    # Plot Stability Heatmap (Flip Rate across perturbations)
    df_jev_stab = df_stab_out[df_stab_out["model"] == "Jev (H1n)"]
    pivot_flips = df_jev_stab.pivot_table(index="dataset", columns="perturbation", values="flip_rate_pct")
    pivot_flips = pivot_flips.reindex(columns=["P1", "P2", "P3", "P4", "P5", "P6", "P7"], index=EM_DATASETS)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    cax = ax.matshow(pivot_flips.values, cmap="YlOrRd", vmin=0, vmax=15)
    fig.colorbar(cax)

    ax.set_xticks(range(7))
    ax.set_xticklabels(["P1 (Swap)", "P2 (Shuffle)", "P3 (Para 1)", "P4 (Para 2)", "P5 (Rev Choice)", "P6 (Neutral)", "P7 (Text)"], fontsize=10)
    ax.set_yticks(range(4))
    ax.set_yticklabels([ds.upper() for ds in EM_DATASETS], fontsize=11, fontweight="bold")
    ax.set_title("Decision Flip Rate under Input Perturbations (%) [Jev-H1n]", fontsize=13, fontweight="bold", pad=20)

    for i in range(4):
        for j in range(7):
            val = pivot_flips.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color="black" if val < 8 else "white", fontsize=10.5, fontweight="bold")

    fig.tight_layout()
    fig.savefig(REPORTS_FIGS / "f_stability_heatmap.png")
    fig.savefig(REPORTS_FIGS / "f_stability_heatmap.pdf")
    plt.close(fig)
    print(f"[SAVED] Heatmap in {REPORTS_FIGS}")

    print("\n" + "=" * 70)
    print("E5 STABILITY EXPERIMENT COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    run_e5()
