"""
E2: Attribution Matrix (RQ1 核心大表)
Systems: J-H1n(0-label), J-H1+LR, J-D0, J-D+LR, J-D+C+LR, C+LR,
         L1-H1(0-label), L1-H1+LR, L2-H1(0-label), L2-H1+LR
Budgets: 0 (J-D0/J-H1n/L*-H1), 50, 200, 1000, full (Jev only)
Seeds:   b=50/200 → 10 seeds; b=1000 → 5 seeds; full → 3 seeds
Output:  reports/tables/t_attribution.csv
"""

import os, sys, json, time, yaml, re, math
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.clients.systemone import SystemOneClient
from src.clients.llm import LLMClient
from src.clients.ledger import Ledger

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_CANONICAL = BASE_DIR / "data" / "canonical"
DATA_POOLS     = BASE_DIR / "data" / "pools"
DATA_BUDGETS   = BASE_DIR / "data" / "budgets"
QUESTIONS_DIR  = BASE_DIR / "questions"
REPORTS_TABLES = BASE_DIR / "reports" / "tables"
RUNS_DIR       = BASE_DIR / "runs"

EM_DATASETS = ["wa", "ag", "da", "ab"]

POLARITIES = {
    "wa": {
        "same_brand": 1, "same_model_id": 1, "same_product_type": 1,
        "variant_mismatch": -1, "accessory_bundle": -1, "title_same_item": 1
    },
    "ag": {
        "same_manufacturer": 1, "same_product_line": 1,
        "version_mismatch": -1, "edition_mismatch": -1,
        "license_mismatch": -1, "platform_mismatch": -1, "accessory_bundle": -1
    },
    "da": {
        "same_title": 1, "same_authors": 1, "same_venue": 1
    },
    "ab": {
        "same_brand": 1, "same_model_id": 1, "same_product_type": 1,
        "variant_mismatch": -1, "accessory_bundle": -1, "name_same_item": 1
    },
}

# ============================================================
# Utility helpers
# ============================================================

EPS = 1e-4

def safe_logit(arr):
    arr = np.clip(np.asarray(arr, dtype=float), EPS, 1 - EPS)
    return np.log(arr / (1 - arr))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))

def norm_str(s):
    """Lowercase + strip spaces/hyphens/slashes/dots."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    return re.sub(r"[\s\-_/.]", "", str(s).lower().strip())

def word_jaccard(s1, s2):
    a = set(str(s1 or "").lower().split())
    b = set(str(s2 or "").lower().split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def jaro_winkler(s1, s2):
    try:
        from rapidfuzz.distance import JaroWinkler
        return JaroWinkler.normalized_similarity(str(s1 or ""), str(s2 or ""))
    except ImportError:
        return float(s1 == s2) if s1 and s2 else 0.0

def price_feats(p1, p2):
    """Return (|log(p1/p2)|, is_missing)."""
    try:
        v1 = float(re.sub(r"[,$]", "", str(p1).strip()))
        v2 = float(re.sub(r"[,$]", "", str(p2).strip()))
        if v1 > 0 and v2 > 0:
            return abs(math.log(v1 / v2)), 0
        return 0.0, 1
    except Exception:
        return 0.0, 1

def extract_version(s):
    if not s:
        return None
    m = re.search(r"\bv?(\d+(?:\.\d+)+)\b", str(s), re.IGNORECASE)
    return m.group(1) if m else None

def extract_model_candidates(s):
    tokens = re.findall(r"[a-zA-Z0-9]{4,}", str(s or ""))
    return {norm_str(t) for t in tokens if re.search(r"[a-zA-Z]", t) and re.search(r"\d", t)}

def author_lastnames(s):
    if not s:
        return set()
    lastnames = set()
    for auth in str(s).split(","):
        parts = auth.strip().split()
        if parts:
            lastnames.add(norm_str(parts[-1]))
    return lastnames

# ============================================================
# Code Features (per dataset)
# ============================================================

def compute_code_features(ds, pairs):
    rows = []
    if ds == "wa":
        for p in pairs:
            a, b = p["record_a"], p["record_b"]
            pr, pm = price_feats(a.get("price"), b.get("price"))
            mn_a, mn_b = norm_str(a.get("modelno")), norm_str(b.get("modelno"))
            br_a, br_b = norm_str(a.get("brand")),   norm_str(b.get("brand"))
            rows.append({
                "c_brand_equal":    1.0 if (br_a and br_b and br_a == br_b) else 0.0,
                "c_modelno_equal":  1.0 if (mn_a and mn_b and mn_a == mn_b) else 0.0,
                "c_modelno_jw":     jaro_winkler(mn_a, mn_b) if mn_a and mn_b else 0.0,
                "c_title_jaccard":  word_jaccard(a.get("title"), b.get("title")),
                "c_price_log_ratio": pr,
                "c_price_missing":  float(pm),
            })

    elif ds == "ag":
        for p in pairs:
            a, b = p["record_a"], p["record_b"]
            pr, pm = price_feats(a.get("price"), b.get("price"))
            ver_a, ver_b = extract_version(a.get("title")), extract_version(b.get("title"))
            mfr_a, mfr_b = norm_str(a.get("manufacturer")), norm_str(b.get("manufacturer"))
            if ver_a is None or ver_b is None:
                ve, vn, vm = 0.0, 0.0, 1.0
            elif ver_a == ver_b:
                ve, vn, vm = 1.0, 0.0, 0.0
            else:
                ve, vn, vm = 0.0, 1.0, 0.0
            rows.append({
                "c_title_jaccard":      word_jaccard(a.get("title"), b.get("title")),
                "c_version_equal":      ve,
                "c_version_mismatch":   vn,
                "c_version_missing":    vm,
                "c_price_log_ratio":    pr,
                "c_price_missing":      float(pm),
                "c_manufacturer_equal": 1.0 if (mfr_a and mfr_b and mfr_a == mfr_b) else 0.0,
            })

    elif ds == "da":
        for p in pairs:
            a, b = p["record_a"], p["record_b"]
            try:
                yr_eq = float(int(str(a.get("year","")).strip()) == int(str(b.get("year","")).strip()))
            except Exception:
                yr_eq = 0.0
            ln_a, ln_b = author_lastnames(a.get("authors")), author_lastnames(b.get("authors"))
            auth_j = len(ln_a & ln_b) / len(ln_a | ln_b) if (ln_a and ln_b) else 0.0
            vn_a, vn_b = norm_str(a.get("venue")), norm_str(b.get("venue"))
            rows.append({
                "c_year_equal":              yr_eq,
                "c_title_jaccard":           word_jaccard(a.get("title"), b.get("title")),
                "c_author_lastname_jaccard": auth_j,
                "c_venue_equal":             1.0 if (vn_a and vn_b and vn_a == vn_b) else 0.0,
            })

    elif ds == "ab":
        for p in pairs:
            a, b = p["record_a"], p["record_b"]
            pr, pm = price_feats(a.get("price"), b.get("price"))
            cands_a = extract_model_candidates(str(a.get("name","")) + " " + str(a.get("description","")))
            cands_b = extract_model_candidates(str(b.get("name","")) + " " + str(b.get("description","")))
            model_j = len(cands_a & cands_b) / len(cands_a | cands_b) if (cands_a and cands_b) else 0.0
            rows.append({
                "c_model_id_jaccard": model_j,
                "c_price_log_ratio":  pr,
                "c_price_missing":    float(pm),
                "c_name_jaccard":     word_jaccard(a.get("name"), b.get("name")),
            })

    else:
        raise ValueError(f"Unknown dataset: {ds}")

    return pd.DataFrame(rows)


# ============================================================
# Jev Decomposed Features
# ============================================================

def _jev_decomposed_one(p, decomposed_q, jev_client):
    state = {"record_a": p["record_a"], "record_b": p["record_b"]}
    resp, _ = jev_client.ask(state, decomposed_q)
    ans = resp.get("answers", {})
    return {qid: float(qans.get("noul", 0.5))
            for qid, qans in ans.items() if isinstance(qans, dict)}


def compute_jev_decomposed(ds, pairs, jev_client, label="", max_workers=12):
    """Returns DataFrame with columns jev_{qid} for each question."""
    with open(QUESTIONS_DIR / ds / "decomposed.yaml", encoding="utf-8") as f:
        decomposed_q = yaml.safe_load(f)
    q_ids = list(decomposed_q.keys())

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {ex.submit(_jev_decomposed_one, p, decomposed_q, jev_client): i
                   for i, p in enumerate(pairs)}
        done, total, t0 = 0, len(pairs), time.perf_counter()
        for fut in as_completed(fut_map):
            idx = fut_map[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                print(f"  [WARN] Jev-D pair {idx}: {e}", flush=True)
                results[idx] = {}
            done += 1
            if done % 200 == 0 or done == total:
                elapsed = time.perf_counter() - t0
                print(f"  Jev-D {ds}{label}: {done}/{total}  ({done/max(elapsed,0.01):.0f} p/s)", flush=True)

    rows = [{f"jev_{qid}": results[i].get(qid, 0.5) for qid in q_ids}
            for i in range(len(pairs))]
    return pd.DataFrame(rows)


# ============================================================
# Jev Holistic Features (for pool / full-train)
# ============================================================

def _jev_holistic_one(p, holistic_q, jev_client):
    from src.experiments.e1_holistic import extract_noul
    state = {"record_a": p["record_a"], "record_b": p["record_b"]}
    resp, _ = jev_client.ask(state, holistic_q)
    ans = resp.get("answers", {})
    h1n_ans = ans.get("H1_noul", {})
    return float(extract_noul(h1n_ans)) if isinstance(h1n_ans, dict) else 0.5


def compute_jev_holistic_p_h1n(ds, pairs, jev_client, label="", max_workers=12):
    with open(QUESTIONS_DIR / ds / "holistic.yaml", encoding="utf-8") as f:
        holistic_q = yaml.safe_load(f)
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {ex.submit(_jev_holistic_one, p, holistic_q, jev_client): i
                   for i, p in enumerate(pairs)}
        done, total, t0 = 0, len(pairs), time.perf_counter()
        for fut in as_completed(fut_map):
            idx = fut_map[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                print(f"  [WARN] Jev-H pair {idx}: {e}", flush=True)
                results[idx] = 0.5
            done += 1
            if done % 200 == 0 or done == total:
                elapsed = time.perf_counter() - t0
                print(f"  Jev-H {ds}{label}: {done}/{total}  ({done/max(elapsed,0.01):.0f} p/s)", flush=True)
    return pd.DataFrame({"p_h1n": [results[i] for i in range(len(pairs))]})


# ============================================================
# LLM Holistic Features (for pool2000)
# ============================================================

_LLM_HOL_SYSTEM = "You are a careful data analyst. Evaluate whether two records refer to the same real-world entity."

def _llm_holistic_one(p, luna_client, ds_client):
    from src.experiments.e1_holistic import extract_llm_prob
    content = (
        f"Record A: {json.dumps(p['record_a'], ensure_ascii=False)}\n"
        f"Record B: {json.dumps(p['record_b'], ensure_ascii=False)}\n\n"
        "Do Record A and Record B refer to the same real-world entity? "
        'Output strictly as JSON: {"answer": "yes" or "no", "match_probability": <0.0-1.0>}'
    )
    messages = [{"role": "system", "content": _LLM_HOL_SYSTEM},
                {"role": "user",   "content": content}]

    luna_resp, _ = luna_client.ask(messages, max_tokens=700, reasoning_effort="low")
    p_luna = extract_llm_prob(luna_resp["choices"][0]["message"]["content"])

    ds_resp, _ = ds_client.ask(messages, max_tokens=300)
    p_deepseek = extract_llm_prob(ds_resp["choices"][0]["message"]["content"])

    return p_luna, p_deepseek


def compute_llm_holistic(ds, pairs, luna_client, ds_client, label="", max_workers=8):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {ex.submit(_llm_holistic_one, p, luna_client, ds_client): i
                   for i, p in enumerate(pairs)}
        done, total, t0 = 0, len(pairs), time.perf_counter()
        for fut in as_completed(fut_map):
            idx = fut_map[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                print(f"  [WARN] LLM-H pair {idx}: {e}", flush=True)
                results[idx] = (0.5, 0.5)
            done += 1
            if done % 100 == 0 or done == total:
                elapsed = time.perf_counter() - t0
                print(f"  LLM-H {ds}{label}: {done}/{total}  ({done/max(elapsed,0.01):.0f} p/s)", flush=True)
    return pd.DataFrame({
        "p_luna":     [results[i][0] for i in range(len(pairs))],
        "p_deepseek": [results[i][1] for i in range(len(pairs))],
    })


# ============================================================
# Metric Helpers
# ============================================================

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


# ============================================================
# LR Experiment Core
# ============================================================

def run_lr(train_df, test_df, feature_cols, budget, seed=42):
    from src.recovery.core import fit_aggregator
    model, threshold, _, _ = fit_aggregator(
        train_df[feature_cols].values, train_df['label'].values, budget, seed)
    Xte = test_df[feature_cols].values.astype(float)
    if not np.isfinite(Xte).all():
        raise ValueError('Incomplete test features; use the recovery pipeline coverage report')
    return eval_at_threshold(test_df['label'].values, model.predict_proba(Xte)[:, 1], threshold)


def run_system_lr(system_name, feature_cols, pool_df, test_df,
                  budget_files, full_train_df=None,
                  budgets=(50, 200, 1000), seeds_per_budget=(10, 10, 5)):
    """
    Run LR for a system at each budget, average over seeds.
    Returns list of result dicts.
    """
    ds = str(pool_df["dataset"].iloc[0])
    records = []

    for b, n_seeds in zip(budgets, seeds_per_budget):
        seed_results = []
        for seed_idx in range(n_seeds):
            key = f"b{b}_seed{seed_idx}"
            bf = budget_files.get(key)
            if bf is None:
                print(f"  [WARN] Missing budget file: {key}", flush=True)
                continue
            with open(bf) as f:
                labeled_ids = set(json.load(f)["pair_ids"])

            train_part = pool_df[pool_df["pair_id"].isin(labeled_ids)].copy()
            if len(train_part) == 0:
                continue

            # Ensure feature cols exist
            missing = [c for c in feature_cols if c not in train_part.columns]
            if missing:
                print(f"  [WARN] {system_name}: missing cols {missing}", flush=True)
                continue

            m = run_lr(train_part, test_df, feature_cols, b, seed=seed_idx)
            seed_results.append(m)

        if seed_results:
            avg = {k: round(float(np.mean([r[k] for r in seed_results])), 2)
                   for k in seed_results[0]}
            std_f1 = round(float(np.std([r["f1"] for r in seed_results])), 2)
            records.append({
                "dataset": ds, "system": system_name, "budget": b,
                **avg, "f1_std": std_f1, "n_seeds": len(seed_results)
            })
            print(f"  [{ds}] {system_name} b={b}: F1={avg['f1']:.2f}% ±{std_f1:.2f}  (n={len(seed_results)})", flush=True)

    # Full budget (Jev systems only; LLM has no full-train features)
    if full_train_df is not None:
        missing = [c for c in feature_cols if c not in full_train_df.columns]
        if not missing:
            seed_results = []
            for seed_idx in range(3):
                m = run_lr(full_train_df, test_df, feature_cols, len(full_train_df), seed=seed_idx)
                seed_results.append(m)
            avg = {k: round(float(np.mean([r[k] for r in seed_results])), 2)
                   for k in seed_results[0]}
            std_f1 = round(float(np.std([r["f1"] for r in seed_results])), 2)
            records.append({
                "dataset": ds, "system": system_name, "budget": "full",
                **avg, "f1_std": std_f1, "n_seeds": 3
            })
            print(f"  [{ds}] {system_name} full:  F1={avg['f1']:.2f}% ±{std_f1:.2f}", flush=True)

    return records


# ============================================================
# Cache helpers
# ============================================================

def load_or_compute_csv(cache_path, compute_fn, expected_rows, label=""):
    cache_path = Path(cache_path)
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        if len(df) == expected_rows:
            print(f"  [cache hit] {label} ({len(df)} rows) ← {cache_path.name}", flush=True)
            return df
        print(f"  [cache stale] {label}: expected {expected_rows}, got {len(df)}", flush=True)
    df = compute_fn()
    df.to_csv(cache_path, index=False)
    print(f"  [saved] {label} ({len(df)} rows) → {cache_path.name}", flush=True)
    return df


# ============================================================
# Main
# ============================================================

def load_pairs(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run_e2():
    REPORTS_TABLES.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    ledger     = Ledger(ledger_path=str(RUNS_DIR / "ledger.csv"))
    jev_client = SystemOneClient(model="jev-latest", ledger=ledger)
    luna_client= LLMClient(model="gpt-6-luna",        ledger=ledger)
    dsk_client = LLMClient(model="deepseek-v4-flash", ledger=ledger)

    all_results = []

    print("\n" + "=" * 70, flush=True)
    print("E2: Attribution Matrix — RQ1 core table", flush=True)
    print("=" * 70, flush=True)

    for ds in EM_DATASETS:
        print(f"\n{'=' * 70}", flush=True)
        print(f">>> [DATASET] {ds.upper()}", flush=True)
        print(f"{'=' * 70}", flush=True)

        # ── Data ──────────────────────────────────────────────────────────────
        test_pairs = load_pairs(DATA_CANONICAL / ds / "test.jsonl")
        pool_pairs = load_pairs(DATA_POOLS / f"{ds}_pool2000.jsonl")

        # Full train pairs (for "full" budget setting; only Jev features needed)
        full_train_pairs = load_pairs(DATA_CANONICAL / ds / "train.jsonl")

        n_test, n_pool, n_full = len(test_pairs), len(pool_pairs), len(full_train_pairs)
        print(f"  test={n_test}  pool={n_pool}  full-train={n_full}", flush=True)

        # ── E1 test predictions (holistic) ───────────────────────────────────
        e1_test = pd.read_csv(RUNS_DIR / f"e1_preds_{ds}.csv")
        # columns: pair_id, label, p_h1n, p_h1s, p_h1c, p_h0, p_luna, p_deepseek

        # ── Step 1: Jev Decomposed on test ───────────────────────────────────
        print("\n[Step 1] Jev decomposed features — test set", flush=True)
        jev_d_test = load_or_compute_csv(
            RUNS_DIR / f"e2_jev_d_{ds}_test.csv",
            lambda: compute_jev_decomposed(ds, test_pairs, jev_client, f" test"),
            n_test, f"Jev-D {ds} test"
        )

        # ── Step 2: Jev Decomposed on pool2000 ──────────────────────────────
        print("\n[Step 2] Jev decomposed features — pool2000", flush=True)
        jev_d_pool = load_or_compute_csv(
            RUNS_DIR / f"e2_jev_d_{ds}_pool.csv",
            lambda: compute_jev_decomposed(ds, pool_pairs, jev_client, f" pool"),
            n_pool, f"Jev-D {ds} pool"
        )

        # ── Step 3: Jev Decomposed on full train ─────────────────────────────
        print("\n[Step 3] Jev decomposed features — full train", flush=True)
        jev_d_full = load_or_compute_csv(
            RUNS_DIR / f"e2_jev_d_{ds}_full.csv",
            lambda: compute_jev_decomposed(ds, full_train_pairs, jev_client, f" full"),
            n_full, f"Jev-D {ds} full"
        )

        # ── Step 4: Jev Holistic on pool2000 ────────────────────────────────
        print("\n[Step 4] Jev holistic p_h1n — pool2000", flush=True)
        jev_h_pool = load_or_compute_csv(
            RUNS_DIR / f"e2_jev_h_{ds}_pool.csv",
            lambda: compute_jev_holistic_p_h1n(ds, pool_pairs, jev_client, " pool"),
            n_pool, f"Jev-H {ds} pool"
        )

        # Jev holistic on full train
        print("\n[Step 4b] Jev holistic p_h1n — full train", flush=True)
        jev_h_full = load_or_compute_csv(
            RUNS_DIR / f"e2_jev_h_{ds}_full.csv",
            lambda: compute_jev_holistic_p_h1n(ds, full_train_pairs, jev_client, " full"),
            n_full, f"Jev-H {ds} full"
        )

        # ── Step 5: LLM Holistic on pool2000 ────────────────────────────────
        print("\n[Step 5] LLM holistic features — pool2000", flush=True)
        llm_h_pool = load_or_compute_csv(
            RUNS_DIR / f"e2_llm_h_{ds}_pool.csv",
            lambda: compute_llm_holistic(ds, pool_pairs, luna_client, dsk_client, " pool"),
            n_pool, f"LLM-H {ds} pool"
        )

        # ── Step 6: Code Features ────────────────────────────────────────────
        print("\n[Step 6] Code features", flush=True)
        code_test = compute_code_features(ds, test_pairs)
        code_pool = compute_code_features(ds, pool_pairs)
        code_full = compute_code_features(ds, full_train_pairs)
        code_cols = list(code_test.columns)
        print(f"  code features ({len(code_cols)}): {code_cols}", flush=True)

        # ── Assemble feature DataFrames ──────────────────────────────────────
        q_ids = list(POLARITIES[ds].keys())
        jd_cols = [f"jev_{q}" for q in q_ids]  # decomposed Jev feature cols

        # TEST DataFrame
        test_meta = pd.DataFrame({
            "pair_id": [p["pair_id"] for p in test_pairs],
            "label":   [p["label"]   for p in test_pairs],
            "dataset": ds,
        })
        # IMPORTANT: e1_preds is sorted by pair_id, test.jsonl may not be.
        # Merge on pair_id to ensure correct alignment.
        e1_aligned = e1_test[["pair_id", "p_h1n", "p_luna", "p_deepseek"]]
        test_df_base = test_meta.merge(e1_aligned, on="pair_id", how="left")
        # Append jev_d and code features positionally after sorting both by pair_id
        test_pair_ids = [p["pair_id"] for p in test_pairs]
        jev_d_test_aligned  = jev_d_test.copy()
        jev_d_test_aligned["pair_id"] = test_pair_ids
        code_test_aligned   = code_test.copy()
        code_test_aligned["pair_id"]  = test_pair_ids
        test_df = test_df_base.merge(jev_d_test_aligned,  on="pair_id", how="left") \
                               .merge(code_test_aligned,   on="pair_id", how="left")

        # Add logit columns for LR
        for col in ["p_h1n", "p_luna", "p_deepseek"] + jd_cols:
            if col in test_df.columns:
                test_df[f"logit_{col}"] = safe_logit(test_df[col].values)

        # POOL DataFrame
        pool_meta = pd.DataFrame({
            "pair_id": [p["pair_id"] for p in pool_pairs],
            "label":   [p["label"]   for p in pool_pairs],
            "dataset": ds,
        })
        pool_df = pd.concat([
            pool_meta.reset_index(drop=True),
            jev_h_pool[["p_h1n"]].reset_index(drop=True),
            llm_h_pool[["p_luna", "p_deepseek"]].reset_index(drop=True),
            jev_d_pool.reset_index(drop=True),
            code_pool.reset_index(drop=True),
        ], axis=1)
        for col in ["p_h1n", "p_luna", "p_deepseek"] + jd_cols:
            if col in pool_df.columns:
                pool_df[f"logit_{col}"] = safe_logit(pool_df[col].values)

        # FULL TRAIN DataFrame
        full_meta = pd.DataFrame({
            "pair_id": [p["pair_id"] for p in full_train_pairs],
            "label":   [p["label"]   for p in full_train_pairs],
            "dataset": ds,
        })
        full_df = pd.concat([
            full_meta.reset_index(drop=True),
            jev_h_full[["p_h1n"]].reset_index(drop=True),
            jev_d_full.reset_index(drop=True),
            code_full.reset_index(drop=True),
        ], axis=1)
        for col in ["p_h1n"] + jd_cols:
            if col in full_df.columns:
                full_df[f"logit_{col}"] = safe_logit(full_df[col].values)

        # ── Budget files ─────────────────────────────────────────────────────
        budget_files = {
            bf.stem: bf
            for bf in (DATA_BUDGETS / ds).glob("*.json")
        }

        # ── 0. Zero-label baselines (no LR) ──────────────────────────────────
        print("\n[Systems] Zero-label baselines", flush=True)

        # J-H1n @ 0.5
        m = eval_at_threshold(test_df["label"].values, test_df["p_h1n"].values)
        all_results.append({"dataset": ds, "system": "J-H1n",  "budget": 0, **m, "f1_std": 0.0, "n_seeds": 1})
        print(f"  [{ds}] J-H1n   (0-label @0.5): F1={m['f1']:.2f}%", flush=True)

        # J-D0: polarity-weighted logit sum, threshold 0
        d0 = np.zeros(len(test_df))
        for qid, pol in POLARITIES[ds].items():
            col = f"jev_{qid}"
            if col in test_df.columns:
                d0 += pol * safe_logit(test_df[col].values)
        m = eval_at_threshold(test_df["label"].values, sigmoid(d0))
        all_results.append({"dataset": ds, "system": "J-D0",   "budget": 0, **m, "f1_std": 0.0, "n_seeds": 1})
        print(f"  [{ds}] J-D0    (0-label rule): F1={m['f1']:.2f}%", flush=True)

        # L1-H1 @ 0.5
        m = eval_at_threshold(test_df["label"].values, test_df["p_luna"].values)
        all_results.append({"dataset": ds, "system": "L1-H1",  "budget": 0, **m, "f1_std": 0.0, "n_seeds": 1})
        print(f"  [{ds}] L1-H1   (0-label @0.5): F1={m['f1']:.2f}%", flush=True)

        # L2-H1 @ 0.5
        m = eval_at_threshold(test_df["label"].values, test_df["p_deepseek"].values)
        all_results.append({"dataset": ds, "system": "L2-H1",  "budget": 0, **m, "f1_std": 0.0, "n_seeds": 1})
        print(f"  [{ds}] L2-H1   (0-label @0.5): F1={m['f1']:.2f}%", flush=True)

        # ── LR Systems ──────────────────────────────────────────────────────
        print("\n[Systems] LR experiments (b=50/200/1000 + full)", flush=True)

        lr_systems = [
            # (name, feature_cols, pool_df_for_train, full_df_for_train)
            ("J-H1+LR",   ["logit_p_h1n"],
             pool_df,  full_df),

            ("J-D+LR",    [f"logit_{c}" for c in jd_cols],
             pool_df,  full_df),

            ("J-D+C+LR",  [f"logit_{c}" for c in jd_cols] + code_cols,
             pool_df,  full_df),

            ("C+LR",      code_cols,
             pool_df,  full_df),

            ("L1-H1+LR",  ["logit_p_luna"],
             pool_df,  None),       # no full-train for LLM

            ("L2-H1+LR",  ["logit_p_deepseek"],
             pool_df,  None),
        ]

        for sys_name, feat_cols, p_df, ft_df in lr_systems:
            recs = run_system_lr(
                sys_name, feat_cols, p_df, test_df, budget_files,
                full_train_df=ft_df,
                budgets=(50, 200, 1000),
                seeds_per_budget=(10, 10, 5),
            )
            all_results.extend(recs)

        # ── 50% milestone ────────────────────────────────────────────────────
        # (Already printed per budget above)

        # ── 100% milestone ───────────────────────────────────────────────────
        df_ds = pd.DataFrame([r for r in all_results if r.get("dataset") == ds])
        print(f"\n{'=' * 70}", flush=True)
        print(f">>> [Milestone 100%] {ds.upper()} complete:", flush=True)
        print(f"{'=' * 70}", flush=True)
        if not df_ds.empty:
            pivot = df_ds.pivot_table(index="system", columns="budget",
                                       values="f1", aggfunc="first")
            print(pivot.round(2).to_string(), flush=True)
        print(f"{'=' * 70}\n", flush=True)

        # Save interim results
        pd.DataFrame(all_results).to_csv(
            REPORTS_TABLES / "t_attribution_interim.csv", index=False)

    # ── Final output ─────────────────────────────────────────────────────────
    df_final = pd.DataFrame(all_results)
    out_path = REPORTS_TABLES / "t_attribution.csv"
    df_final.to_csv(out_path, index=False)

    print("\n" + "=" * 70, flush=True)
    print(f"E2 COMPLETE  →  {out_path}", flush=True)
    print("=" * 70, flush=True)
    print(df_final[["dataset", "system", "budget", "f1", "precision", "recall", "auprc"]].to_string(), flush=True)


if __name__ == "__main__":
    run_e2()
