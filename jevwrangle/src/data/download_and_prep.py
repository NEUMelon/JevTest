import os
import sys
import json
import tarfile
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

BASE_DIR = Path("/root/autodl-tmp/jevwrangle")
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_CANONICAL = BASE_DIR / "data" / "canonical"
DATA_DESIGN = BASE_DIR / "data" / "design"
DATA_POOLS = BASE_DIR / "data" / "pools"
DATA_BUDGETS = BASE_DIR / "data" / "budgets"
REPORTS_TABLES = BASE_DIR / "reports" / "tables"

def ensure_dirs():
    for d in [DATA_RAW, DATA_CANONICAL, DATA_DESIGN, DATA_POOLS, DATA_BUDGETS, REPORTS_TABLES]:
        d.mkdir(parents=True, exist_ok=True)

def download_file(url, target_path):
    if not target_path.exists():
        print(f"Downloading {url} -> {target_path}...")
        urllib.request.urlretrieve(url, target_path)
        print(f"Downloaded {target_path.name}")
    else:
        print(f"File {target_path.name} already exists, skipping.")

def download_raw_data():
    ensure_dirs()
    # 1. FM-Data-Tasks S3 tar.gz
    s3_tar = DATA_RAW / "datasets.tar.gz"
    download_file("https://fm-data-tasks.s3.us-west-1.amazonaws.com/datasets.tar.gz", s3_tar)
    
    extracted_flag = DATA_RAW / "datasets"
    if not extracted_flag.exists():
        print("Extracting datasets.tar.gz...")
        with tarfile.open(s3_tar, "r:gz") as tar:
            tar.extractall(DATA_RAW)
        print("Extracted datasets.tar.gz")

    # 2. Abt-Buy from Ditto repo
    abt_buy_dir = DATA_RAW / "Abt-Buy"
    abt_buy_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train.txt", "valid.txt", "test.txt"]:
        fpath = abt_buy_dir / split
        url = f"https://raw.githubusercontent.com/megagonlabs/ditto/master/data/er_magellan/Textual/Abt-Buy/{split}"
        download_file(url, fpath)

    # 3. Flights from Raha repo
    flights_dir = DATA_RAW / "Flights"
    flights_dir.mkdir(parents=True, exist_ok=True)
    for fname in ["dirty.csv", "clean.csv"]:
        fpath = flights_dir / fname
        url = f"https://raw.githubusercontent.com/BigDaMa/raha/master/datasets/flights/{fname}"
        download_file(url, fpath)

def parse_ditto_txt(filepath):
    """解析 Ditto 格式的 txt 文件: COL <attr> VAL <val> ... \t COL <attr> VAL <val> ... \t <label>"""
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            rec_a_str, rec_b_str, label_str = parts[0], parts[1], parts[2]
            
            def parse_rec(s):
                rec = {}
                tokens = s.split("COL ")
                for tok in tokens:
                    if not tok.strip():
                        continue
                    if " VAL " in tok:
                        attr, val = tok.split(" VAL ", 1)
                        rec[attr.strip().lower()] = val.strip()
                return rec
                
            rec_a = parse_rec(rec_a_str)
            rec_b = parse_rec(rec_b_str)
            label = int(label_str)
            pairs.append((idx, idx, rec_a, rec_b, label))
    return pairs

def prep_structured_em(ds_code, ds_folder):
    """处理带有 tableA, tableB, train.csv, valid.csv, test.csv 的标准 EM 数据集"""
    print(f"Processing structured EM: {ds_code} from {ds_folder}...")
    folder = DATA_RAW / "datasets" / "entity_matching" / "structured" / ds_folder
    tableA = pd.read_csv(folder / "tableA.csv")
    tableB = pd.read_csv(folder / "tableB.csv")
    
    # 建立 id -> record dict
    def clean_record(row):
        d = {}
        for col, val in row.items():
            if col == "id":
                continue
            col_lower = str(col).lower()
            if pd.isna(val) or val is None or str(val).strip() == "" or str(val).lower() == "nan":
                d[col_lower] = None
            else:
                d[col_lower] = str(val).strip()
        return d

    id_to_a = {row["id"]: clean_record(row) for _, row in tableA.iterrows()}
    id_to_b = {row["id"]: clean_record(row) for _, row in tableB.iterrows()}

    out_dir = DATA_CANONICAL / ds_code
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "valid", "test"]:
        split_file = folder / f"{split}.csv"
        df_split = pd.read_csv(split_file)
        jsonl_path = out_dir / f"{split}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f_out:
            for idx, row in df_split.iterrows():
                l_id = row["ltable_id"]
                r_id = row["rtable_id"]
                label = int(row["label"])
                item = {
                    "pair_id": f"{ds_code}-{split}-{idx:06d}",
                    "dataset": ds_code,
                    "split": split,
                    "ltable_id": int(l_id),
                    "rtable_id": int(r_id),
                    "record_a": id_to_a.get(l_id, {}),
                    "record_b": id_to_b.get(r_id, {}),
                    "label": label
                }
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  Wrote {jsonl_path} ({len(df_split)} pairs)")

def prep_abt_buy():
    """处理 Abt-Buy 数据集"""
    print("Processing Abt-Buy (ab)...")
    folder = DATA_RAW / "Abt-Buy"
    out_dir = DATA_CANONICAL / "ab"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "valid", "test"]:
        txt_path = folder / f"{split}.txt"
        pairs = parse_ditto_txt(txt_path)
        jsonl_path = out_dir / f"{split}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f_out:
            for idx, (lid, rid, ra, rb, label) in enumerate(pairs):
                item = {
                    "pair_id": f"ab-{split}-{idx:06d}",
                    "dataset": "ab",
                    "split": split,
                    "ltable_id": lid,
                    "rtable_id": rid,
                    "record_a": ra,
                    "record_b": rb,
                    "label": label
                }
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  Wrote {jsonl_path} ({len(pairs)} pairs)")

def prep_hospital():
    """处理 Hospital 错误检测数据集 (Narayan 协议)"""
    print("Processing Hospital (ho)...")
    folder = DATA_RAW / "datasets" / "error_detection" / "Hospital"
    if not folder.exists():
        print(f"Folder {folder} not found, skipping.")
        return
    table = pd.read_csv(folder / "table.csv")
    out_dir = DATA_CANONICAL / "ho"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 计算全局列 profile (前 5 常见值)
    col_profiles = {}
    for col in table.columns:
        vals = table[col].dropna().astype(str).tolist()
        top5 = [k for k, _ in Counter(vals).most_common(5)]
        sample5 = vals[:5] if len(vals) >= 5 else vals
        col_profiles[col.lower()] = {"top_values": top5, "example_values": sample5}

    for split in ["train", "valid", "test"]:
        split_file = folder / f"{split}.csv"
        if not split_file.exists():
            continue
        df_split = pd.read_csv(split_file)
        jsonl_path = out_dir / f"{split}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f_out:
            for idx, row in df_split.iterrows():
                row_id = int(row["row_id"])
                col_name = str(row["col_name"]).lower()
                is_clean = int(row["is_clean"])
                row_data = {str(k).lower(): (str(v) if pd.notna(v) else None) for k, v in table.iloc[row_id].items()}
                item = {
                    "cell_id": f"ho-{split}-{row_id:04d}-{col_name}",
                    "dataset": "ho",
                    "split": split,
                    "row_id": row_id,
                    "col": col_name,
                    "value": row_data.get(col_name),
                    "row": row_data,
                    "column_profile": col_profiles.get(col_name, {}),
                    "label_error": 1 if is_clean == 0 else 0
                }
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  Wrote {jsonl_path} ({len(df_split)} cells)")

def prep_adult():
    """处理 Adult 错误检测数据集 (Narayan 协议)"""
    print("Processing Adult (ad)...")
    folder = DATA_RAW / "datasets" / "error_detection" / "Adult"
    if not folder.exists():
        print(f"Folder {folder} not found, skipping.")
        return
    out_dir = DATA_CANONICAL / "ad"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "valid", "test"]:
        split_file = folder / f"{split}.csv"
        if not split_file.exists():
            continue
        df_split = pd.read_csv(split_file)
        feature_cols = [c for c in df_split.columns if c not in ["Unnamed: 0", "", "col_name", "is_clean"]]
        
        col_profiles = {}
        for col in feature_cols:
            vals = df_split[col].dropna().astype(str).tolist()
            top5 = [k for k, _ in Counter(vals).most_common(5)]
            sample5 = vals[:5] if len(vals) >= 5 else vals
            col_profiles[col.lower()] = {"top_values": top5, "example_values": sample5}

        jsonl_path = out_dir / f"{split}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f_out:
            for idx, row in df_split.iterrows():
                col_name = str(row["col_name"]).lower()
                is_clean = int(row["is_clean"])
                row_data = {str(k).lower(): (str(v) if pd.notna(v) else None) for k, v in row[feature_cols].items()}
                item = {
                    "cell_id": f"ad-{split}-{idx:06d}-{col_name}",
                    "dataset": "ad",
                    "split": split,
                    "row_id": idx,
                    "col": col_name,
                    "value": row_data.get(col_name),
                    "row": row_data,
                    "column_profile": col_profiles.get(col_name, {}),
                    "label_error": 1 if is_clean == 0 else 0
                }
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  Wrote {jsonl_path} ({len(df_split)} cells)")

def prep_flights():
    """处理 Flights 错误检测数据集 (Raha 协议)"""
    print("Processing Flights (fl)...")
    folder = DATA_RAW / "Flights"
    if not (folder / "dirty.csv").exists() or not (folder / "clean.csv").exists():
        print(f"Flights dirty/clean csv not found, skipping.")
        return
    df_dirty = pd.read_csv(folder / "dirty.csv")
    df_clean = pd.read_csv(folder / "clean.csv")

    cols = [c for c in df_dirty.columns if c != "tuple_id"]
    n_rows = len(df_dirty)

    np.random.seed(20260929)
    indices = np.arange(n_rows)
    np.random.shuffle(indices)
    split_point = int(n_rows * 0.2)
    train_idx = set(indices[:split_point])
    test_idx = set(indices[split_point:])

    fl_b_dir = DATA_BUDGETS / "fl"
    fl_b_dir.mkdir(parents=True, exist_ok=True)
    with open(fl_b_dir / "split_rows.json", "w", encoding="utf-8") as f:
        json.dump({"train_rows": [int(x) for x in sorted(list(train_idx))], "test_rows": [int(x) for x in sorted(list(test_idx))]}, f, indent=2)

    col_profiles = {}
    for col in cols:
        vals = df_dirty[col].dropna().astype(str).tolist()
        top5 = [k for k, _ in Counter(vals).most_common(5)]
        sample5 = vals[:5] if len(vals) >= 5 else vals
        col_profiles[col.lower()] = {"top_values": top5, "example_values": sample5}

    out_dir = DATA_CANONICAL / "fl"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_set in [("train", train_idx), ("test", test_idx)]:
        jsonl_path = out_dir / f"{split_name}.jsonl"
        count = 0
        with open(jsonl_path, "w", encoding="utf-8") as f_out:
            for r_i in sorted(list(split_set)):
                row_dirty = df_dirty.iloc[r_i]
                row_clean = df_clean.iloc[r_i]
                row_dict = {str(c).lower(): (str(row_dirty[c]) if pd.notna(row_dirty[c]) else None) for c in cols}
                for c in cols:
                    v_d = row_dirty[c]
                    v_c = row_clean[c]
                    s_d = "" if pd.isna(v_d) else str(v_d).strip()
                    s_c = "" if pd.isna(v_c) else str(v_c).strip()
                    is_err = 1 if s_d != s_c else 0
                    c_lower = str(c).lower()
                    item = {
                        "cell_id": f"fl-{split_name}-{r_i:04d}-{c_lower}",
                        "dataset": "fl",
                        "split": split_name,
                        "row_id": int(r_i),
                        "col": c_lower,
                        "value": row_dict.get(c_lower),
                        "row": row_dict,
                        "column_profile": col_profiles.get(c_lower, {}),
                        "label_error": is_err
                    }
                    f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                    count += 1
        print(f"  Wrote {jsonl_path} ({count} cells)")

def create_pools_and_budgets(ds_code):
    """创建预算池 (pool2000) 与分层预算抽样 (b=50, 200, 1000)"""
    train_path = DATA_CANONICAL / ds_code / "train.jsonl"
    if not train_path.exists():
        return
    with open(train_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    df = pd.DataFrame(records)
    # 分层抽取 2000 对作为 pool2000
    pos_df = df[df["label"] == 1]
    neg_df = df[df["label"] == 0]
    total_len = len(df)
    pool_size = min(2000, total_len)
    pos_ratio = len(pos_df) / total_len
    pos_target = int(round(pool_size * pos_ratio))
    neg_target = pool_size - pos_target

    np.random.seed(20260929)
    pool_pos = pos_df.sample(n=min(pos_target, len(pos_df)), random_state=20260929)
    pool_neg = neg_df.sample(n=min(neg_target, len(neg_df)), random_state=20260929)
    pool_df = pd.concat([pool_pos, pool_neg]).sample(frac=1.0, random_state=20260929)

    pool_path = DATA_POOLS / f"{ds_code}_pool2000.jsonl"
    pool_records = pool_df.to_dict(orient="records")
    with open(pool_path, "w", encoding="utf-8") as f:
        for r in pool_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Created {pool_path} with {len(pool_records)} items (pos: {len(pool_pos)})")

    # 分层抽取 budgets: b in {50, 200, 1000}
    b_dir = DATA_BUDGETS / ds_code
    b_dir.mkdir(parents=True, exist_ok=True)

    pool_pos_recs = [r for r in pool_records if r["label"] == 1]
    pool_neg_recs = [r for r in pool_records if r["label"] == 0]

    for b, num_seeds in [(50, 10), (200, 10), (1000, 5)]:
        if b > len(pool_records):
            continue
        for s in range(num_seeds):
            rng = np.random.RandomState(s)
            target_pos = max(5, int(round(b * pos_ratio)))
            target_neg = b - target_pos
            
            chosen_pos = rng.choice(len(pool_pos_recs), size=min(target_pos, len(pool_pos_recs)), replace=False)
            chosen_neg = rng.choice(len(pool_neg_recs), size=min(target_neg, len(pool_neg_recs)), replace=False)
            
            sampled_items = [pool_pos_recs[i]["pair_id"] for i in chosen_pos] + [pool_neg_recs[i]["pair_id"] for i in chosen_neg]
            rng.shuffle(sampled_items)

            out_f = b_dir / f"b{b}_seed{s}.json"
            with open(out_f, "w", encoding="utf-8") as f:
                json.dump({"budget": b, "seed": s, "pair_ids": sampled_items}, f, indent=2)

def create_design_set(ds_code):
    """从 valid.csv 抽取 200 对设计集，正例占比提升至 30%"""
    valid_path = DATA_CANONICAL / ds_code / "valid.jsonl"
    if not valid_path.exists():
        return
    with open(valid_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    
    pos_recs = [r for r in records if r["label"] == 1]
    neg_recs = [r for r in records if r["label"] == 0]
    
    rng = np.random.RandomState(20260929)
    target_pos = min(60, len(pos_recs))
    target_neg = min(140, len(neg_recs))

    chosen_pos = [pos_recs[i] for i in rng.choice(len(pos_recs), size=target_pos, replace=False)]
    chosen_neg = [neg_recs[i] for i in rng.choice(len(neg_recs), size=target_neg, replace=False)]
    
    design_recs = chosen_pos + chosen_neg
    rng.shuffle(design_recs)

    design_path = DATA_DESIGN / f"{ds_code}.jsonl"
    with open(design_path, "w", encoding="utf-8") as f:
        for r in design_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Created design set {design_path} with {len(design_recs)} pairs (pos: {len(chosen_pos)})")

def verify_and_generate_stats():
    """数据校验与统计表生成 (4.4 节验收标准)"""
    print("\nVerifying datasets and generating reports/tables/data_stats.csv...")
    stats = []
    em_datasets = ["wa", "ag", "da", "ab"]
    
    for ds in em_datasets:
        ds_dir = DATA_CANONICAL / ds
        if not ds_dir.exists():
            continue
        for split in ["train", "valid", "test"]:
            fpath = ds_dir / f"{split}.jsonl"
            if not fpath.exists():
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
            total = len(rows)
            pos = sum(1 for r in rows if r["label"] == 1)
            pos_rate = pos / total if total > 0 else 0.0

            # 统计属性缺失率与平均长度
            attr_stats = {}
            if total > 0:
                sample_attrs = set(rows[0]["record_a"].keys()) | set(rows[0]["record_b"].keys())
                for attr in sample_attrs:
                    missing_cnt = 0
                    total_len = 0
                    val_cnt = 0
                    for r in rows:
                        for side in ["record_a", "record_b"]:
                            v = r[side].get(attr)
                            if v is None or str(v).strip() == "":
                                missing_cnt += 1
                            else:
                                val_cnt += 1
                                total_len += len(str(v))
                    total_slots = total * 2
                    missing_rate = missing_cnt / total_slots
                    avg_len = total_len / val_cnt if val_cnt > 0 else 0
                    attr_stats[f"{attr}_miss%"] = round(missing_rate * 100, 1)
                    attr_stats[f"{attr}_len"] = round(avg_len, 1)

            entry = {
                "dataset": ds,
                "task": "EM",
                "split": split,
                "total_items": total,
                "positive_items": pos,
                "positive_rate%": round(pos_rate * 100, 2),
            }
            entry.update(attr_stats)
            stats.append(entry)

    # 错误检测统计
    ed_datasets = ["ho", "ad", "fl"]
    for ds in ed_datasets:
        ds_dir = DATA_CANONICAL / ds
        if not ds_dir.exists():
            continue
        for split in ["train", "valid", "test"]:
            fpath = ds_dir / f"{split}.jsonl"
            if not fpath.exists():
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
            total = len(rows)
            pos = sum(1 for r in rows if r["label_error"] == 1)
            pos_rate = pos / total if total > 0 else 0.0
            entry = {
                "dataset": ds,
                "task": "ED",
                "split": split,
                "total_items": total,
                "positive_items": pos,
                "positive_rate%": round(pos_rate * 100, 2),
            }
            stats.append(entry)

    df_stats = pd.DataFrame(stats)
    out_csv = REPORTS_TABLES / "data_stats.csv"
    df_stats.to_csv(out_csv, index=False)
    print(f"Generated {out_csv}:")
    print(df_stats[["dataset", "task", "split", "total_items", "positive_items", "positive_rate%"]].to_string())

if __name__ == "__main__":
    download_raw_data()
    prep_structured_em("wa", "Walmart-Amazon")
    prep_structured_em("ag", "Amazon-Google")
    prep_structured_em("da", "DBLP-ACM")
    prep_abt_buy()
    prep_hospital()
    prep_adult()
    prep_flights()

    for ds in ["wa", "ag", "da", "ab"]:
        create_pools_and_budgets(ds)
        create_design_set(ds)

    verify_and_generate_stats()
