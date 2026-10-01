"""
Magellan feature extractor and evaluation for E3.
Computes standard entity matching similarity features across all shared attributes:
- 3-gram Jaccard
- Token Jaccard
- Token Levenshtein / Normalized Levenshtein
- Jaro-Winkler
- Exact match (binary)
- Numerical: absolute difference, relative difference, missing indicator
"""

import re, math
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein, JaroWinkler

def qgrams(s, q=3):
    s = str(s or "").lower().strip()
    if len(s) < q:
        return {s} if s else set()
    return {s[i:i+q] for i in range(len(s) - q + 1)}

def qgram_jaccard(s1, s2, q=3):
    q1 = qgrams(s1, q)
    q2 = qgrams(s2, q)
    if not q1 or not q2:
        return 0.0
    return len(q1 & q2) / len(q1 | q2)

def token_jaccard(s1, s2):
    t1 = set(re.findall(r"\w+", str(s1 or "").lower()))
    t2 = set(re.findall(r"\w+", str(s2 or "").lower()))
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)

def norm_lev(s1, s2):
    s1, s2 = str(s1 or ""), str(s2 or "")
    if not s1 and not s2:
        return 1.0
    return Levenshtein.normalized_similarity(s1, s2)

def jaro_winkler(s1, s2):
    s1, s2 = str(s1 or ""), str(s2 or "")
    if not s1 and not s2:
        return 1.0
    return JaroWinkler.normalized_similarity(s1, s2)

def extract_price(p):
    if p is None:
        return None
    try:
        clean = re.sub(r"[^\d.]", "", str(p))
        val = float(clean)
        return val if val > 0 else None
    except Exception:
        return None

def compute_magellan_features_for_pair(a, b, ds):
    feat = {}
    
    # Common text attributes by dataset
    if ds == "wa":
        text_attrs = ["title", "category", "brand", "modelno"]
        num_attrs = ["price"]
    elif ds == "ag":
        text_attrs = ["title", "manufacturer"]
        num_attrs = ["price"]
    elif ds == "da":
        text_attrs = ["title", "authors", "venue"]
        num_attrs = ["year"]
    elif ds == "ab":
        text_attrs = ["name", "description"]
        num_attrs = ["price"]
    else:
        raise ValueError(f"Unknown dataset: {ds}")
        
    for attr in text_attrs:
        v1 = a.get(attr, "")
        v2 = b.get(attr, "")
        feat[f"m_{attr}_3gram_jaccard"] = qgram_jaccard(v1, v2, 3)
        feat[f"m_{attr}_token_jaccard"] = token_jaccard(v1, v2)
        feat[f"m_{attr}_levenshtein"]   = norm_lev(v1, v2)
        feat[f"m_{attr}_jarowinkler"]   = jaro_winkler(v1, v2)
        feat[f"m_{attr}_exact"]         = 1.0 if (v1 and v2 and str(v1).strip().lower() == str(v2).strip().lower()) else 0.0
        feat[f"m_{attr}_missing"]       = 1.0 if (not v1 or not v2) else 0.0

    for attr in num_attrs:
        v1 = extract_price(a.get(attr))
        v2 = extract_price(b.get(attr))
        if v1 is not None and v2 is not None:
            feat[f"m_{attr}_abs_diff"] = abs(v1 - v2)
            feat[f"m_{attr}_rel_diff"] = abs(v1 - v2) / max(v1, v2, 1e-4)
            feat[f"m_{attr}_missing"]  = 0.0
        else:
            feat[f"m_{attr}_abs_diff"] = 0.0
            feat[f"m_{attr}_rel_diff"] = 0.0
            feat[f"m_{attr}_missing"]  = 1.0
            
    return feat

def compute_magellan_features(ds, pairs):
    rows = [compute_magellan_features_for_pair(p["record_a"], p["record_b"], ds) for p in pairs]
    return pd.DataFrame(rows)

if __name__ == "__main__":
    import json
    with open("data/canonical/wa/test.jsonl", encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f][:5]
    df = compute_magellan_features("wa", pairs)
    print("WA Magellan features sample:")
    print(df.iloc[0].to_dict())
