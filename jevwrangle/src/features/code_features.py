import re
import math
import numpy as np
from rapidfuzz import fuzz

def safe_str(val):
    if val is None or val is np.nan:
        return ""
    s = str(val).strip()
    return "" if s.lower() == "nan" else s

def safe_float(val):
    if val is None or val is np.nan:
        return None
    s = str(val).strip().replace("$", "").replace(",", "")
    try:
        return float(s)
    except:
        return None

def word_jaccard(s1, s2):
    w1 = set(re.findall(r"\w+", s1.lower()))
    w2 = set(re.findall(r"\w+", s2.lower()))
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)

def char_ngram_jaccard(s1, s2, n=3):
    s1, s2 = s1.lower(), s2.lower()
    if len(s1) < n or len(s2) < n:
        return float(s1 == s2)
    g1 = set(s1[i:i+n] for i in range(len(s1)-n+1))
    g2 = set(s2[i:i+n] for i in range(len(s2)-n+1))
    return len(g1 & g2) / len(g1 | g2)

def extract_code_features(ds_code, record_a, record_b):
    """
    抽取用于 C+LR 与 J-D+C+LR 的代码数值与字符串特征
    返回字典形式的特征名 -> float 值
    """
    feats = {}
    
    # 共有基础特征: 标题/名称
    title_a = safe_str(record_a.get("title") or record_a.get("name"))
    title_b = safe_str(record_b.get("title") or record_b.get("name"))
    
    feats["title_jaccard"] = word_jaccard(title_a, title_b)
    feats["title_ngram3"] = char_ngram_jaccard(title_a, title_b, 3)
    feats["title_fuzz_ratio"] = fuzz.ratio(title_a.lower(), title_b.lower()) / 100.0
    feats["title_token_set"] = fuzz.token_set_ratio(title_a.lower(), title_b.lower()) / 100.0

    # 价格特征
    p_a = safe_float(record_a.get("price"))
    p_b = safe_float(record_b.get("price"))
    if p_a is not None and p_b is not None and p_a > 0 and p_b > 0:
        feats["price_log_ratio"] = abs(math.log(p_a / p_b))
        feats["price_diff"] = abs(p_a - p_b)
        feats["price_missing"] = 0.0
    else:
        feats["price_log_ratio"] = 0.0
        feats["price_diff"] = 0.0
        feats["price_missing"] = 1.0

    if ds_code == "wa":
        # Walmart-Amazon 特有
        m_a = safe_str(record_a.get("modelno"))
        m_b = safe_str(record_b.get("modelno"))
        if m_a and m_b:
            clean_ma = re.sub(r"[\s\-_/]", "", m_a.lower())
            clean_mb = re.sub(r"[\s\-_/]", "", m_b.lower())
            feats["modelno_match"] = 1.0 if clean_ma == clean_mb else -1.0
        else:
            feats["modelno_match"] = 0.0

        b_a = safe_str(record_a.get("brand")).lower()
        b_b = safe_str(record_b.get("brand")).lower()
        if b_a and b_b:
            feats["brand_match"] = 1.0 if b_a == b_b else -1.0
        else:
            feats["brand_match"] = 0.0

    elif ds_code == "ag":
        # Amazon-Google 特有
        mf_a = safe_str(record_a.get("manufacturer")).lower()
        mf_b = safe_str(record_b.get("manufacturer")).lower()
        if mf_a and mf_b:
            feats["mfr_match"] = 1.0 if mf_a == mf_b else -1.0
        else:
            feats["mfr_match"] = 0.0

        # 正则抽取版本号/年份
        v_a = set(re.findall(r"\b(?:v(?:er)?\.?\s*)?(\d+(?:\.\d+)*)\b", title_a.lower()))
        v_b = set(re.findall(r"\b(?:v(?:er)?\.?\s*)?(\d+(?:\.\d+)*)\b", title_b.lower()))
        if v_a and v_b:
            feats["version_overlap"] = 1.0 if (v_a & v_b) else -1.0
        else:
            feats["version_overlap"] = 0.0

    elif ds_code == "da":
        # DBLP-ACM 特有
        y_a = safe_float(record_a.get("year"))
        y_b = safe_float(record_b.get("year"))
        if y_a is not None and y_b is not None:
            feats["year_diff"] = abs(y_a - y_b)
            feats["year_equal"] = 1.0 if y_a == y_b else 0.0
        else:
            feats["year_diff"] = 0.0
            feats["year_equal"] = 0.0

        auth_a = safe_str(record_a.get("authors"))
        auth_b = safe_str(record_b.get("authors"))
        feats["authors_jaccard"] = word_jaccard(auth_a, auth_b)

        ven_a = safe_str(record_a.get("venue")).lower()
        ven_b = safe_str(record_b.get("venue")).lower()
        feats["venue_fuzz"] = fuzz.ratio(ven_a, ven_b) / 100.0

    elif ds_code == "ab":
        # Abt-Buy 特有
        desc_a = safe_str(record_a.get("description"))
        desc_b = safe_str(record_b.get("description"))
        feats["desc_jaccard"] = word_jaccard(desc_a, desc_b)
        feats["desc_fuzz"] = fuzz.partial_ratio(desc_a.lower()[:300], desc_b.lower()[:300]) / 100.0

        # 型号词元匹配 (长度>=4 且含字母和数字)
        codes_a = set(re.findall(r"\b(?=[a-zA-Z]*\d)(?=\d*[a-zA-Z])[a-zA-Z0-9]{4,}\b", (title_a + " " + desc_a).lower()))
        codes_b = set(re.findall(r"\b(?=[a-zA-Z]*\d)(?=\d*[a-zA-Z])[a-zA-Z0-9]{4,}\b", (title_b + " " + desc_b).lower()))
        if codes_a and codes_b:
            feats["model_code_match"] = 1.0 if (codes_a & codes_b) else -1.0
        else:
            feats["model_code_match"] = 0.0

    return feats
