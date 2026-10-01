"""Normalize missing markers for automatic slices, never benchmark labels."""
from src.experiments.e8_error_taxonomy import classify_pair_taxonomy as legacy_rule

def normalize_missing(record):
    return {key:'' if value is None or (isinstance(value,str) and value.strip().lower() in {'','none','null','nan','n/a'}) else value
            for key,value in record.items()}

def classify_pair_taxonomy(dataset,record_a,record_b,label):
    # The legacy AG rule turns None into "none" and misses this missing-value
    # marker. Apply the same absence semantics to both sides and all fields.
    return legacy_rule(dataset,normalize_missing(record_a),normalize_missing(record_b),label)
