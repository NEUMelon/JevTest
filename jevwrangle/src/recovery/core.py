"""Strict probability decoding and the preregistered LR aggregation procedure."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, f1_score


def probability(value):
    p = float(value)
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("Probability must be finite and in [0, 1]")
    return p


def distribution(answer, classes):
    values = answer.get("probabilities")
    if not isinstance(values, dict) or any(k not in values for k in classes):
        raise ValueError("Full class probabilities are required")
    p = np.array([probability(values[k]) for k in classes])
    # Permit response serialization rounding, but never invent absent classes.
    if abs(p.sum() - 1) > 0.02:
        raise ValueError("Class probabilities do not sum to one")
    return p


def noul(answer):
    if isinstance(answer, dict):
        if "noul" in answer:
            return probability(answer["noul"])
        if "probability" in answer:
            return probability(answer["probability"])
        raise ValueError("Missing Noul probability")
    return probability(answer)


def score(answer):
    return probability(distribution(answer, ["0", "1", "2"])[2])


def choice(answer):
    return probability(distribution(answer, ["same", "different", "cannot_tell"])[0])


def llm_probability(content):
    """Keep the reported match probability even when the answer contradicts it."""
    if isinstance(content, str):
        content = json.loads(content)
    if not isinstance(content, dict):
        raise ValueError("Expected a JSON decision object")
    for key in ["match_probability", "p_yes", "probability", "prob"]:
        if content.get(key) is not None:
            return probability(content[key])
    raise ValueError("Missing reported probability; do not fabricate 0.99/0.01")


def logit(p):
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid or missing feature probability")
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def best_oof_threshold(y, p):
    """Exact unique-probability grid, computed by cumulative counts in O(n log n)."""
    y,p=np.asarray(y,dtype=int),np.asarray(p,dtype=float)
    order=np.argsort(p,kind='stable')
    values,first=np.unique(p[order],return_index=True)
    positive=int(y.sum())
    prefix=np.r_[0,np.cumsum(y[order])]
    tp=positive-prefix[first]
    denom=len(y)-first+positive
    scores=np.divide(2*tp,denom,out=np.zeros_like(tp,dtype=float),where=denom>0)
    return float(values[np.argmax(scores)])


def metrics(y, p, threshold=0.5):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if len(y) != len(p) or not len(y) or not np.isfinite(p).all():
        raise ValueError("Metrics require complete finite predictions")
    pred = p >= threshold
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return dict(f1=100*float(f1), precision=100*float(prec), recall=100*float(rec),
                auprc=100*float(average_precision_score(y, p)) if y.sum() else np.nan,
                tp=int(np.sum(pred & (y == 1))), fp=int(np.sum(pred & (y == 0))),
                fn=int(np.sum(~pred & (y == 1))), tn=int(np.sum(~pred & (y == 0))))


@threadpool_limits.wrap(limits=1)
def fit_aggregator(X, y, budget, seed=0):
    """Train-only scaling, log-loss C search and 3-repeat OOF F1 threshold.

    C is selected once on labeled training data (the plan's stated sequence).
    The threshold OOF folds reuse that C; this is not nested model selection.
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
    if not np.isfinite(X).all() or len(X) != len(y):
        raise ValueError("Missing features cannot silently become zero")
    counts = np.bincount(y, minlength=2)
    k = min(5, int(counts.min()))
    if k < 2:
        raise ValueError("OOF aggregation requires at least two labels per class")
    def model(C):
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000,
                                                                  random_state=seed))
    C, scores = 1.0, {}
    if budget >= 1000:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        for c in [0.01, 0.1, 1.0, 10.0]:
            scores[c] = float(cross_val_score(model(c), X, y, cv=cv, scoring="neg_log_loss").mean())
        C = max(scores, key=scores.get)
    sums, visits = np.zeros(len(y)), np.zeros(len(y), dtype=int)
    splitter = RepeatedStratifiedKFold(n_splits=k, n_repeats=3, random_state=seed)
    for tr, va in splitter.split(X, y):
        m = model(C).fit(X[tr], y[tr])
        sums[va] += m.predict_proba(X[va])[:, 1]
        visits[va] += 1
    if not np.all(visits == 3):
        raise AssertionError("Every training item must have three held-out predictions")
    oof = sums / visits
    # np.unique ascending + first maximum exactly follows plan Appendix B.3.
    threshold = best_oof_threshold(y, oof)
    fitted = model(C).fit(X, y)
    return fitted, threshold, oof, dict(C=C, folds=k, repeats=3, cv_neg_log_loss=scores)
