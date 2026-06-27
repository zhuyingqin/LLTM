"""Faithful copy of AnomLLM's evaluation (src/utils.py: compute_metrics + interval/vector
helpers), depending only on sklearn + the affiliation package (NOT their openai_api chain).
This guarantees our numbers are computed by the SAME metric the paper reports (affinity F1)."""
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from affiliation.generics import convert_vector_to_events
from affiliation.metrics import pr_from_events


def interval_to_vector(intervals, length):
    """anom intervals (list of (start,end)) -> binary vector of given length."""
    v = np.zeros(length, dtype=int)
    for start, end in intervals:
        v[int(start):int(end)] = 1
    return v


def vector_to_interval(vector):
    intervals, in_iv, start = [], False, 0
    for i, val in enumerate(vector):
        if val == 1 and not in_iv:
            start, in_iv = i, True
        elif val == 0 and in_iv:
            intervals.append((start, i)); in_iv = False
    if in_iv:
        intervals.append((start, len(vector)))
    return intervals


def compute_metrics(gt, prediction):
    """EXACT port of AnomLLM utils.compute_metrics. gt/prediction = binary vectors."""
    if prediction is None:
        return dict(zip(["precision", "recall", "f1", "affi precision", "affi recall", "affi f1"], [0]*6))
    if np.count_nonzero(gt) == 0 and np.count_nonzero(prediction) == 0:
        return dict(zip(["precision", "recall", "f1", "affi precision", "affi recall", "affi f1"], [1]*6))
    if np.count_nonzero(gt) == 0 or np.count_nonzero(prediction) == 0:
        return dict(zip(["precision", "recall", "f1", "affi precision", "affi recall", "affi f1"], [0]*6))
    precision = precision_score(gt, prediction)
    recall = recall_score(gt, prediction)
    f1 = f1_score(gt, prediction)
    events_pred = convert_vector_to_events(prediction)
    events_gt = convert_vector_to_events(gt)
    aff = pr_from_events(events_pred, events_gt, (0, len(prediction)))
    if aff["precision"] + aff["recall"] == 0:
        affi_f1 = 0.0
    else:
        affi_f1 = 2 * aff["precision"] * aff["recall"] / (aff["precision"] + aff["recall"])
    return {"precision": precision, "recall": recall, "f1": f1,
            "affi precision": aff["precision"], "affi recall": aff["recall"], "affi f1": affi_f1}


def mean_metrics(gts, preds):
    """Average compute_metrics over a list of (gt, pred) binary-vector pairs."""
    keys = ["precision", "recall", "f1", "affi precision", "affi recall", "affi f1"]
    acc = {k: 0.0 for k in keys}
    for gt, pr in zip(gts, preds):
        m = compute_metrics(gt, pr)
        for k in keys:
            acc[k] += m[k]
    n = max(len(gts), 1)
    return {k: acc[k] / n for k in keys}
