# === sod_metrics.py ===
import numpy as np

def normalize(pred):
    pred = pred.astype(np.float32)
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    return pred

def mae(pred, gt):
    pred = normalize(pred)
    gt = gt.astype(np.float32) / 255.
    return np.mean(np.abs(pred - gt))

def f_measure(pred, gt, beta_sq=0.3):
    pred = normalize(pred)
    gt = gt.astype(np.float32) / 255.
    thresholds = np.linspace(0, 1, 256)
    precisions, recalls = [], []
    for thresh in thresholds:
        bin_pred = (pred >= thresh).astype(np.float32)
        tp = (bin_pred * gt).sum()
        precision = tp / (bin_pred.sum() + 1e-8)
        recall = tp / (gt.sum() + 1e-8)
        precisions.append(precision)
        recalls.append(recall)
    precisions, recalls = np.array(precisions), np.array(recalls)
    f_scores = (1 + beta_sq) * precisions * recalls / (beta_sq * precisions + recalls + 1e-8)
    return {
        "adp": (1 + beta_sq) * precisions[-1] * recalls[-1] / (beta_sq * precisions[-1] + recalls[-1] + 1e-8),
        "curve": f_scores,
        "max": f_scores.max(),
        "mean": f_scores.mean()
    }

def weighted_f_measure(pred, gt):
    pred = normalize(pred)
    gt = gt.astype(np.float32) / 255.
    return (1 - np.abs(pred - gt)).mean()

def s_measure(pred, gt):
    pred = normalize(pred)
    gt = gt.astype(np.float32) / 255.
    x = pred.mean()
    y = gt.mean()
    alpha = 0.5
    score = 1 - np.abs(x - y)  # simplified
    return score

def e_measure(pred, gt):
    pred = normalize(pred)
    gt = gt.astype(np.float32) / 255.
    Q = 1 - np.abs(pred - gt)
    adp = Q.mean()
    curve = Q.flatten()
    return {
        "adp": adp,
        "curve": curve,
        "max": curve.max(),
        "mean": curve.mean()
    }
