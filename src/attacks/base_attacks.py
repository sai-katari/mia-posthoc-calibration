import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def tpr_at_fpr(y_true, scores, target_fpr):
    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = fpr <= target_fpr
    return float(tpr[valid].max()) if valid.any() else 0.0



def run_attack(y_true, scores, name):
    auc  = roc_auc_score(y_true, scores)
    tpr1 = tpr_at_fpr(y_true, scores, 0.001)
    tpr2 = tpr_at_fpr(y_true, scores, 0.01)
    return {
        "attack":          name,
        "auc":             round(auc,  4),
        "tpr_at_0001_fpr": round(tpr1, 4),
        "tpr_at_001_fpr":  round(tpr2, 4),
    }


def run_all_attacks(records, suffix=""):
    """Loss, confidence, entropy attacks on a list of sample records."""
    y_true      = [r["membership"] for r in records]
    loss_scores = [-r["loss"]      for r in records]
    conf_scores = [r["max_conf"]   for r in records]
    entr_scores = [-r["entropy"]   for r in records]
    return [
        run_attack(y_true, loss_scores,  f"loss{suffix}"),
        run_attack(y_true, conf_scores,  f"confidence{suffix}"),
        run_attack(y_true, entr_scores,  f"entropy{suffix}"),
    ]
