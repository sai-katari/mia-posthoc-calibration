import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score


def tpr_at_fpr(y_true, scores, target_fpr):
    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = fpr <= target_fpr
    return float(tpr[valid].max()) if valid.any() else 0.0


def attack_accuracy(y_true, scores, rng_seed=0):
    """Balanced attack accuracy on a stratified calibration/evaluation split."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    rng    = np.random.default_rng(rng_seed)

    member_idx    = np.where(y_true == 1)[0]
    nonmember_idx = np.where(y_true == 0)[0]
    n             = min(len(member_idx), len(nonmember_idx))
    member_idx    = rng.choice(member_idx,    size=n, replace=False)
    nonmember_idx = rng.choice(nonmember_idx, size=n, replace=False)

    idx   = rng.permutation(np.concatenate([member_idx, nonmember_idx]))
    y_bal = y_true[idx]
    s_bal = scores[idx]

    member_bal    = np.where(y_bal == 1)[0]
    nonmember_bal = np.where(y_bal == 0)[0]
    rng.shuffle(member_bal)
    rng.shuffle(nonmember_bal)

    m_half = len(member_bal)    // 2
    n_half = len(nonmember_bal) // 2

    cal_idx  = np.concatenate([member_bal[:m_half],  nonmember_bal[:n_half]])
    eval_idx = np.concatenate([member_bal[m_half:],  nonmember_bal[n_half:]])

    cal_y, cal_s = y_bal[cal_idx],  s_bal[cal_idx]
    ev_y,  ev_s  = y_bal[eval_idx], s_bal[eval_idx]

    fpr, tpr, thresholds = roc_curve(cal_y, cal_s)
    best_thresh = thresholds[np.argmax(tpr - fpr)]
    preds = (ev_s >= best_thresh).astype(int)

    return float(balanced_accuracy_score(ev_y, preds))


def run_attack(y_true, scores, name):
    auc  = roc_auc_score(y_true, scores)
    tpr1 = tpr_at_fpr(y_true, scores, 0.001)
    tpr2 = tpr_at_fpr(y_true, scores, 0.01)
    acc  = attack_accuracy(y_true, scores)
    return {
        "attack":            name,
        "auc":               round(auc,  4),
        "tpr_at_0001_fpr":   round(tpr1, 4),
        "tpr_at_001_fpr":    round(tpr2, 4),
        "balanced_accuracy": round(acc,  4),
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
