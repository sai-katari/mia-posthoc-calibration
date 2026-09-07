import numpy as np
from sklearn.metrics import roc_auc_score
from src.attacks.adaptive_attack import invert_temperature_exact
from src.attacks.base_attacks import run_attack

CANDIDATE_TEMPS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0]


def estimate_temperature(
    shadow_scaled_probs,
    shadow_true_classes,
    candidate_temps=CANDIDATE_TEMPS,
):
    """Estimate T using shadow data with class labels only.

    For each candidate T, invert shadow probs and compute NLL.
    The correct T minimises NLL — it recovers the best-calibrated
    (most accurate) original distribution on the shadow data.

    shadow_true_classes: integer class labels (0..C-1), NOT membership labels.
    No membership labels are needed for T estimation.
    """
    best_T, best_nll = 1.0, float("inf")

    for T in candidate_temps:
        inverted = invert_temperature_exact(shadow_scaled_probs, T)
        p_true   = inverted[np.arange(len(shadow_true_classes)),
                            shadow_true_classes]
        nll_val  = -np.log(np.clip(p_true, 1e-300, None)).mean()
        if nll_val < best_nll:
            best_nll = nll_val
            best_T   = T

    return best_T, float(best_nll)


def run_semiadaptive_attacks(
    target_scaled_probs,
    target_true_classes,
    target_membership,
    shadow_scaled_probs,
    shadow_true_classes,
    candidate_temps=CANDIDATE_TEMPS,
):
    """Semi-adaptive attack: attacker knows calibration was applied, not T.

    Parameters (kept intentionally separate to avoid the naming bug
    where a single 'shadow_labels' variable is used for two distinct
    concepts):
        target_scaled_probs  : (N_target, C) scaled probability vectors
        target_true_classes  : (N_target,)   integer class labels
        target_membership    : (N_target,)   binary membership labels (0/1)
        shadow_scaled_probs  : (N_shadow, C) attacker's shadow queries
        shadow_true_classes  : (N_shadow,)   class labels for shadow samples
        candidate_temps      : grid of T values to search
    """
    estimated_T, shadow_nll = estimate_temperature(
        shadow_scaled_probs,
        shadow_true_classes,
        candidate_temps,
    )

    inverted = invert_temperature_exact(target_scaled_probs, estimated_T)

    loss_scores = [
        -float(np.log(np.clip(inverted[i, target_true_classes[i]],
                               1e-300, None)))
        for i in range(len(target_true_classes))
    ]
    conf_scores = [float(inverted[i].max())
                   for i in range(len(target_true_classes))]
    entr_scores = [
        -float((inverted[i] *
                np.log(np.clip(inverted[i], 1e-300, None))).sum())
        for i in range(len(target_true_classes))
    ]

    results = [
        run_attack(target_membership, loss_scores,  "loss_semiadaptive"),
        run_attack(target_membership, conf_scores,  "confidence_semiadaptive"),
        run_attack(target_membership, entr_scores,  "entropy_semiadaptive"),
    ]

    return results, estimated_T, shadow_nll
