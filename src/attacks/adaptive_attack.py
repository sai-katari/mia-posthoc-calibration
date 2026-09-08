import numpy as np
from scipy.special import logsumexp
from src.attacks.base_attacks import run_attack


def invert_temperature_exact(scaled_probs, T):
    """Exact inversion of temperature scaling in log-space.

    Derivation:
        q_i = softmax(z_i / T)
        log q_i = z_i/T - log Z_T
        T * log q_i = z_i - T * log Z_T
        softmax(T * log q_i) = softmax(z_i) = p_i   (exact)

    Valid when:
        - Complete float32 probability vector (reconstruction is approximate)
        - Known positive T
        - No top-k truncation or label-only output

    Uses log-space arithmetic for numerical stability.
    Uses float64 internally. Reconstruction from stored float32 probabilities
    is approximate; the float64 synthetic test serves as the exact sanity check.
    """
    log_q = np.log(np.clip(scaled_probs, 1e-300, None))
    log_p = T * log_q
    log_p -= logsumexp(log_p, axis=1, keepdims=True)
    return np.exp(log_p)


def run_adaptive_attacks(records, T):
    """Known-T adaptive attack.

    Attacker receives the complete float32 scaled probability vector and knows T exactly.
    Probabilities are converted to float64 and renormalized before inversion.
    Inverts the scaling before computing attack scores.

    This is a sanity check: if the implementation is correct and T > 1,
    AUC should recover to approximately the pre-scaling baseline.
    If it does not, suspect the implementation before interpreting
    the gap as a privacy improvement.
    """
    scaled_probs = np.array([r["prob_vector"] for r in records], dtype=np.float32).astype(np.float64)
    scaled_probs /= scaled_probs.sum(axis=1, keepdims=True)
    true_classes = np.array([r["true_class"] for r in records])
    y_true       = [r["membership"] for r in records]

    inverted = invert_temperature_exact(scaled_probs, T)

    # Score convention: higher score = more likely MEMBER.
    # Members have LOW loss and LOW entropy, so both are negated.
    tiny = np.finfo(np.float64).tiny
    log_inv = np.log(np.clip(inverted, tiny, 1.0))
    nll_arr     = -log_inv[np.arange(len(true_classes)), true_classes]
    entropy_arr = -(inverted * log_inv).sum(axis=1)

    loss_scores = list(-nll_arr)
    conf_scores = list(inverted.max(axis=1))
    entr_scores = list(-entropy_arr)

    return [
        run_attack(y_true, loss_scores,  "loss_adaptive"),
        run_attack(y_true, conf_scores,  "confidence_adaptive"),
        run_attack(y_true, entr_scores,  "entropy_adaptive"),
    ]
