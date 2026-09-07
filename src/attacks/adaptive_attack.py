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
        - Full-precision complete probability vector
        - Known positive T
        - No rounding, top-k truncation, quantization, or label-only output

    Uses log-space arithmetic for numerical stability.
    clip to 1e-300 only protects against log(0) on entries that are
    genuinely zero; non-zero entries are not distorted.
    """
    log_q = np.log(np.clip(scaled_probs, 1e-300, None))
    log_p = T * log_q
    log_p -= logsumexp(log_p, axis=1, keepdims=True)
    return np.exp(log_p)


def run_adaptive_attacks(records, T):
    """Known-T adaptive attack.

    Attacker receives full-precision scaled probabilities and knows T exactly.
    Inverts the scaling before computing attack scores.

    This is a sanity check: if the implementation is correct and T > 1,
    AUC should recover to approximately the pre-scaling baseline.
    If it does not, suspect the implementation before interpreting
    the gap as a privacy improvement.
    """
    scaled_probs = np.array([r["prob_vector"] for r in records])
    true_classes = [r["true_class"]  for r in records]
    y_true       = [r["membership"]  for r in records]

    inverted = invert_temperature_exact(scaled_probs, T)

    # Score convention (must match base_attacks.run_all_attacks):
    # higher score = more likely MEMBER. Members have LOW loss and LOW
    # entropy, so both are negated. Confidence needs no negation.
    nll = [
        float(-np.log(np.clip(inverted[i, true_classes[i]], 1e-300, None)))
        for i in range(len(records))
    ]
    entropy = [
        float(-(inverted[i] *
                np.log(np.clip(inverted[i], 1e-300, None))).sum())
        for i in range(len(records))
    ]
    loss_scores = [-v for v in nll]
    conf_scores = [float(inverted[i].max()) for i in range(len(records))]
    entr_scores = [-v for v in entropy]

    return [
        run_attack(y_true, loss_scores,  "loss_adaptive"),
        run_attack(y_true, conf_scores,  "confidence_adaptive"),
        run_attack(y_true, entr_scores,  "entropy_adaptive"),
    ]
