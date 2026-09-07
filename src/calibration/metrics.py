import numpy as np


def expected_calibration_error(probs, labels, n_bins=10):
    """ECE with final-bin edge case fix to include confidence == 1.0.

    Without the fix, samples with confidence exactly 1.0 fall outside
    every half-open interval [a, b) and are silently dropped.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct     = (predictions == labels).astype(float)
    bin_edges   = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (confidences >= bin_edges[i]) & \
                   (confidences <= bin_edges[i + 1])
        else:
            mask = (confidences >= bin_edges[i]) & \
                   (confidences <  bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(correct[mask].mean() -
                                  confidences[mask].mean())
    return float(ece)


def nll(probs, labels):
    p_true = probs[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(p_true, 1e-300, None)).mean())


def brier_score(probs, labels):
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1
    return float(((probs - one_hot) ** 2).sum(axis=1).mean())


def avg_entropy(probs):
    return float(
        -(probs * np.log(np.clip(probs, 1e-300, None))).sum(axis=1).mean()
    )


def member_nonmember_entropy(records):
    """Entropy reported separately for members and non-members.

    This is a privacy analysis metric, not a calibration metric.
    Do not mix the two populations — calibration metrics use test set only.
    """
    m_probs  = np.array([r["prob_vector"] for r in records
                         if r["membership"] == 1])
    nm_probs = np.array([r["prob_vector"] for r in records
                         if r["membership"] == 0])
    m_entr   = avg_entropy(m_probs)
    nm_entr  = avg_entropy(nm_probs)
    return {
        "member_entropy":    round(m_entr,  6),
        "nonmember_entropy": round(nm_entr, 6),
        "entropy_gap":       round(nm_entr - m_entr, 6),
    }


def compute_all_metrics(probs, labels):
    """All calibration metrics. probs and labels must be from test set only."""
    return {
        "ece":         round(expected_calibration_error(probs, labels), 6),
        "nll":         round(nll(probs, labels), 6),
        "brier":       round(brier_score(probs, labels), 6),
        "avg_entropy": round(avg_entropy(probs), 6),
    }
