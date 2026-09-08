"""Attack implementation tests — no data download or GPU needed."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.attacks.adaptive_attack import invert_temperature_exact


def make_probs(n=200, n_classes=7, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    logits = rng.normal(0, 1, (n, n_classes))
    logits -= logits.max(axis=1, keepdims=True)
    exp_l  = np.exp(logits)
    return exp_l / exp_l.sum(axis=1, keepdims=True)


def apply_temperature(probs, T):
    """Apply temperature scaling in log-space (same as invert with 1/T)."""
    log_p = np.log(np.clip(probs, 1e-300, None))
    log_q = log_p / T
    from scipy.special import logsumexp
    log_q -= logsumexp(log_q, axis=1, keepdims=True)
    return np.exp(log_q)


def test_invert_identity_T1():
    """Inversion with T=1 should return the same probabilities."""
    probs    = make_probs()
    inverted = invert_temperature_exact(probs, T=1.0)
    np.testing.assert_allclose(inverted, probs, rtol=1e-5)


def test_invert_exact_recovery():
    """Inverting scaled probs with the correct T recovers the originals."""
    original  = make_probs()
    T         = 2.5
    scaled    = apply_temperature(original, T)
    recovered = invert_temperature_exact(scaled, T)
    np.testing.assert_allclose(recovered, original, rtol=1e-5)


def test_invert_sums_to_one():
    probs    = make_probs()
    inverted = invert_temperature_exact(probs, T=3.0)
    np.testing.assert_allclose(inverted.sum(axis=1),
                                np.ones(len(probs)), rtol=1e-6)


def test_invert_all_positive():
    probs    = make_probs()
    inverted = invert_temperature_exact(probs, T=2.0)
    assert (inverted > 0).all()

