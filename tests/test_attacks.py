"""Attack implementation tests — no data download or GPU needed."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.attacks.adaptive_attack import invert_temperature_exact
from src.attacks.semiadaptive_attack import estimate_temperature


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


def test_estimate_temperature_recovers_T():
    """T estimation should find the correct T when given enough shadow data."""
    original = make_probs(n=500)
    T_true   = 2.0
    scaled   = apply_temperature(original, T_true)
    true_classes = original.argmax(axis=1)

    candidates = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    est_T, _   = estimate_temperature(scaled, true_classes, candidates)
    assert est_T == pytest.approx(T_true, abs=0.5)


def test_semiadaptive_separate_variables():
    """Confirm shadow_true_classes and shadow_membership are kept separate."""
    from src.attacks.semiadaptive_attack import run_semiadaptive_attacks
    rng = np.random.default_rng(1)

    n_target, n_shadow, C = 100, 50, 7
    target_probs = make_probs(n_target)
    shadow_probs = make_probs(n_shadow, rng_seed=1)

    target_true_classes = rng.integers(0, C, n_target).tolist()
    target_membership   = rng.integers(0, 2, n_target).tolist()
    shadow_true_classes = rng.integers(0, C, n_shadow).tolist()
    # shadow_membership is intentionally NOT passed — the function
    # signature confirms separation

    results, est_T, _ = run_semiadaptive_attacks(
        target_probs,
        target_true_classes,
        target_membership,
        shadow_probs,
        shadow_true_classes,
    )
    assert len(results) == 3
    assert all("semiadaptive" in r["attack"] for r in results)
    assert 0.0 < est_T
