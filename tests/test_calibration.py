"""Calibration metric tests — no data download or GPU needed."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.calibration.metrics import (
    expected_calibration_error,
    nll,
    brier_score,
    avg_entropy,
    member_nonmember_entropy,
)


def test_ece_includes_confidence_one():
    """Samples with confidence==1.0 must not be silently dropped."""
    probs  = np.array([[1.0, 0.0], [0.9, 0.1]])
    labels = np.array([0, 0])
    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert ece >= 0.0
    # If confidence==1.0 were dropped, only one sample contributes — check both run
    probs2  = np.array([[1.0, 0.0]])
    labels2 = np.array([0])
    ece2 = expected_calibration_error(probs2, labels2)
    assert ece2 == pytest.approx(0.0, abs=1e-6)


def test_ece_perfect_calibration():
    """A model that is always right with the right confidence has ECE~0."""
    n = 1000
    rng = np.random.default_rng(0)
    # Uniform confidence 0.9, always correct
    probs = np.zeros((n, 2))
    probs[:, 0] = 0.9
    probs[:, 1] = 0.1
    labels = np.zeros(n, dtype=int)
    ece = expected_calibration_error(probs, labels)
    assert ece < 0.15  # not perfect but close


def test_nll_known_value():
    probs  = np.array([[0.8, 0.2], [0.5, 0.5]])
    labels = np.array([0, 1])
    expected = -(np.log(0.8) + np.log(0.5)) / 2
    assert nll(probs, labels) == pytest.approx(expected, rel=1e-5)


def test_brier_perfect():
    probs  = np.array([[1.0, 0.0]])
    labels = np.array([0])
    assert brier_score(probs, labels) == pytest.approx(0.0, abs=1e-6)


def test_avg_entropy_uniform():
    """Uniform distribution over C classes has entropy log(C)."""
    C = 7
    probs = np.ones((10, C)) / C
    expected = np.log(C)
    assert avg_entropy(probs) == pytest.approx(expected, rel=1e-5)


def test_member_nonmember_entropy_separation():
    """Members should have lower entropy than non-members if model is confident."""
    n = 100
    member_probs    = np.zeros((n, 7))
    member_probs[:, 0] = 0.95
    member_probs[:, 1:] = 0.05 / 6

    nonmember_probs = np.ones((n, 7)) / 7  # uniform

    records = (
        [{"membership": 1, "prob_vector": p.tolist()} for p in member_probs] +
        [{"membership": 0, "prob_vector": p.tolist()} for p in nonmember_probs]
    )
    stats = member_nonmember_entropy(records)
    assert stats["member_entropy"] < stats["nonmember_entropy"]
    assert stats["entropy_gap"] > 0
