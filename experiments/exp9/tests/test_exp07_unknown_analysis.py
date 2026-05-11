from __future__ import annotations

import numpy as np

from src.signal.evidence import (
    EVIDENCE_NAMES,
    analysis_feature_vector,
    compute_signal_evidence,
    decide_unknown_protocol,
    forced_label_from_decision,
)


def test_evidence_feature_shape_and_finite() -> None:
    rng = np.random.default_rng(7)
    iq = (rng.normal(size=2048) + 1j * rng.normal(size=2048)).astype(np.complex64)
    channels = np.stack([iq.real, iq.imag]).astype(np.float32)
    evidence = compute_signal_evidence(channels)
    assert list(evidence.keys()) == EVIDENCE_NAMES
    assert all(np.isfinite(value) for value in evidence.values())
    assert all(0.0 <= value <= 1.0 for value in evidence.values())
    vec = analysis_feature_vector(np.array([0.6, 0.3, 0.1], dtype=np.float32), evidence)
    assert vec.shape == (11,)
    assert np.all(np.isfinite(vec))


def test_bask_top1_high_bfsk_evidence_becomes_ambiguous() -> None:
    evidence = {name: 0.0 for name in EVIDENCE_NAMES}
    evidence["instantaneous_frequency_bimodality_score"] = 0.8
    decision = decide_unknown_protocol(np.array([0.55, 0.35, 0.10]), evidence)
    assert decision.final_decision == "AMBIGUOUS_BASK_LIKE_WITH_BFSK_EVIDENCE"
    assert "BFSK" in decision.candidate_modulations


def test_bask_top1_high_bpsk_evidence_becomes_ambiguous() -> None:
    evidence = {name: 0.0 for name in EVIDENCE_NAMES}
    evidence["differential_phase_transition_score"] = 0.7
    decision = decide_unknown_protocol(np.array([0.57, 0.10, 0.33]), evidence)
    assert decision.final_decision == "AMBIGUOUS_BASK_LIKE_WITH_BPSK_EVIDENCE"
    assert "BPSK" in decision.candidate_modulations


def test_low_confidence_becomes_unknown() -> None:
    evidence = {name: 0.0 for name in EVIDENCE_NAMES}
    decision = decide_unknown_protocol(np.array([0.40, 0.32, 0.28]), evidence)
    assert decision.final_decision == "UNKNOWN_LOW_CONFIDENCE"


def test_low_confidence_non_bask_candidate_is_not_forced_unknown() -> None:
    evidence = {name: 0.0 for name in EVIDENCE_NAMES}
    decision = decide_unknown_protocol(np.array([0.32, 0.40, 0.28]), evidence)
    assert decision.final_decision == "BFSK"
    assert "BASK" in decision.candidate_modulations


def test_forced_label_from_ambiguous_decision() -> None:
    assert forced_label_from_decision("AMBIGUOUS_BASK_LIKE_WITH_BFSK_EVIDENCE") == "BFSK"
    assert forced_label_from_decision("AMBIGUOUS_BASK_LIKE_WITH_BPSK_EVIDENCE") == "BPSK"
    assert forced_label_from_decision("BASK") == "BASK"
