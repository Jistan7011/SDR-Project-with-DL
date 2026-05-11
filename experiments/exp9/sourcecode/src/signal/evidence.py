from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.common import CLASS_NAMES
from src.signal.processing import complex64_from_channels, normalize_iq


EVIDENCE_NAMES = [
    "envelope_dominance_score",
    "amplitude_bimodality_score",
    "instantaneous_frequency_bimodality_score",
    "spectral_peak_separation_score",
    "differential_phase_transition_score",
]

FINAL_DECISIONS = [
    "BASK",
    "BFSK",
    "BPSK",
    "AMBIGUOUS_BASK_LIKE_WITH_BFSK_EVIDENCE",
    "AMBIGUOUS_BASK_LIKE_WITH_BPSK_EVIDENCE",
    "UNKNOWN_LOW_CONFIDENCE",
]


@dataclass(frozen=True)
class EvidenceDecision:
    final_decision: str
    candidate_modulations: list[str]
    bask_absorption_risk: str
    unknown_score: float
    selection_reason: str


def compute_signal_evidence(channels: np.ndarray) -> dict[str, float]:
    iq = normalize_iq(complex64_from_channels(channels[:2]))
    mag = np.abs(iq).astype(np.float32)
    phase_step = _phase_step(iq)
    spectrum_power = np.log1p(np.abs(np.fft.fftshift(np.fft.fft(iq))) ** 2).astype(np.float32)

    evidence = {
        "envelope_dominance_score": _clip01(float(np.var(mag) / (np.var(iq.real) + np.var(iq.imag) + 1e-8))),
        "amplitude_bimodality_score": _bimodality_score(mag),
        "instantaneous_frequency_bimodality_score": _bimodality_score(phase_step),
        "spectral_peak_separation_score": _spectral_peak_score(spectrum_power),
        "differential_phase_transition_score": _phase_transition_score(iq),
    }
    return evidence


def evidence_vector(evidence: dict[str, float]) -> np.ndarray:
    return np.asarray([float(evidence[name]) for name in EVIDENCE_NAMES], dtype=np.float32)


def classifier_uncertainty(probs: np.ndarray) -> dict[str, float | str]:
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    order = np.argsort(-p)
    confidence = float(p[order[0]])
    second = float(p[order[1]]) if len(order) > 1 else 0.0
    entropy = float(-np.sum(p * np.log(p + 1e-12)))
    return {
        "top1": CLASS_NAMES[int(order[0])],
        "top2": CLASS_NAMES[int(order[1])] if len(order) > 1 else "",
        "top3": CLASS_NAMES[int(order[2])] if len(order) > 2 else "",
        "base_confidence": confidence,
        "confidence_margin": confidence - second,
        "softmax_entropy": entropy,
    }


def analysis_feature_vector(probs: np.ndarray, evidence: dict[str, float]) -> np.ndarray:
    uncertainty = classifier_uncertainty(probs)
    return np.concatenate(
        [
            np.asarray(probs, dtype=np.float32).reshape(-1),
            evidence_vector(evidence),
            np.asarray(
                [
                    float(uncertainty["base_confidence"]),
                    float(uncertainty["confidence_margin"]),
                    float(uncertainty["softmax_entropy"]),
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)


def decide_unknown_protocol(
    probs: np.ndarray,
    evidence: dict[str, float],
    thresholds: dict[str, Any] | None = None,
    forced_idx: int | None = None,
) -> EvidenceDecision:
    th = thresholds or {}
    low_conf = float(th.get("low_confidence", 0.48))
    low_margin = float(th.get("low_margin", 0.18))
    high_freq = float(th.get("high_frequency_separation", 0.58))
    high_phase = float(th.get("high_phase_transition", 0.42))
    high_entropy = float(th.get("high_unknown_entropy", 1.02))

    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    order = np.argsort(-p)
    base_top1 = CLASS_NAMES[int(order[0])]
    top1_idx = int(order[0] if forced_idx is None else forced_idx)
    top1 = CLASS_NAMES[top1_idx]
    uncertainty = classifier_uncertainty(p)
    confidence = float(uncertainty["base_confidence"])
    margin = float(uncertainty["confidence_margin"])
    entropy = float(uncertainty["softmax_entropy"])
    freq_score = float(evidence.get("instantaneous_frequency_bimodality_score", 0.0))
    spectral_score = float(evidence.get("spectral_peak_separation_score", 0.0))
    phase_score = float(evidence.get("differential_phase_transition_score", 0.0))
    unknown_score = _clip01((1.0 - confidence) * 0.55 + min(entropy / 1.10, 1.0) * 0.45)

    candidates = _topk_names(order, 3)
    if ((confidence < low_conf and margin < low_margin) or (entropy > high_entropy and confidence < low_conf)) and base_top1 != "BASK":
        return EvidenceDecision(base_top1, candidates, "low", unknown_score, "low_confidence_base_non_bask_candidate")

    if (confidence < low_conf and margin < low_margin) or (entropy > high_entropy and confidence < low_conf):
        return EvidenceDecision("UNKNOWN_LOW_CONFIDENCE", candidates, "medium", unknown_score, "low_confidence_and_low_margin")

    if top1 == "BASK":
        risk_reasons: list[str] = []
        if margin < low_margin:
            risk_reasons.append("low_margin")
        if max(freq_score, spectral_score) >= high_freq:
            risk_reasons.append("frequency_evidence")
            candidates = _append_unique(candidates, "BFSK")
        if phase_score >= high_phase:
            risk_reasons.append("phase_evidence")
            candidates = _append_unique(candidates, "BPSK")
        if "frequency_evidence" in risk_reasons:
            return EvidenceDecision("AMBIGUOUS_BASK_LIKE_WITH_BFSK_EVIDENCE", candidates, "high", unknown_score, "+".join(risk_reasons))
        if "phase_evidence" in risk_reasons:
            return EvidenceDecision("AMBIGUOUS_BASK_LIKE_WITH_BPSK_EVIDENCE", candidates, "high", unknown_score, "+".join(risk_reasons))
        if risk_reasons:
            non_bask_candidates = [name for name in candidates if name != "BASK"]
            if non_bask_candidates:
                if non_bask_candidates[0] == "BFSK":
                    return EvidenceDecision("AMBIGUOUS_BASK_LIKE_WITH_BFSK_EVIDENCE", candidates, "medium", unknown_score, "+".join(risk_reasons))
                return EvidenceDecision("AMBIGUOUS_BASK_LIKE_WITH_BPSK_EVIDENCE", candidates, "medium", unknown_score, "+".join(risk_reasons))
            return EvidenceDecision("UNKNOWN_LOW_CONFIDENCE", candidates, "medium", unknown_score, "+".join(risk_reasons))
        return EvidenceDecision("BASK", ["BASK"], "low", unknown_score, "hard_bask")

    return EvidenceDecision(top1, [top1], "low", unknown_score, f"hard_{top1.lower()}")


def forced_label_from_decision(decision: str) -> str:
    if decision in CLASS_NAMES:
        return decision
    if decision == "AMBIGUOUS_BASK_LIKE_WITH_BFSK_EVIDENCE":
        return "BFSK"
    if decision == "AMBIGUOUS_BASK_LIKE_WITH_BPSK_EVIDENCE":
        return "BPSK"
    return "BASK"


def forced_label_from_decision_with_candidates(decision: str, candidates: list[str]) -> str:
    if decision in CLASS_NAMES:
        return decision
    if decision == "AMBIGUOUS_BASK_LIKE_WITH_BFSK_EVIDENCE":
        return "BFSK"
    if decision == "AMBIGUOUS_BASK_LIKE_WITH_BPSK_EVIDENCE":
        return "BPSK"
    for candidate in candidates:
        if candidate in CLASS_NAMES:
            return candidate
    return "BASK"


def _phase_step(iq: np.ndarray) -> np.ndarray:
    if len(iq) < 2:
        return np.zeros(len(iq), dtype=np.float32)
    step = np.angle(iq[1:] * np.conj(iq[:-1])).astype(np.float32)
    return np.pad(step, (1, 0))


def _bimodality_score(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(x) < 8 or float(np.std(x)) < 1e-8:
        return 0.0
    low = x[x <= np.median(x)]
    high = x[x > np.median(x)]
    if len(low) == 0 or len(high) == 0:
        return 0.0
    separation = abs(float(np.mean(high) - np.mean(low))) / (float(np.std(x)) + 1e-8)
    balance = 1.0 - abs(len(high) - len(low)) / max(len(x), 1)
    return _clip01((separation / 2.5) * balance)


def _spectral_peak_score(power: np.ndarray) -> float:
    x = np.asarray(power, dtype=np.float32).reshape(-1)
    if len(x) < 16:
        return 0.0
    z = (x - np.mean(x)) / (np.std(x) + 1e-8)
    left = z[: len(z) // 2]
    right = z[len(z) // 2 :]
    left_idx = int(np.argmax(left))
    right_idx = int(np.argmax(right)) + len(z) // 2
    prominence = max(0.0, min(float(z[left_idx]), float(z[right_idx])) / 5.0)
    separation = abs(right_idx - left_idx) / max(len(z), 1)
    return _clip01(1.8 * separation * prominence)


def _phase_transition_score(iq: np.ndarray, symbol_samples: int = 32) -> float:
    delay = max(1, int(symbol_samples))
    if len(iq) <= delay:
        return 0.0
    dphase = np.abs(np.angle(iq[delay:] * np.conj(iq[:-delay])))
    near_pi = np.mean(dphase > (0.65 * np.pi))
    concentration = np.mean(np.abs(dphase - np.pi) < (0.25 * np.pi))
    return _clip01(float(0.65 * near_pi + 0.35 * concentration))


def _topk_names(order: np.ndarray, k: int) -> list[str]:
    return [CLASS_NAMES[int(idx)] for idx in order[:k]]


def _append_unique(values: list[str], value: str) -> list[str]:
    return values if value in values else values + [value]


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))
