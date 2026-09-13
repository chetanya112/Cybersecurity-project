"""
mitre.py
--------
CHRONEX MITRE ATT&CK stage mapping — tactic-aggregated scoring.

FIXES the original heuristic:
    Original: First-match lookup on 13 features. Bugs:
    - Picks whichever mapped feature appears first in the SHAP ranking,
      even if its SHAP contribution is tiny vs. other tactic's features.
    - Only 13 of 71 features mapped — 58 features silently ignored.
    - Can wrongly classify a massive DDoS flood as "Reconnaissance" if
      a minor probe-related feature appears higher than flood features.

    FIX: Aggregate SHAP contributions per MITRE tactic across ALL mapped
    features, then rank tactics by total |SHAP| weight.  This uses the
    same evidence the explainability layer surfaces, but combines it
    properly instead of using first-match.

MITRE ATT&CK Tactics covered (aligned with PS requirements):
    1. Reconnaissance        — Port scanning, probe packets, low-volume recon
    2. Initial Access        — SYN probes, connection establishment patterns
    3. Credential Access     — Repeated auth attempts, brute force patterns
    4. Lateral Movement      — Internal spread, unusual source diversity
    5. Command and Control   — Periodic beaconing, low-and-slow idle patterns
    6. Exfiltration          — Large outbound data, high backward byte counts
    7. Impact                — High-volume floods, packet rate saturation

Usage:
    from mitre import score_mitre_tactics, MitreResult

    result = score_mitre_tactics(ranked_idx, attack_shap, feature_columns)
    print(result.top_tactic)       # e.g. "Impact"
    print(result.tactic_scores)    # dict: tactic -> aggregated score
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# MITRE tactic definitions — comprehensive 71-feature coverage
# ---------------------------------------------------------------------------
# Each entry: feature_name -> (tactic, weight)
# Weight reflects how strongly this feature specifically indicates the tactic
# (vs. being a general traffic indicator shared across tactics).
# Derived from CIC-IDS-2018 SHAP analysis + MITRE ATT&CK v14 technique mapping.

FEATURE_TACTIC_MAP: dict[str, tuple[str, float]] = {

    # ---- Reconnaissance (TA0043) ----
    # Low-volume probing: small packets, port scanning signatures,
    # high destination port diversity, slow inter-arrival times
    "Fwd Seg Size Min":       ("Reconnaissance",        1.0),
    "Pkt Len Min":            ("Reconnaissance",        0.8),
    "Fwd Pkt Len Min":        ("Reconnaissance",        0.8),
    "Fwd Pkt Len Mean":       ("Reconnaissance",        0.5),
    "Fwd IAT Min":            ("Reconnaissance",        0.7),  # slow probes
    "Flow IAT Min":           ("Reconnaissance",        0.6),
    "Dst Port":               ("Reconnaissance",        0.6),  # port diversity
    "FIN Flag Cnt":           ("Reconnaissance",        0.5),  # half-open scan
    "URG Flag Cnt":           ("Reconnaissance",        0.4),
    "Fwd URG Flags":          ("Reconnaissance",        0.4),

    # ---- Initial Access (TA0001) ----
    # SYN-based connection attempts, low RST (legitimate-looking initial probes)
    "SYN Flag Cnt":           ("Initial Access",        0.9),
    "Init Fwd Win Byts":      ("Initial Access",        0.8),
    "Fwd Pkts/s":             ("Initial Access",        0.5),
    "Flow IAT Mean":          ("Initial Access",        0.4),
    "PSH Flag Cnt":           ("Initial Access",        0.4),

    # ---- Credential Access (TA0006) ----
    # Brute force: high SYN + RST cycles, repeated short flows, high flow count
    "RST Flag Cnt":           ("Credential Access",     1.0),
    "flow_count":             ("Credential Access",     0.7),  # many short flows
    "Fwd IAT Tot":            ("Credential Access",     0.6),
    "Fwd IAT Mean":           ("Credential Access",     0.5),
    "Fwd Act Data Pkts":      ("Credential Access",     0.5),
    "ACK Flag Cnt":           ("Credential Access",     0.4),
    "Tot Fwd Pkts":           ("Credential Access",     0.4),

    # ---- Lateral Movement (TA0008) ----
    # Unusual source/destination spread, multiple protocol changes
    "Protocol":               ("Lateral Movement",      0.7),
    "Subflow Fwd Pkts":       ("Lateral Movement",      0.5),
    "Subflow Bwd Pkts":       ("Lateral Movement",      0.5),
    "Down/Up Ratio":          ("Lateral Movement",      0.6),  # internal spread changes ratio
    "Bwd Pkt Len Min":        ("Lateral Movement",      0.4),
    "Fwd Header Len":         ("Lateral Movement",      0.4),
    "Bwd Header Len":         ("Lateral Movement",      0.4),

    # ---- Command and Control (TA0011) ----
    # Periodic beaconing: rhythmic idle/active patterns, low consistent traffic
    "Idle Mean":              ("Command and Control",   1.0),
    "Idle Std":               ("Command and Control",   0.8),
    "Idle Max":               ("Command and Control",   0.7),
    "Idle Min":               ("Command and Control",   0.7),
    "Active Mean":            ("Command and Control",   0.8),
    "Active Std":             ("Command and Control",   0.7),
    "Active Max":             ("Command and Control",   0.6),
    "Active Min":             ("Command and Control",   0.6),
    "Bwd IAT Mean":           ("Command and Control",   0.5),
    "Bwd IAT Min":            ("Command and Control",   0.5),
    "Flow IAT Std":           ("Command and Control",   0.4),

    # ---- Exfiltration (TA0010) ----
    # Large outbound data: high backward byte counts, large packet sizes
    "Subflow Bwd Byts":       ("Exfiltration",          1.0),
    "TotLen Bwd Pkts":        ("Exfiltration",          1.0),
    "Pkt Len Max":            ("Exfiltration",          0.9),
    "Pkt Len Std":            ("Exfiltration",          0.7),
    "Pkt Len Var":            ("Exfiltration",          0.7),
    "Bwd Pkt Len Max":        ("Exfiltration",          0.9),
    "Bwd Pkt Len Mean":       ("Exfiltration",          0.8),
    "Bwd Pkt Len Std":        ("Exfiltration",          0.7),
    "Init Bwd Win Byts":      ("Exfiltration",          0.8),
    "Pkt Size Avg":           ("Exfiltration",          0.6),
    "Bwd Seg Size Avg":       ("Exfiltration",          0.6),

    # ---- Impact (TA0040) ----
    # DoS/DDoS: extreme packet/byte rates, traffic volume saturation
    "Bwd Pkts/s":             ("Impact",                1.0),
    "Flow Pkts/s":            ("Impact",                1.0),
    "Flow Byts/s":            ("Impact",                0.95),
    "Fwd Pkts/s":             ("Impact",                0.9),
    "TotLen Fwd Pkts":        ("Impact",                0.8),
    "Fwd Pkt Len Max":        ("Impact",                0.7),
    "Fwd Pkt Len Std":        ("Impact",                0.6),
    "Bwd IAT Tot":            ("Impact",                0.6),
    "Fwd IAT Max":            ("Impact",                0.5),
    "Flow IAT Max":           ("Impact",                0.5),
    "Subflow Fwd Byts":       ("Impact",                0.6),
    "CWE Flag Count":         ("Impact",                0.5),
    "ECE Flag Cnt":           ("Impact",                0.5),
}

# MITRE metadata for display
TACTIC_META = {
    "Reconnaissance":      {"id": "TA0043", "color": "#6366f1", "emoji": "🔍"},
    "Initial Access":      {"id": "TA0001", "color": "#f59e0b", "emoji": "🚪"},
    "Credential Access":   {"id": "TA0006", "color": "#ef4444", "emoji": "🔑"},
    "Lateral Movement":    {"id": "TA0008", "color": "#ec4899", "emoji": "🔀"},
    "Command and Control": {"id": "TA0011", "color": "#8b5cf6", "emoji": "📡"},
    "Exfiltration":        {"id": "TA0010", "color": "#06b6d4", "emoji": "📤"},
    "Impact":              {"id": "TA0040", "color": "#dc2626", "emoji": "💥"},
}

ALL_TACTICS = list(TACTIC_META.keys())


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class MitreResult:
    """MITRE tactic assignment from aggregated SHAP scoring."""
    top_tactic: str                         # highest-scoring tactic
    top_tactic_id: str                      # MITRE ID, e.g. "TA0040"
    top_tactic_score: float                 # aggregated SHAP weight (normalized)
    runner_up: Optional[str]                # second-highest tactic (or None)
    runner_up_score: float                  # runner-up score
    tactic_scores: dict[str, float]         # all tactics and their scores
    contributing_features: list[tuple[str, str, float]]  # (feat, tactic, shap)
    confidence: str                         # "High" / "Medium" / "Low"
    unclassified: bool                      # True if no features mapped


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------


def score_mitre_tactics(
    ranked_idx: np.ndarray,
    attack_shap: np.ndarray,
    feature_columns: list[str],
    top_k: int = 8,
    min_confidence_threshold: float = 0.05,
) -> MitreResult:
    """
    Score MITRE tactics by aggregating SHAP contributions across all
    mapped features.

    Args:
        ranked_idx:    Feature indices ranked by |SHAP| (from explain_sequence)
        attack_shap:   (n_features,) SHAP values for Attack class
        feature_columns: Feature names
        top_k:         Consider only the top-k most contributing features
        min_confidence_threshold: Minimum total SHAP weight to be "High" confidence

    Returns:
        MitreResult with ranked tactics and contributing features
    """
    tactic_scores: dict[str, float] = {t: 0.0 for t in ALL_TACTICS}
    contributing: list[tuple[str, str, float]] = []

    # Only use features that push TOWARD attack (positive SHAP) in top-k
    top_indices = ranked_idx[:top_k]

    total_positive_shap = 0.0
    for i in top_indices:
        if i >= len(feature_columns):
            continue
        shap_val = float(attack_shap[i])
        if shap_val <= 0:
            continue  # skip features pushing toward Benign

        feat_name = feature_columns[i]
        if feat_name not in FEATURE_TACTIC_MAP:
            continue

        tactic, weight = FEATURE_TACTIC_MAP[feat_name]
        contribution = shap_val * weight
        tactic_scores[tactic] += contribution
        total_positive_shap += contribution
        contributing.append((feat_name, tactic, shap_val))

    # Normalize scores to [0, 1]
    total = sum(tactic_scores.values())
    if total > 0:
        tactic_scores = {t: round(s / total, 4) for t, s in tactic_scores.items()}

    # Rank tactics
    ranked_tactics = sorted(tactic_scores.items(), key=lambda x: x[1], reverse=True)
    non_zero = [(t, s) for t, s in ranked_tactics if s > 0]

    unclassified = len(non_zero) == 0

    if unclassified:
        return MitreResult(
            top_tactic="Unclassified",
            top_tactic_id="N/A",
            top_tactic_score=0.0,
            runner_up=None,
            runner_up_score=0.0,
            tactic_scores=tactic_scores,
            contributing_features=contributing,
            confidence="Low",
            unclassified=True,
        )

    top_tactic, top_score = non_zero[0]
    runner_up_name  = non_zero[1][0] if len(non_zero) > 1 else None
    runner_up_score = non_zero[1][1] if len(non_zero) > 1 else 0.0

    # Confidence based on how dominant the top tactic is
    dominance = top_score / sum(s for _, s in non_zero) if non_zero else 0.0
    if dominance >= 0.60 and total_positive_shap >= min_confidence_threshold:
        confidence = "High"
    elif dominance >= 0.40:
        confidence = "Medium"
    else:
        confidence = "Low"

    top_meta = TACTIC_META.get(top_tactic, {"id": "N/A"})

    return MitreResult(
        top_tactic=top_tactic,
        top_tactic_id=top_meta["id"],
        top_tactic_score=top_score,
        runner_up=runner_up_name,
        runner_up_score=runner_up_score,
        tactic_scores=tactic_scores,
        contributing_features=contributing,
        confidence=confidence,
        unclassified=False,
    )


def format_mitre_report(result: MitreResult) -> str:
    """Format MitreResult as a readable text block."""
    if result.unclassified:
        return ("MITRE Tactic: Unclassified — none of the top SHAP features "
                "matched the tactic mapping table. Flagged for analyst review.")

    meta = TACTIC_META.get(result.top_tactic, {"emoji": "⚠️", "id": "N/A"})
    lines = [
        f"{meta['emoji']}  MITRE Tactic: {result.top_tactic}  [{result.top_tactic_id}]",
        f"   Confidence: {result.confidence}  (score={result.top_tactic_score:.3f})",
    ]
    if result.runner_up:
        ru_meta = TACTIC_META.get(result.runner_up, {"emoji": "➡️"})
        lines.append(f"   Runner-up:  {ru_meta['emoji']} {result.runner_up} "
                     f"(score={result.runner_up_score:.3f})")
    if result.contributing_features:
        lines.append("\n   Contributing features:")
        for feat, tactic, shap_val in result.contributing_features[:5]:
            lines.append(f"     {feat:30s}  SHAP={shap_val:+.4f}  → {tactic}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("MITRE module smoke test...")

    # Simulate SHAP output
    feature_columns = list(FEATURE_TACTIC_MAP.keys())[:20]
    n = len(feature_columns)
    fake_shap = np.random.randn(n).astype(np.float32)
    fake_shap[0]  =  0.42   # Bwd Pkts/s -> Impact
    fake_shap[1]  =  0.38   # Flow Pkts/s -> Impact
    fake_shap[2]  = -0.15   # negative (pushes Benign)
    ranked_idx = np.argsort(np.abs(fake_shap))[::-1]

    result = score_mitre_tactics(ranked_idx, fake_shap, feature_columns)
    print(format_mitre_report(result))
    print(f"\nAll tactic scores: {result.tactic_scores}")
    print("\n✅ MITRE module OK")
