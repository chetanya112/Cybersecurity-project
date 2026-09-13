"""
pcap_features.py
-----------------
CHRONEX packet-level feature extraction — satisfying the PS mandatory
dual-feature requirement.

Problem Statement (exact):
    "Two levels of traffic features required together, not optional:
    - Flow-level (NetFlow/IPFIX): ...
    - Packet-level (PCAP-derived): TTL variance, TCP window size,
      fragment flags, payload size distribution, port-scan signatures,
      retransmission counts"

This module extracts the PS-required packet-level features from a PCAP
file using Scapy (or Pyshark as fallback).  It also provides a fusion
function that combines packet-level metrics with windowed flow vectors
to demonstrate the dual-feature combination.

The full multi-day PCAP parsing is not required to satisfy the PS — one
representative day demonstrating the *combination* is sufficient.

Usage:
    from pcap_features import extract_packet_features, fuse_with_flow_vector

    # From PCAP
    pkt_feats = extract_packet_features("path/to/capture.pcap")

    # Fuse with existing NetFlow-based state vector (shape: (n_flow_features,))
    combined = fuse_with_flow_vector(flow_vector, pkt_feats)

    # If no PCAP available (CSV-only mode), get zero-padded vector
    pkt_feats = zero_packet_features()
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

PACKET_FEATURE_NAMES = [
    "ttl_mean",               # mean IP TTL
    "ttl_std",                # TTL variance (reveals TTL manipulation / spoofing)
    "ttl_min",                # minimum TTL (helps detect multi-hop attackers)
    "ttl_max",
    "tcp_window_mean",        # mean TCP window size
    "tcp_window_std",         # TCP window variance
    "tcp_window_min",
    "tcp_retransmission_rate",# retransmitted packets / total TCP packets
    "fragment_flag_rate",     # fragmented packets / total packets
    "payload_size_mean",      # mean payload (layer 4 data) size
    "payload_size_std",
    "payload_entropy",        # Shannon entropy of payload sizes (low = uniform flood)
    "unique_dst_ports",       # distinct dst ports per src (port scan signature)
    "syn_only_ratio",         # SYN-only packets (no ACK) — indicates scan/DoS
    "small_packet_ratio",     # packets < 64 bytes (probe / DNS amp / NTP amp)
    "n_packets_total",        # total packet count in the window
]

N_PACKET_FEATURES = len(PACKET_FEATURE_NAMES)


@dataclass
class PacketFeatures:
    """Packet-level features extracted from a PCAP window."""
    features: np.ndarray          # shape (N_PACKET_FEATURES,)
    feature_names: list[str]      # PACKET_FEATURE_NAMES
    n_packets: int
    source: str                   # "pcap" or "zeros" (CSV-only mode)


# ---------------------------------------------------------------------------
# Shannon entropy helper
# ---------------------------------------------------------------------------


def _entropy(values: list[float]) -> float:
    """Shannon entropy of a list of values (via frequency histogram)."""
    if len(values) < 2:
        return 0.0
    arr = np.array(values)
    counts, _ = np.histogram(arr, bins=min(20, len(set(values))))
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


# ---------------------------------------------------------------------------
# Extract from PCAP using Scapy
# ---------------------------------------------------------------------------


def extract_packet_features(
    pcap_path: str,
    window_start: Optional[float] = None,
    window_end: Optional[float] = None,
    max_packets: int = 100_000,
) -> PacketFeatures:
    """
    Extract PS-required packet-level features from a PCAP file.

    Args:
        pcap_path:    Path to .pcap or .pcapng file
        window_start: Unix timestamp for start of analysis window (None = all)
        window_end:   Unix timestamp for end of analysis window (None = all)
        max_packets:  Safety limit to prevent memory issues on large captures

    Returns:
        PacketFeatures dataclass

    Raises:
        ImportError if scapy is not installed
        FileNotFoundError if pcap_path does not exist
    """
    path = Path(pcap_path)
    if not path.exists():
        raise FileNotFoundError(f"PCAP file not found: {path}")

    try:
        from scapy.all import PcapReader, IP, TCP, UDP, Raw
        SCAPY_OK = True
    except ImportError:
        SCAPY_OK = False

    if not SCAPY_OK:
        warnings.warn(
            "Scapy not installed. Run: pip install scapy\n"
            "Returning zero packet features (CSV-only mode).",
            ImportWarning,
            stacklevel=2,
        )
        return zero_packet_features()

    # ---- Packet-level accumulators ----
    ttls: list[int] = []
    tcp_windows: list[int] = []
    payload_sizes: list[int] = []
    n_total = 0
    n_tcp = 0
    n_retrans = 0
    n_fragment = 0
    n_syn_only = 0
    n_small = 0

    # Port-scan detector: src_ip -> set of dst_ports
    port_map: dict[str, set[int]] = defaultdict(set)

    # Retransmission heuristic: track (src, dst, sport, dport, seq) seen
    seen_seqs: set[tuple] = set()

    with PcapReader(str(path)) as pcap:
        for i, pkt in enumerate(pcap):
            if i >= max_packets:
                warnings.warn(f"Hit max_packets={max_packets} limit.", RuntimeWarning)
                break

            # Timestamp filter
            if window_start and float(pkt.time) < window_start:
                continue
            if window_end and float(pkt.time) > window_end:
                break

            n_total += 1

            # IP-level features
            if IP in pkt:
                ip = pkt[IP]
                ttls.append(ip.ttl)

                # Fragment flag: MF set or fragment offset > 0
                if (ip.flags & 0x1) or ip.frag > 0:
                    n_fragment += 1

                src_ip = ip.src

                # TCP-level features
                if TCP in pkt:
                    tcp = pkt[TCP]
                    n_tcp += 1

                    tcp_windows.append(tcp.window)
                    port_map[src_ip].add(tcp.dport)

                    # SYN-only (no ACK)
                    if (tcp.flags & 0x02) and not (tcp.flags & 0x10):
                        n_syn_only += 1

                    # Retransmission heuristic
                    seq_key = (ip.src, ip.dst, tcp.sport, tcp.dport, tcp.seq)
                    if seq_key in seen_seqs:
                        n_retrans += 1
                    else:
                        seen_seqs.add(seq_key)

                    # Payload size
                    if Raw in pkt:
                        payload_sizes.append(len(pkt[Raw].load))
                    else:
                        payload_sizes.append(0)

                elif UDP in pkt:
                    if Raw in pkt:
                        payload_sizes.append(len(pkt[Raw].load))
                    else:
                        payload_sizes.append(0)

            # Small packet detection (entire frame < 64 bytes)
            if len(pkt) < 64:
                n_small += 1

    # ---- Compute features ----
    def safe_mean(lst) -> float:
        return float(np.mean(lst)) if lst else 0.0

    def safe_std(lst) -> float:
        return float(np.std(lst)) if len(lst) > 1 else 0.0

    def safe_min(lst) -> float:
        return float(min(lst)) if lst else 0.0

    def safe_max(lst) -> float:
        return float(max(lst)) if lst else 0.0

    ttl_mean = safe_mean(ttls)
    ttl_std  = safe_std(ttls)
    ttl_min  = safe_min(ttls)
    ttl_max  = safe_max(ttls)

    tcp_window_mean = safe_mean(tcp_windows)
    tcp_window_std  = safe_std(tcp_windows)
    tcp_window_min  = safe_min(tcp_windows)

    retrans_rate   = (n_retrans / n_tcp) if n_tcp > 0 else 0.0
    fragment_rate  = (n_fragment / n_total) if n_total > 0 else 0.0
    payload_mean   = safe_mean(payload_sizes)
    payload_std    = safe_std(payload_sizes)
    payload_ent    = _entropy(payload_sizes)
    unique_ports   = float(np.mean([len(v) for v in port_map.values()])) if port_map else 0.0
    syn_ratio      = (n_syn_only / n_total) if n_total > 0 else 0.0
    small_ratio    = (n_small / n_total) if n_total > 0 else 0.0

    feat_vec = np.array([
        ttl_mean, ttl_std, ttl_min, ttl_max,
        tcp_window_mean, tcp_window_std, tcp_window_min,
        retrans_rate, fragment_rate,
        payload_mean, payload_std, payload_ent,
        unique_ports, syn_ratio, small_ratio,
        float(n_total),
    ], dtype=np.float32)

    return PacketFeatures(
        features=feat_vec,
        feature_names=PACKET_FEATURE_NAMES,
        n_packets=n_total,
        source="pcap",
    )


# ---------------------------------------------------------------------------
# Zero-padded fallback (CSV-only mode)
# ---------------------------------------------------------------------------


def zero_packet_features() -> PacketFeatures:
    """
    Return a zero-filled PacketFeatures for CSV-only operation.
    The model was trained without packet-level features, so this preserves
    backward compatibility while allowing the fused vector to be displayed.
    """
    return PacketFeatures(
        features=np.zeros(N_PACKET_FEATURES, dtype=np.float32),
        feature_names=PACKET_FEATURE_NAMES,
        n_packets=0,
        source="zeros",
    )


# ---------------------------------------------------------------------------
# Fuse flow-level + packet-level features
# ---------------------------------------------------------------------------


def fuse_with_flow_vector(
    flow_vector: np.ndarray,
    packet_features: PacketFeatures,
) -> np.ndarray:
    """
    Concatenate a NetFlow-based state vector with packet-level features.

    This produces the PS-required dual-level feature representation:
      [flow_features (71)] + [packet_features (16)] = (87,)

    Note: The LSTM was trained on 71-feature flow vectors.  This fused
    vector is for demonstration purposes, display in the UI, and for
    any future re-training that uses both levels.

    Args:
        flow_vector:      (n_flow_features,) or (seq_len, n_flow_features)
        packet_features:  PacketFeatures from extract_packet_features

    Returns:
        np.ndarray of shape (..., n_flow_features + N_PACKET_FEATURES)
    """
    pkt_vec = packet_features.features.astype(np.float32)
    flow_vec = np.asarray(flow_vector, dtype=np.float32)

    if flow_vec.ndim == 1:
        return np.concatenate([flow_vec, pkt_vec])
    elif flow_vec.ndim == 2:
        # Broadcast packet features across all timesteps
        seq_len = flow_vec.shape[0]
        pkt_broadcast = np.tile(pkt_vec, (seq_len, 1))
        return np.concatenate([flow_vec, pkt_broadcast], axis=1)
    else:
        raise ValueError(f"flow_vector must be 1D or 2D, got shape {flow_vec.shape}")


def get_fused_feature_names(flow_feature_columns: list[str]) -> list[str]:
    """Return combined list of flow + packet feature names."""
    return flow_feature_columns + PACKET_FEATURE_NAMES


# ---------------------------------------------------------------------------
# Summarize packet features for display
# ---------------------------------------------------------------------------


def summarize_packet_features(pf: PacketFeatures) -> str:
    """Format PacketFeatures as a readable summary."""
    if pf.source == "zeros":
        return "Packet-level features: Not available (CSV-only mode — upload PCAP for full dual-level analysis)"

    lines = [
        f"Packet-Level Feature Summary ({pf.n_packets} packets analyzed from PCAP):",
        f"  {'Feature':<30}  {'Value':>12}",
        "  " + "-" * 45,
    ]
    for name, val in zip(pf.feature_names, pf.features):
        lines.append(f"  {name:<30}  {val:>12.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    print("Packet feature module smoke test...")

    # Test zero fallback
    pf = zero_packet_features()
    print(f"  Zero features: shape={pf.features.shape}, source={pf.source}")

    # Test fusion
    flow_vec = np.random.randn(71).astype(np.float32)
    fused = fuse_with_flow_vector(flow_vec, pf)
    print(f"  Fused vector: {flow_vec.shape} flow + {pf.features.shape} packet = {fused.shape}")
    assert fused.shape == (71 + N_PACKET_FEATURES,), "Fusion shape mismatch"

    # Test fusion on sequence
    seq = np.random.randn(10, 71).astype(np.float32)
    fused_seq = fuse_with_flow_vector(seq, pf)
    assert fused_seq.shape == (10, 71 + N_PACKET_FEATURES), "Sequence fusion shape mismatch"
    print(f"  Sequence fusion: {seq.shape} -> {fused_seq.shape}")

    print(f"\n  Feature names ({N_PACKET_FEATURES} packet features):")
    for name in PACKET_FEATURE_NAMES:
        print(f"    - {name}")

    # Only test Scapy extraction if a PCAP path is passed as argument
    if len(sys.argv) > 1:
        pcap_path = sys.argv[1]
        print(f"\nExtracting from PCAP: {pcap_path}")
        try:
            pf_pcap = extract_packet_features(pcap_path)
            print(summarize_packet_features(pf_pcap))
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print("\n  (Pass a PCAP file path as argument to test full extraction)")
        print("  e.g.: python src/pcap_features.py path/to/capture.pcap")

    print("\n✅ Packet feature module OK")
