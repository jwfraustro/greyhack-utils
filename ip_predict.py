#!/usr/bin/env python3
"""
Crack an IP address from (worldSeed, BSSID) by brute-forcing the 2^32 IP space.

The forward chain (mirroring ServerMap.RouterNode):
  network_seed = world_seed + IP.GetSeedFromIP(ip)
  rng = new System.Random(network_seed)
  rng.Next(4)                # TLD index, discarded for BSSID-only matching
  bssid = ":".join(f"{rng.Next(256):02X}" for _ in range(6))

For a fixed world_seed, the mapping IP → BSSID is effectively bijective
(48 bits of output, 32 bits of input → expected collisions ~1/65536).
So given a BSSID, brute-forcing the IP space yields ~1 hit on a modern GPU
in seconds-to-minutes.

Usage:
  ip_predict.py --seed -1285005987 --bssid AA:BB:CC:DD:EE:FF
  ip_predict.py --seed -1285005987 --bssid AA:BB:CC:DD:EE:FF \\
                --domain www.somehost.net   # optional disambiguation
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List

import numpy as np

from python_tools.appleseed import (
    _CUDA_AVAILABLE,
    BASE_CONSONANTS,
    BASE_VOWELS,
    TLDS,
    DotNetRandom,
    get_seed_from_ip_like_csharp,
    ip_to_unique_name,
    normalize_bssid,
    normalize_domain,
    shuffle_array,
    to_int32,
)
from router_predict import predict_router

if _CUDA_AVAILABLE:
    from numba import cuda
    from numba import int32 as numba_int32

    # Reuse the validated init/sample primitives from appleseed. They're
    # @cuda.jit(device=True) functions in that module's namespace.
    from python_tools.appleseed import _dn_init, _dn_sample


# ---------------------------------------------------------------------------
# CUDA kernel: brute-force the IP space against a BSSID observation.
# ---------------------------------------------------------------------------
#
# Each thread represents one candidate IP (uint32, big-endian byte ordering,
# packed via (a<<24)|(b<<16)|(c<<8)|d). From the IP we derive:
#   - ip_seed_signed = BitConverter.ToInt32 of the bytes in network order
#     (which in C# means little-endian read of the network-order buffer)
#   - network_seed   = (world_seed + ip_seed_signed) as int32
#
# Then we init DotNetRandom(network_seed), draw 7 ints:
#   - Next(4)    — TLD, discarded (the draw still advances RNG state)
#   - Next(256) × 6 — the BSSID bytes
#
# Compare against the observed bytes; record matches.
#
# Cost per candidate: ~220 ops for _dn_init, 7 draws (~20 ops each).
# A modern GPU should chew through 2^32 candidates in seconds.

if _CUDA_AVAILABLE:

    @cuda.jit
    def _cuda_bssid_search(
        world_seed,  # int32: the known world seed
        target_bytes,  # int32[6]: the 6 BSSID bytes (0..255)
        ip_offset,  # int64: added to thread_id to form the candidate IP
        ip_limit,  # int64: upper bound on thread_id (exclusive)
        out_ips,  # uint32[buffer]: matching IPs (big-endian packed)
        counter,  # int32[1]: atomic write index
    ):
        tid = cuda.grid(1)
        if tid >= ip_limit:
            return

        # ip_be is the IP packed as (a<<24)|(b<<16)|(c<<8)|d
        ip_be = (tid + ip_offset) & 0xFFFFFFFF

        # ip_seed_signed = little-endian read of the network-order bytes:
        #   ip_seed = a | (b<<8) | (c<<16) | (d<<24), interpreted as int32
        # Equivalent: byte-reverse ip_be, then signed-cast.
        a = (ip_be >> 24) & 0xFF
        b = (ip_be >> 16) & 0xFF
        c = (ip_be >> 8) & 0xFF
        d = (ip_be) & 0xFF
        ip_seed_u = a | (b << 8) | (c << 16) | (d << 24)
        # Sign-cast to int32
        if ip_seed_u >= 0x80000000:
            ip_seed = ip_seed_u - 0x100000000
        else:
            ip_seed = ip_seed_u

        # network_seed = (world_seed + ip_seed) as int32
        net_seed_64 = world_seed + ip_seed
        net_seed_64 &= 0xFFFFFFFF
        if net_seed_64 >= 0x80000000:
            net_seed = net_seed_64 - 0x100000000
        else:
            net_seed = net_seed_64

        # Run the RNG.
        state = cuda.local.array(56, dtype=numba_int32)
        _dn_init(state, net_seed)
        inext = 0
        inextp = 21

        # First draw is Next(4) for TLD — we just need to advance state.
        # Implement Next(maxValue) = (int)(Sample() * maxValue) inline.
        sample, inext, inextp = _dn_sample(state, inext, inextp)
        # tld_idx = (sample * 4) // MBIG — discarded, but the draw advances state.

        # Six BSSID byte draws.
        for byte_i in range(6):
            sample, inext, inextp = _dn_sample(state, inext, inextp)
            byte_val = (sample * 256) // 2147483647
            if byte_val != target_bytes[byte_i]:
                return

        # All 6 bytes matched — record the IP.
        idx = cuda.atomic.add(counter, 0, 1)
        if idx < out_ips.size:
            out_ips[idx] = ip_be


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _ip_be_to_str(ip_be: int) -> str:
    a = (ip_be >> 24) & 0xFF
    b = (ip_be >> 16) & 0xFF
    c = (ip_be >> 8) & 0xFF
    d = ip_be & 0xFF
    return f"{a}.{b}.{c}.{d}"


def _verify_match_cpu(world_seed: int, ip_be: int, target_bytes: List[int]) -> bool:
    """Sanity-check a GPU match by re-running the chain on CPU."""
    ip_str = _ip_be_to_str(ip_be)
    ip_seed = get_seed_from_ip_like_csharp(ip_str)
    network_seed = to_int32(world_seed + ip_seed)
    rng = DotNetRandom(network_seed)
    rng.next(max_value=4)  # TLD
    for i in range(6):
        if rng.next(max_value=256) != target_bytes[i]:
            return False
    return True


def crack_ip_from_bssid(
    world_seed: int,
    target_bytes: List[int],
    ip_range_start: int = 0,
    ip_range_end: int = (1 << 32) - 1,
    buffer_size: int = 256,
) -> List[int]:
    """Brute-force the IP space on GPU and return all matching IPs (as ip_be).

    target_bytes is a list of 6 ints (0..255).
    Returns a list of uint32 packed big-endian IPs.
    """
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA is not available")

    if ip_range_start < 0 or ip_range_end >= (1 << 32) or ip_range_start > ip_range_end:
        raise ValueError(f"ip_range invalid: [{ip_range_start}, {ip_range_end}]")

    target_arr = np.array(target_bytes, dtype=np.int32)
    d_target = cuda.to_device(target_arr)

    out_ips = np.zeros(buffer_size, dtype=np.uint32)
    d_out = cuda.to_device(out_ips)
    counter = np.zeros(1, dtype=np.int32)
    d_counter = cuda.to_device(counter)

    threads_per_block = 128  # local array uses ~224 bytes per thread

    # Process in chunks of up to 2^31 to keep numba's grid indexing happy.
    chunk_max = (
        1 << 30
    )  # 1B candidates per launch — comfortably under int32 grid limits
    cursor = ip_range_start
    total_width = (ip_range_end - ip_range_start) + 1

    t0 = time.time()
    while cursor <= ip_range_end:
        chunk_end = min(ip_range_end, cursor + chunk_max - 1)
        width = (chunk_end - cursor) + 1
        blocks_per_grid = (width + threads_per_block - 1) // threads_per_block

        _cuda_bssid_search[blocks_per_grid, threads_per_block](
            np.int32(world_seed),
            d_target,
            np.int64(cursor),
            np.int64(width),
            d_out,
            d_counter,
        )

        elapsed = time.time() - t0
        progress = ((chunk_end - ip_range_start + 1) / total_width) * 100
        print(
            f"  ...processed {chunk_end - ip_range_start + 1:,}/{total_width:,} "
            f"IPs ({progress:.1f}%) in {elapsed:.1f}s",
            flush=True,
        )
        cursor = chunk_end + 1

    out_ips = d_out.copy_to_host()
    counter = d_counter.copy_to_host()
    total = int(counter[0])
    stored = min(total, buffer_size)

    if total > buffer_size:
        print(
            f"WARNING: output buffer overflow — {total} matches but only "
            f"{buffer_size} stored. Increase --buffer-size.",
            file=sys.stderr,
            flush=True,
        )

    return out_ips[:stored].tolist()


def _disambiguate_by_domain(
    world_seed: int, ip_bes: List[int], observed_domain: str
) -> List[int]:
    """If multiple IPs match the BSSID, narrow by re-deriving the domain.

    The domain is purely IP-derived once we know the shuffled consonant/vowel
    tables (which depend only on world_seed). No RNG involvement per IP beyond
    the TLD draw, which we already consumed in the brute force.
    """
    shuffled_cons = shuffle_array(BASE_CONSONANTS, world_seed)
    shuffled_vow = shuffle_array(BASE_VOWELS, world_seed)

    obs_norm = normalize_domain(observed_domain)

    keep: List[int] = []
    for ip_be in ip_bes:
        ip_str = _ip_be_to_str(ip_be)
        name = ip_to_unique_name(ip_str, shuffled_cons, shuffled_vow).lower()

        # Re-run RNG just for the TLD index.
        ip_seed = get_seed_from_ip_like_csharp(ip_str)
        network_seed = to_int32(world_seed + ip_seed)
        rng = DotNetRandom(network_seed)
        tld = TLDS[rng.next(max_value=len(TLDS))]

        candidate_domain = f"www.{name}.{tld}"
        if candidate_domain == obs_norm:
            keep.append(ip_be)
    return keep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crack an IP address from a known (worldSeed, BSSID)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="The world seed (signed int32, e.g. -1285005987)",
    )
    parser.add_argument(
        "--bssid",
        type=str,
        required=True,
        help="The observed BSSID, e.g. AA:BB:CC:DD:EE:FF",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Optional observed domain (e.g. www.foo.net) used to disambiguate "
        "the very rare case where 2+ IPs collide on the same BSSID.",
    )
    parser.add_argument(
        "--ip-start",
        type=int,
        default=0,
        help="Start of IP search range as uint32 (default: 0).",
    )
    parser.add_argument(
        "--ip-end",
        type=int,
        default=(1 << 32) - 1,
        help="End of IP search range as uint32 (default: 2^32 - 1).",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=256,
        help="Max candidates stored on the GPU before re-running with a "
        "larger buffer (default: 256 — overkill for expected 1 match).",
    )
    parser.add_argument(
        "--essid",
        type=str,
        default=None,
        help="Optional ESSID to verify in the rare case of multiple BSSID matches. ",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not _CUDA_AVAILABLE:
        print(
            "CUDA is not available. This tool is GPU-only — a CPU version "
            "would take days. Install numba with CUDA support and a CUDA-capable "
            "GPU, then retry.",
            file=sys.stderr,
        )
        return 1

    # Parse BSSID
    bssid_norm = normalize_bssid(args.bssid)
    try:
        target_bytes = [int(b, 16) for b in bssid_norm.split(":")]
    except ValueError:
        print(f"Invalid BSSID format: {args.bssid!r}", file=sys.stderr)
        return 1
    if len(target_bytes) != 6 or any(b < 0 or b > 255 for b in target_bytes):
        print(
            f"BSSID must be 6 colon-separated hex bytes; got {args.bssid!r}",
            file=sys.stderr,
        )
        return 1

    print(
        f"Searching IPs in [{args.ip_start}, {args.ip_end}] "
        f"({args.ip_end - args.ip_start + 1:,} candidates) for BSSID {bssid_norm} "
        f"under world seed {args.seed}",
        flush=True,
    )

    t0 = time.time()
    matches = crack_ip_from_bssid(
        args.seed, target_bytes, args.ip_start, args.ip_end, args.buffer_size
    )
    elapsed = time.time() - t0
    print(
        f"GPU search complete in {elapsed:.2f}s: {len(matches)} match(es)", flush=True
    )

    if not matches:
        print("No matching IP found in range.")
        return 1

    # Sanity-check matches on CPU (catches any RNG-port discrepancies)
    verified: List[int] = []
    for ip_be in matches:
        if _verify_match_cpu(args.seed, ip_be, target_bytes):
            verified.append(ip_be)
        else:
            print(
                f"WARNING: GPU match {_ip_be_to_str(ip_be)} failed CPU verification — "
                f"GPU/CPU RNG drift; report as a bug.",
                file=sys.stderr,
            )

    # Disambiguate if needed
    if len(verified) > 1 and args.domain:
        before = len(verified)
        verified = _disambiguate_by_domain(args.seed, verified, args.domain)
        print(
            f"Disambiguation with domain {args.domain!r}: {before} → {len(verified)}",
            flush=True,
        )

    if args.essid and len(verified) > 1:
        for ip_be in verified:
            ip_str = _ip_be_to_str(ip_be)
            result = predict_router(args.seed, ip_str)
            if result.essid == args.essid:
                verified = [ip_be]
                print(
                    f"Disambiguation with ESSID {args.essid!r} succeeded: "
                    f"IP is {_ip_be_to_str(ip_be)}",
                    flush=True,
                )
                print("\nRouter prediction details:")
                print(f"  IP Address: {result.ip_address}")
                print(f"  Domain: {result.domain}")
                print(f"  ESSID: {result.essid}")
                print(f"  BSSID: {result.bssid}")
                print(f"  Router Password: {result.router_password}")
                break
        else:
            print(
                f"Disambiguation with ESSID {args.essid!r} failed: no candidates had that ESSID.",
                flush=True,
            )
            verified = []

    if not verified:
        print("No IPs survived verification.")
        return 1

    print(f"\nFound {len(verified)} matching IP(s):")
    for ip_be in verified:
        print(f"  {_ip_be_to_str(ip_be)}")

    if len(verified) > 1 and not args.domain:
        print(
            "\nMultiple matches: provide --domain (the observed www.X.Y address) "
            "to narrow down. Or pick the right one by running router_predict.py "
            "on each and matching other observables.",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
