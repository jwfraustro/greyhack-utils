#!/usr/bin/env python3
"""
Crack Grey Hack world seeds using full-seed observables.

This script mirrors the C# generation path used in ServerMap.RouterNode:
  seed = worldSeed + IP.GetSeedFromIP(ipAddress)
  random = new System.Random(seed)
  webAddress = "www." + IpGenerator.GetDomainName(ipAddress, random)
  bssid = Networking.GeneraMacAddress(random)

Unlike site type alone (which leaks only 4 bits), domain/BSSID depend on full
32-bit seed behavior through System.Random and therefore can disambiguate seeds.

Input JSON format (list of objects):
[
  {
    "ip": "99.71.91.182",
    "site_type": 7,
    "domain": "www.somehost.net",
    "bssid": "AA:BB:CC:DD:EE:FF"
  }
]

Any of site_type/domain/bssid can be omitted, but at least one observable per
entry should be provided.

For best performance, provide multiple site_type observations: the CUDA
prefilter eliminates 15/16 of seeds per site_type observation in pure arithmetic
before any expensive RNG work runs on CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

# CUDA is optional; the script falls back to pure CPU search if numba is not available.
try:
    from numba import cuda
    from numba import int8 as numba_int8
    from numba import int32 as numba_int32
    _CUDA_AVAILABLE = cuda.is_available()
except Exception:
    cuda = None
    numba_int8 = None
    numba_int32 = None
    _CUDA_AVAILABLE = False

MBIG = 2147483647
MSEED = 161803398

BASE_CONSONANTS = list("bcdfghjklmnpqrstvwxyz")
BASE_VOWELS = list("aeiou")
TLDS = ("com", "net", "org", "info")

TIPO_RED = [
    "Comisaria",
    "Universidades",
    "Supermercados",
    "FastFood",
    "Taller",
    "MobileShop",
    "Hospitales",
    "Bancos",
    "Particulares",
    "MailServices",
    "HackShop",
    "TiendaInformatica",
    "NetServices",
    "HardwareManufacturer",
    "Neurobox",
    "CurrencyCreation",
]


def to_uint32(value: int) -> int:
    return value & 0xFFFFFFFF


def to_int32(value: int) -> int:
    value &= 0xFFFFFFFF
    if value >= 0x80000000:
        value -= 0x100000000
    return value


def ip_to_uint_be(ip: str) -> int:
    a, b, c, d = (int(x) for x in ip.split("."))
    return ((a & 0xFF) << 24) | ((b & 0xFF) << 16) | ((c & 0xFF) << 8) | (d & 0xFF)


def get_seed_from_ip_like_csharp(ip: str) -> int:
    # C#: BitConverter.ToInt32(IPAddress.Parse(ip).GetAddressBytes(), 0)
    # IPAddress bytes are network order [a,b,c,d], and BitConverter is little-endian.
    a, b, c, d = (int(x) for x in ip.split("."))
    raw = (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24)
    return to_int32(raw)


class DotNetRandom:
    """Accurate port of the decompiled System.Random implementation."""

    def __init__(self, seed: int):
        self._seed_array = [0] * 56
        self._inext = 0
        self._inextp = 21

        num = 0
        num2 = MBIG if seed == -2147483648 else abs(seed)
        num3 = MSEED - num2
        self._seed_array[55] = num3
        num4 = 1

        for _ in range(1, 55):
            num += 21
            if num >= 55:
                num -= 55
            self._seed_array[num] = num4
            num4 = to_int32(num3 - num4)
            if num4 < 0:
                num4 += MBIG
            num3 = self._seed_array[num]

        for _ in range(1, 5):
            for k in range(1, 56):
                num5 = k + 30
                if num5 >= 55:
                    num5 -= 55
                self._seed_array[k] = to_int32(
                    self._seed_array[k] - self._seed_array[1 + num5]
                )
                if self._seed_array[k] < 0:
                    self._seed_array[k] += MBIG

        self._inext = 0
        self._inextp = 21

    def _internal_sample(self) -> int:
        num = self._inext
        num2 = self._inextp

        num += 1
        if num >= 56:
            num = 1
        num2 += 1
        if num2 >= 56:
            num2 = 1

        num3 = self._seed_array[num] - self._seed_array[num2]
        if num3 == MBIG:
            num3 -= 1
        if num3 < 0:
            num3 += MBIG

        self._seed_array[num] = num3
        self._inext = num
        self._inextp = num2
        return num3

    def _sample(self) -> float:
        return float(self._internal_sample()) * 4.656612875245797e-10

    def next(
        self, min_value: Optional[int] = None, max_value: Optional[int] = None
    ) -> int:
        if min_value is None and max_value is None:
            return self._internal_sample()

        if min_value is None:
            if max_value < 0:
                raise ValueError("max_value must be >= 0")
            return int(self._sample() * float(max_value))

        if max_value is None:
            raise ValueError("max_value is required when min_value is provided")
        if min_value > max_value:
            raise ValueError("min_value cannot be greater than max_value")

        num = int(max_value) - int(min_value)
        if num <= MBIG:
            return int(self._sample() * float(num)) + int(min_value)

        # Large-range path from C# Random.Next(min,max)
        sample = self._internal_sample()
        if self._internal_sample() % 2 == 0:
            sample = -sample
        large = (float(sample) + 2147483646.0) / 4294967293.0
        return int(large * float(num)) + int(min_value)


def shuffle_array(chars: Sequence[str], seed: int) -> List[str]:
    rng = DotNetRandom(seed)
    decorated = [(rng.next(), idx, ch) for idx, ch in enumerate(chars)]
    decorated.sort(key=lambda t: (t[0], t[1]))
    return [ch for _, _, ch in decorated]


def ip_to_unique_name(
    ip: str, shuffled_consonants: Sequence[str], shuffled_vowels: Sequence[str]
) -> str:
    num = ip_to_uint_be(ip)
    out: List[str] = []
    use_consonant = True

    while num > 0:
        if use_consonant:
            idx = num % len(shuffled_consonants)
            num //= len(shuffled_consonants)
            out.append(shuffled_consonants[idx])
        else:
            idx = num % len(shuffled_vowels)
            num //= len(shuffled_vowels)
            out.append(shuffled_vowels[idx])
        use_consonant = not use_consonant

    out.reverse()
    return "".join(out)


@dataclass(frozen=True)
class Observation:
    ip: str
    ip_uint_be: int
    ip_seed_signed: int
    site_type: Optional[int] = None
    domain: Optional[str] = None
    bssid: Optional[str] = None


@dataclass
class SearchConfig:
    start_seed: int
    end_seed: int
    workers: int
    chunk_size: int
    max_results: int
    use_cuda: bool = True
    cuda_buffer_size: int = 0  # 0 = auto-size


# ---------------------------------------------------------------------------
# CUDA prefilter (phase 1: site-type only)
# ---------------------------------------------------------------------------

if _CUDA_AVAILABLE:
    @cuda.jit
    def _cuda_sitetype_filter(
        ip_uints,         # uint32[N]: big-endian IP integers
        expected_types,   # int32[N]: expected site type per observation
        seed_offset,      # int64: added to thread_id to form the unsigned seed
        start_signed,     # int64: signed seed corresponding to thread_id == 0
        seed_limit,       # int64: upper bound on thread_id (exclusive)
        out_seeds,        # int32[buffer_size]: output buffer for survivors
        counter,          # int32[1]: atomic counter for output index
    ):
        tid = cuda.grid(1)
        if tid >= seed_limit:
            return

        # seed_uint is the unsigned 32-bit form used in the XOR
        seed_uint = (tid + seed_offset) & 0xFFFFFFFF

        # Check all site_type observations; bail on first mismatch
        n = ip_uints.shape[0]
        for i in range(n):
            computed = ((ip_uints[i] ^ seed_uint) & 0x7FFFFFFF) % 16
            if computed != expected_types[i]:
                return

        # Survived all checks; record the signed seed
        # signed = start_signed + tid (works for both halves)
        seed_signed = start_signed + tid
        idx = cuda.atomic.add(counter, 0, 1)
        if idx < out_seeds.size:
            out_seeds[idx] = seed_signed


def _cuda_prefilter_survivors(
    observations: Sequence[Observation],
    start_seed: int,
    end_seed: int,
    buffer_size: int,
) -> Tuple[List[int], int]:
    """Run CUDA phase-1 prefilter over [start_seed, end_seed] inclusive.

    Returns (survivors, total_found). If total_found > len(survivors), the
    buffer overflowed and the caller should rerun with a larger buffer or
    fall back to CPU search.
    """
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA is not available")

    site_obs = [o for o in observations if o.site_type is not None]
    if not site_obs:
        raise ValueError("CUDA prefilter requires at least one site_type observation")

    ip_uints = np.array([o.ip_uint_be for o in site_obs], dtype=np.uint32)
    expected_types = np.array([o.site_type for o in site_obs], dtype=np.int32)

    d_ip_uints = cuda.to_device(ip_uints)
    d_expected = cuda.to_device(expected_types)

    out_seeds = np.full(buffer_size, 0, dtype=np.int32)
    d_out = cuda.to_device(out_seeds)
    counter = np.zeros(1, dtype=np.int32)
    d_counter = cuda.to_device(counter)

    # The full int32 range is 2^32 values, which overflows int32 grid indexing.
    # Split into two halves: [start, mid] and [mid+1, end], each up to 2^31 wide.
    threads_per_block = 256

    # We always split at zero (the natural signed/unsigned boundary) when the
    # range spans both signs, since each half then fits comfortably in 2^31.
    if start_seed < 0 and end_seed >= 0:
        ranges = [(start_seed, -1), (0, end_seed)]
    else:
        ranges = [(start_seed, end_seed)]

    for r_start, r_end in ranges:
        width = (r_end - r_start) + 1
        if width <= 0:
            continue

        # seed_offset: added to thread_id (which is unsigned-ish, 0..width-1)
        # to form the *unsigned* 32-bit seed value used in the XOR.
        # For r_start >= 0: thread 0 represents seed r_start, unsigned = r_start.
        # For r_start <  0: thread 0 represents seed r_start, unsigned = r_start + 2^32.
        if r_start >= 0:
            seed_offset = r_start
        else:
            seed_offset = r_start + 0x100000000

        start_signed = r_start  # signed = start_signed + tid

        blocks_per_grid = (width + threads_per_block - 1) // threads_per_block

        _cuda_sitetype_filter[blocks_per_grid, threads_per_block](
            d_ip_uints,
            d_expected,
            np.int64(seed_offset),
            np.int64(start_signed),
            np.int64(width),
            d_out,
            d_counter,
        )

    # Pull results back
    out_seeds = d_out.copy_to_host()
    counter = d_counter.copy_to_host()
    total_found = int(counter[0])
    stored = min(total_found, buffer_size)
    survivors = out_seeds[:stored].tolist()
    return survivors, total_found


# ---------------------------------------------------------------------------
# CUDA phase 2-3 verifier (domain check)
# ---------------------------------------------------------------------------
#
# Strategy: take phase-1 survivors and verify each against a single (ip, domain)
# observation. The kernel:
#   1. Inits DotNetRandom(worldSeed) and produces shuffledConsonants
#   2. Inits DotNetRandom(worldSeed) again and produces shuffledVowels
#   3. Walks ip_to_unique_name positions and checks each produced character
#      against the expected character from the observed domain
#   4. If name matches, inits DotNetRandom(networkSeed) and checks the TLD
#
# The shuffle is the expensive part. Two DotNetRandom inits (~220 ops each)
# plus two small insertion sorts (~210 compares for 21 elements, ~10 for 5).
# Even at 1 microsecond per seed on a modern GPU, 268M seeds finishes in ~270ms.
#
# Remaining observations (if any) are re-verified on CPU for the handful of
# seeds that pass — that's a small list, so CPU cost is negligible.

if _CUDA_AVAILABLE:

    @cuda.jit(device=True, inline=True)
    def _dn_init(state, seed):
        """Initialize a DotNetRandom state array in place.

        state: int32[56] - per-thread local array
        seed:  int32     - signed seed value
        """
        # num2 = MBIG if seed == -2147483648 else abs(seed)
        if seed == -2147483648:
            num2 = 2147483647
        elif seed < 0:
            num2 = -seed
        else:
            num2 = seed

        num3 = 161803398 - num2
        state[55] = num3
        num4 = 1
        num = 0

        for _ in range(1, 55):
            num += 21
            if num >= 55:
                num -= 55
            state[num] = num4
            num4 = num3 - num4
            if num4 < 0:
                num4 += 2147483647
            num3 = state[num]

        for _ in range(1, 5):
            for k in range(1, 56):
                num5 = k + 30
                if num5 >= 55:
                    num5 -= 55
                state[k] -= state[1 + num5]
                if state[k] < 0:
                    state[k] += 2147483647

        # _inext = 0, _inextp = 21 — we'll store these in state[0] via convention
        # but since we manage them as locals in the calling kernel, just return
        return

    @cuda.jit(device=True, inline=True)
    def _dn_sample(state, inext, inextp):
        """Produce one internal_sample value. Returns (value, new_inext, new_inextp)."""
        inext += 1
        if inext >= 56:
            inext = 1
        inextp += 1
        if inextp >= 56:
            inextp = 1

        num3 = state[inext] - state[inextp]
        if num3 == 2147483647:
            num3 -= 1
        if num3 < 0:
            num3 += 2147483647

        state[inext] = num3
        return num3, inext, inextp

    @cuda.jit(device=True, inline=True)
    def _stable_sort_keys_21(keys, indices):
        """Insertion sort for 21 elements; ties broken by original index.

        keys:    int32[21] - sort keys (will be permuted)
        indices: int8[21]  - 0..20 originally; will be permuted to match
        """
        for i in range(1, 21):
            kk = keys[i]
            ii = indices[i]
            j = i - 1
            # Stable: only swap when strictly greater
            while j >= 0 and (keys[j] > kk or (keys[j] == kk and indices[j] > ii)):
                keys[j + 1] = keys[j]
                indices[j + 1] = indices[j]
                j -= 1
            keys[j + 1] = kk
            indices[j + 1] = ii

    @cuda.jit(device=True, inline=True)
    def _stable_sort_keys_5(keys, indices):
        """Insertion sort for 5 elements; ties broken by original index."""
        for i in range(1, 5):
            kk = keys[i]
            ii = indices[i]
            j = i - 1
            while j >= 0 and (keys[j] > kk or (keys[j] == kk and indices[j] > ii)):
                keys[j + 1] = keys[j]
                indices[j + 1] = indices[j]
                j -= 1
            keys[j + 1] = kk
            indices[j + 1] = ii

    @cuda.jit
    def _cuda_domain_verify(
        candidate_seeds,    # int32[N]: phase-1 survivors
        ip_uint_be,         # int64: big-endian IP integer for the observation
        ip_seed_signed,     # int32: GetSeedFromIP value for the observation
        expected_positions, # int8[L]: position index produced at each step (0..20 for cons, 0..4 for vow)
        expected_base_idx,  # int8[L]: expected base-array index at each step
        expected_is_cons,   # int8[L]: 1 if consonant step, 0 if vowel step
        expected_length,    # int32: number of steps L
        expected_tld_idx,   # int32: expected TLD index (0..3), or -1 to skip TLD check
        out_seeds,          # int32[buffer]: matching seeds
        counter,            # int32[1]: atomic write index
    ):
        tid = cuda.grid(1)
        if tid >= candidate_seeds.shape[0]:
            return

        world_seed = candidate_seeds[tid]

        # ---- Consonant shuffle ----
        cons_state = cuda.local.array(56, dtype=numba_int32)
        cons_keys = cuda.local.array(21, dtype=numba_int32)
        cons_idx = cuda.local.array(21, dtype=numba_int8)

        _dn_init(cons_state, world_seed)
        cons_inext = 0
        cons_inextp = 21
        for i in range(21):
            val, cons_inext, cons_inextp = _dn_sample(cons_state, cons_inext, cons_inextp)
            cons_keys[i] = val
            cons_idx[i] = i

        _stable_sort_keys_21(cons_keys, cons_idx)
        # cons_idx[k] now holds the base-array index at shuffled position k

        # ---- Vowel shuffle (fresh DotNetRandom with same seed) ----
        vow_state = cuda.local.array(56, dtype=numba_int32)
        vow_keys = cuda.local.array(5, dtype=numba_int32)
        vow_idx = cuda.local.array(5, dtype=numba_int8)

        _dn_init(vow_state, world_seed)
        vow_inext = 0
        vow_inextp = 21
        for i in range(5):
            val, vow_inext, vow_inextp = _dn_sample(vow_state, vow_inext, vow_inextp)
            vow_keys[i] = val
            vow_idx[i] = i

        _stable_sort_keys_5(vow_keys, vow_idx)

        # ---- Check the decoded name against expected ----
        for k in range(expected_length):
            pos = expected_positions[k]
            expected = expected_base_idx[k]
            if expected_is_cons[k]:
                if cons_idx[pos] != expected:
                    return
            else:
                if vow_idx[pos] != expected:
                    return

        # ---- Name matches. Check TLD via network_seed RNG ----
        if expected_tld_idx >= 0:
            # network_seed = world_seed + ip_seed_signed, as signed int32
            net_seed_64 = world_seed + ip_seed_signed
            # Wrap to int32
            net_seed_64 &= 0xFFFFFFFF
            if net_seed_64 >= 0x80000000:
                net_seed = net_seed_64 - 0x100000000
            else:
                net_seed = net_seed_64

            net_state = cuda.local.array(56, dtype=numba_int32)
            _dn_init(net_state, net_seed)
            net_inext = 0
            net_inextp = 21
            sample, net_inext, net_inextp = _dn_sample(net_state, net_inext, net_inextp)
            # TLD = int(sample / MBIG * 4)
            # Match C#: (int)(Sample() * maxValue) where Sample = internal/MBIG
            # In int math: (sample * 4) / MBIG, truncating toward zero
            tld = (sample * 4) // 2147483647
            if tld != expected_tld_idx:
                return

        # ---- Match! Record the seed. ----
        out_idx = cuda.atomic.add(counter, 0, 1)
        if out_idx < out_seeds.size:
            out_seeds[out_idx] = world_seed


def _precompute_domain_check(
    obs: Observation,
) -> Optional[Tuple[List[int], List[int], List[int], int]]:
    """Decompose an observation into the data the GPU kernel needs.

    Returns (positions, base_indices, is_consonant_flags, tld_idx) or None if
    the observation doesn't have a domain.
    """
    if obs.domain is None:
        return None

    # Strip www. prefix and .tld suffix
    domain = obs.domain
    if domain.startswith("www."):
        domain_body = domain[4:]
    else:
        domain_body = domain

    # Split off TLD
    dot = domain_body.rfind(".")
    if dot < 0:
        raise ValueError(f"Domain {obs.domain!r} has no TLD")
    name = domain_body[:dot]
    tld = domain_body[dot + 1 :]

    try:
        tld_idx = TLDS.index(tld)
    except ValueError:
        raise ValueError(f"Unrecognized TLD {tld!r} in domain {obs.domain!r}")

    # Reconstruct the position sequence by simulating ip_to_unique_name
    # with the BASE arrays (we just need positions, not characters).
    # The position sequence is purely a function of the IP integer + array lengths.
    num = obs.ip_uint_be
    positions: List[int] = []
    is_cons: List[int] = []
    use_consonant = True
    while num > 0:
        if use_consonant:
            positions.append(num % 21)
            num //= 21
            is_cons.append(1)
        else:
            positions.append(num % 5)
            num //= 5
            is_cons.append(0)
        use_consonant = not use_consonant

    # The decoded name is BUILT in reverse, then reversed before output.
    # So the i-th character of `name` corresponds to the LAST step in our loop,
    # the (i+1)-th to second-to-last, etc.
    # Concretely: positions list above is in append order (innermost first);
    # the observed `name` reads as the REVERSED order.
    positions.reverse()
    is_cons.reverse()

    if len(positions) != len(name):
        raise ValueError(
            f"Position count {len(positions)} != name length {len(name)} for {obs.domain!r}; "
            f"this indicates a mismatch in the decoding logic"
        )

    # Now figure out the expected base-array index for each character
    base_indices: List[int] = []
    for ch, is_c in zip(name, is_cons):
        if is_c:
            if ch not in BASE_CONSONANTS:
                raise ValueError(f"Character {ch!r} expected to be consonant but isn't")
            base_indices.append(BASE_CONSONANTS.index(ch))
        else:
            if ch not in BASE_VOWELS:
                raise ValueError(f"Character {ch!r} expected to be vowel but isn't")
            base_indices.append(BASE_VOWELS.index(ch))

    return positions, base_indices, is_cons, tld_idx


def _cuda_verify_survivors(
    survivors: Sequence[int],
    observation: Observation,
) -> List[int]:
    """GPU-verify the given survivors against a single domain observation."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA is not available")

    precomputed = _precompute_domain_check(observation)
    if precomputed is None:
        raise ValueError("Observation has no domain to verify against")

    positions, base_indices, is_cons, tld_idx = precomputed

    candidate_arr = np.array(survivors, dtype=np.int32)
    pos_arr = np.array(positions, dtype=np.int8)
    base_arr = np.array(base_indices, dtype=np.int8)
    cons_arr = np.array(is_cons, dtype=np.int8)

    d_candidates = cuda.to_device(candidate_arr)
    d_positions = cuda.to_device(pos_arr)
    d_base = cuda.to_device(base_arr)
    d_cons = cuda.to_device(cons_arr)

    # Output buffer: realistically only 1-2 seeds will match a domain observation
    # given the 48-bit info content, but pad to handle small collision cases.
    out_buffer_size = 256
    out_seeds = np.zeros(out_buffer_size, dtype=np.int32)
    d_out = cuda.to_device(out_seeds)
    counter = np.zeros(1, dtype=np.int32)
    d_counter = cuda.to_device(counter)

    threads_per_block = 128  # smaller blocks: kernel uses ~720 bytes of local mem per thread
    blocks_per_grid = (len(survivors) + threads_per_block - 1) // threads_per_block

    _cuda_domain_verify[blocks_per_grid, threads_per_block](
        d_candidates,
        np.int64(observation.ip_uint_be),
        np.int32(observation.ip_seed_signed),
        d_positions,
        d_base,
        d_cons,
        np.int32(len(positions)),
        np.int32(tld_idx),
        d_out,
        d_counter,
    )

    out_seeds = d_out.copy_to_host()
    counter = d_counter.copy_to_host()
    total = int(counter[0])
    stored = min(total, out_buffer_size)

    if total > out_buffer_size:
        print(
            f"WARNING: phase 2-3 buffer overflow — {total} matches but only {out_buffer_size} stored. "
            f"This is unexpected; recheck observations.",
            flush=True,
        )

    return out_seeds[:stored].tolist()


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------

class ProgressTracker:
    def __init__(self, chunks: Sequence[Tuple[int, int]], total_seeds: int):
        self._chunks = list(chunks)
        self._total_chunks = len(chunks)
        self._total_seeds = max(1, total_seeds)
        self._start_time = time.time()
        self._completed_chunks = 0
        self._completed_seeds = 0
        self._last_started: Optional[Tuple[int, int, int]] = None

    def chunk_started(self, chunk_index: int) -> None:
        start_seed, end_seed = self._chunks[chunk_index]
        self._last_started = (chunk_index, start_seed, end_seed)
        print(
            f"[chunk {chunk_index + 1}/{self._total_chunks}] checking seeds [{start_seed}, {end_seed}]",
            flush=True,
        )

    def chunk_completed(self, chunk_index: int, found_in_chunk: int) -> None:
        start_seed, end_seed = self._chunks[chunk_index]
        seeds_in_chunk = (end_seed - start_seed) + 1
        self._completed_chunks += 1
        self._completed_seeds += seeds_in_chunk

        elapsed = max(1e-9, time.time() - self._start_time)
        fraction = min(1.0, self._completed_seeds / self._total_seeds)
        rate = self._completed_seeds / elapsed
        eta = (self._total_seeds - self._completed_seeds) / rate if rate > 0 else float("inf")
        bar_width = 28
        fill = int(fraction * bar_width)
        bar = "#" * fill + "-" * (bar_width - fill)
        eta_text = "inf" if eta == float("inf") else f"{eta:.1f}s"

        print(
            f"[done {self._completed_chunks}/{self._total_chunks}] [{bar}] "
            f"{fraction * 100:6.2f}% elapsed={elapsed:8.1f}s eta={eta_text:>8} "
            f"rate={rate:10.0f} seeds/s chunk=[{start_seed}, {end_seed}] "
            f"chunk_matches={found_in_chunk}",
            flush=True,
        )

    def final_summary(self) -> None:
        elapsed = time.time() - self._start_time
        print(
            f"Search finished: completed {self._completed_chunks}/{self._total_chunks} chunks "
            f"in {elapsed:.1f}s",
            flush=True,
        )


def normalize_domain(domain: str) -> str:
    d = domain.strip().lower()
    if d.startswith("http://"):
        d = d[7:]
    if d.startswith("https://"):
        d = d[8:]
    return d.rstrip("/")


def normalize_bssid(bssid: str) -> str:
    b = bssid.strip().replace("-", ":").upper()
    return b


def load_observations(path: str) -> List[Observation]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError("Input JSON must be a list of observations")

    out: List[Observation] = []
    for i, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Observation #{i} is not an object")

        ip = item.get("ip")
        if not ip:
            raise ValueError(f"Observation #{i} missing 'ip'")

        site_type = item.get("site_type")
        domain = item.get("domain")
        bssid = item.get("bssid")

        if site_type is None and domain is None and bssid is None:
            raise ValueError(
                f"Observation #{i} has no observables; provide at least one of site_type/domain/bssid"
            )

        out.append(
            Observation(
                ip=ip,
                ip_uint_be=ip_to_uint_be(ip),
                ip_seed_signed=get_seed_from_ip_like_csharp(ip),
                site_type=int(site_type) if site_type is not None else None,
                domain=normalize_domain(domain) if domain is not None else None,
                bssid=normalize_bssid(bssid) if bssid is not None else None,
            )
        )

    return out


def predict_for_ip(
    world_seed: int,
    obs: Observation,
    shuffled_consonants: Optional[Sequence[str]],
    shuffled_vowels: Optional[Sequence[str]],
) -> Tuple[int, Optional[str], Optional[str]]:
    site_type = int(
        ((obs.ip_uint_be ^ to_uint32(world_seed)) & 0x7FFFFFFF) % len(TIPO_RED)
    )

    network_seed = to_int32(world_seed + obs.ip_seed_signed)
    rng = DotNetRandom(network_seed)

    # RouterNode constructor generates domain first, then MAC from same RNG state.
    domain = None
    if shuffled_consonants is not None and shuffled_vowels is not None:
        name = ip_to_unique_name(obs.ip, shuffled_consonants, shuffled_vowels)
        tld = TLDS[rng.next(max_value=len(TLDS))]
        domain = f"www.{name.lower()}.{tld}"
    else:
        # Even if domain isn't observed, this RNG draw still happened before BSSID generation.
        rng.next(max_value=len(TLDS))

    mac_bytes = [rng.next(max_value=256) for _ in range(6)]
    bssid = ":".join(f"{x:02X}" for x in mac_bytes)
    return site_type, domain, bssid


def candidate_matches(
    world_seed: int, observations: Sequence[Observation], needs_domain: bool
) -> bool:
    shuffled_consonants = None
    shuffled_vowels = None
    if needs_domain:
        shuffled_consonants = shuffle_array(BASE_CONSONANTS, world_seed)
        shuffled_vowels = shuffle_array(BASE_VOWELS, world_seed)

    for obs in observations:
        pred_site_type, pred_domain, pred_bssid = predict_for_ip(
            world_seed,
            obs,
            shuffled_consonants,
            shuffled_vowels,
        )

        if obs.site_type is not None and pred_site_type != obs.site_type:
            return False
        if obs.domain is not None and pred_domain != obs.domain:
            return False
        if obs.bssid is not None and pred_bssid != obs.bssid:
            return False

    return True


def candidate_matches_fast(
    world_seed: int,
    observations: Sequence[Observation],
    needs_domain: bool,
    needs_full_rng: bool,
) -> bool:
    seed_uint = world_seed & 0xFFFFFFFF

    # Phase 1: site-type filter (cheap, no RNG)
    for obs in observations:
        if obs.site_type is None:
            continue
        pred = ((obs.ip_uint_be ^ seed_uint) & 0x7FFFFFFF) % 16
        if pred != obs.site_type:
            return False

    if not needs_full_rng:
        return True

    # Phase 2: shuffle arrays (only if we passed site-type filter and need domains)
    shuffled_consonants = None
    shuffled_vowels = None
    if needs_domain:
        shuffled_consonants = shuffle_array(BASE_CONSONANTS, world_seed)
        shuffled_vowels = shuffle_array(BASE_VOWELS, world_seed)

    # Phase 3: per-IP RNG checks (domain + BSSID)
    for obs in observations:
        if obs.domain is None and obs.bssid is None:
            continue

        network_seed = to_int32(world_seed + obs.ip_seed_signed)
        rng = DotNetRandom(network_seed)

        tld_idx = rng.next(max_value=len(TLDS))

        if obs.domain is not None:
            name = ip_to_unique_name(obs.ip, shuffled_consonants, shuffled_vowels)
            pred_domain = f"www.{name.lower()}.{TLDS[tld_idx]}"
            if pred_domain != obs.domain:
                return False

        if obs.bssid is not None:
            mac_bytes = [rng.next(max_value=256) for _ in range(6)]
            pred_bssid = ":".join(f"{x:02X}" for x in mac_bytes)
            if pred_bssid != obs.bssid:
                return False

    return True


def search_chunk(
    start_seed: int,
    end_seed: int,
    observations: Sequence[Observation],
    max_results: int,
    needs_domain: bool,
    needs_full_rng: bool = True,
) -> List[int]:
    results: List[int] = []
    for seed in range(start_seed, end_seed + 1):
        if candidate_matches_fast(seed, observations, needs_domain, needs_full_rng):
            results.append(seed)
            if len(results) >= max_results:
                break
    return results


def search_survivors(
    survivors: Sequence[int],
    observations: Sequence[Observation],
    max_results: int,
    needs_domain: bool,
    needs_full_rng: bool,
) -> List[int]:
    """Run phases 2-3 against an explicit list of candidate seeds."""
    results: List[int] = []
    for seed in survivors:
        if candidate_matches_fast(seed, observations, needs_domain, needs_full_rng):
            results.append(seed)
            if len(results) >= max_results:
                break
    return results


def build_chunks(
    start_seed: int, end_seed: int, chunk_size: int
) -> List[Tuple[int, int]]:
    chunks: List[Tuple[int, int]] = []
    s = start_seed
    while s <= end_seed:
        e = min(end_seed, s + chunk_size - 1)
        chunks.append((s, e))
        s = e + 1
    return chunks


def _split_survivors(
    survivors: Sequence[int], num_chunks: int
) -> List[List[int]]:
    if num_chunks <= 1 or len(survivors) <= num_chunks:
        return [list(survivors)]
    chunk_size = (len(survivors) + num_chunks - 1) // num_chunks
    return [
        list(survivors[i : i + chunk_size])
        for i in range(0, len(survivors), chunk_size)
    ]


def _estimate_cuda_buffer_size(
    observations: Sequence[Observation], total_seeds: int, user_override: int
) -> int:
    """Estimate buffer needs for the CUDA prefilter.

    Each site_type observation eliminates ~15/16 of seeds, but observations
    can have correlated low-nibbles in their IPs, so be generous.
    """
    if user_override > 0:
        return user_override

    site_obs_count = sum(1 for o in observations if o.site_type is not None)
    # Conservative: assume each obs gives ~3 bits of effective filtering, not 4,
    # to account for low-nibble correlation between observations.
    expected = max(1, total_seeds >> (3 * site_obs_count))
    # Add 4x headroom and cap at 256M (1 GB of int32).
    return min(expected * 4, 256_000_000)


def crack_world_seed(
    observations: Sequence[Observation], cfg: SearchConfig
) -> List[int]:
    if cfg.start_seed > cfg.end_seed:
        raise ValueError("start_seed cannot be greater than end_seed")

    needs_domain = any(o.domain is not None for o in observations)
    needs_full_rng = any(o.domain is not None or o.bssid is not None for o in observations)
    has_site_types = any(o.site_type is not None for o in observations)
    total_seeds = (cfg.end_seed - cfg.start_seed) + 1

    # CUDA prefilter path: phase 1 on GPU, phases 2-3 on CPU.
    if cfg.use_cuda and _CUDA_AVAILABLE and has_site_types:
        return _crack_with_cuda(observations, cfg, needs_domain, needs_full_rng, total_seeds)

    # CPU-only path (original behavior)
    if cfg.use_cuda and not _CUDA_AVAILABLE:
        print("CUDA requested but not available; falling back to CPU search.", flush=True)
    elif cfg.use_cuda and not has_site_types:
        print("CUDA prefilter requires site_type observations; using CPU search.", flush=True)

    return _crack_with_cpu(observations, cfg, needs_domain, needs_full_rng, total_seeds)


def _crack_with_cuda(
    observations: Sequence[Observation],
    cfg: SearchConfig,
    needs_domain: bool,
    needs_full_rng: bool,
    total_seeds: int,
) -> List[int]:
    print(f"CUDA prefilter: phase 1 over seeds [{cfg.start_seed}, {cfg.end_seed}]", flush=True)

    buffer_size = _estimate_cuda_buffer_size(observations, total_seeds, cfg.cuda_buffer_size)
    print(
        f"Allocating prefilter buffer for {buffer_size:,} candidates "
        f"(~{buffer_size * 4 / 1e9:.2f} GB on device)",
        flush=True,
    )

    t0 = time.time()
    survivors, total_found = _cuda_prefilter_survivors(
        observations, cfg.start_seed, cfg.end_seed, buffer_size
    )
    phase1_elapsed = time.time() - t0

    print(
        f"Phase 1 complete in {phase1_elapsed:.2f}s: "
        f"{total_found:,} survivors out of {total_seeds:,} seeds "
        f"({total_found / total_seeds * 100:.4f}%)",
        flush=True,
    )

    if total_found > buffer_size:
        print(
            f"WARNING: buffer overflow — {total_found:,} survivors found but only "
            f"{buffer_size:,} captured. Increase --cuda-buffer-size or split the search range.",
            flush=True,
        )

    if not needs_full_rng:
        # Site-type only; survivors are the answer.
        return sorted(set(survivors))[: cfg.max_results]

    # Full-RNG observations (domain/BSSID) uniquely determine the seed, so the
    # first match IS the answer — no need to keep searching. Cap effective_max
    # at 1 so workers bail on their first hit and we stop submitting new work.
    effective_max = 1

    # If we have a domain observation, use the GPU verifier — much faster than CPU.
    # The kernel verifies one observation per call; any remaining observations
    # are re-verified on CPU for the tiny set of survivors.
    primary_domain_obs = next((o for o in observations if o.domain is not None), None)
    if primary_domain_obs is not None:
        print(
            f"Phase 2-3 (GPU): verifying {len(survivors):,} survivors against "
            f"domain {primary_domain_obs.domain!r}",
            flush=True,
        )
        t1 = time.time()
        gpu_matches = _cuda_verify_survivors(survivors, primary_domain_obs)
        phase23a_elapsed = time.time() - t1
        print(
            f"GPU verify complete in {phase23a_elapsed:.2f}s: {len(gpu_matches)} match(es)",
            flush=True,
        )

        # Re-verify on CPU against the full observation set to catch the rare case
        # where the GPU match satisfies one domain but not all observations.
        if len(gpu_matches) > 0 and len(observations) > 1:
            print(
                f"Re-verifying {len(gpu_matches)} GPU match(es) on CPU against "
                f"all {len(observations)} observation(s)...",
                flush=True,
            )
            matches = search_survivors(
                gpu_matches, observations, effective_max, needs_domain, needs_full_rng
            )
        else:
            matches = gpu_matches[:effective_max]

        return sorted(set(matches))[: cfg.max_results]

    # No domain observation — fall back to CPU phase 2-3 (BSSID-only path).
    print(
        f"Phase 2-3 (CPU): RNG-based verification on {len(survivors):,} survivors "
        f"with {cfg.workers} worker(s) (stop on first match)",
        flush=True,
    )

    t1 = time.time()
    if cfg.workers <= 1 or len(survivors) < cfg.workers * 4:
        matches = search_survivors(
            survivors, observations, effective_max, needs_domain, needs_full_rng
        )
    else:
        matches = []
        partitions = _split_survivors(survivors, cfg.workers * 4)
        executor = ProcessPoolExecutor(max_workers=cfg.workers)
        try:
            futures = {
                executor.submit(
                    search_survivors,
                    part,
                    observations,
                    effective_max,
                    needs_domain,
                    needs_full_rng,
                ): i
                for i, part in enumerate(partitions)
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    matches.extend(result)
                    if len(matches) >= effective_max:
                        # Got it. Cancel anything still queued.
                        for f in futures:
                            f.cancel()
                        break
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    phase23_elapsed = time.time() - t1
    print(
        f"Phase 2-3 complete in {phase23_elapsed:.2f}s: "
        f"{len(matches)} final match(es)",
        flush=True,
    )

    return sorted(set(matches))[: cfg.max_results]


def _crack_with_cpu(
    observations: Sequence[Observation],
    cfg: SearchConfig,
    needs_domain: bool,
    needs_full_rng: bool,
    total_seeds: int,
) -> List[int]:
    chunks = build_chunks(cfg.start_seed, cfg.end_seed, cfg.chunk_size)
    progress = ProgressTracker(chunks, total_seeds)

    # Full-RNG observations uniquely determine the seed, so the first match
    # IS the answer. Cap effective_max at 1 to stop work as soon as we hit.
    effective_max = 1 if needs_full_rng else cfg.max_results
    if needs_full_rng:
        print("Full-RNG observations present: will stop on first match.", flush=True)

    if cfg.workers <= 1:
        found: List[int] = []
        for i, (s, e) in enumerate(chunks):
            progress.chunk_started(i)
            chunk_results = search_chunk(
                s, e, observations, effective_max - len(found), needs_domain, needs_full_rng
            )
            found.extend(chunk_results)
            progress.chunk_completed(i, len(chunk_results))
            if len(found) >= effective_max:
                break
        progress.final_summary()
        return found

    found = []
    executor = ProcessPoolExecutor(max_workers=cfg.workers)
    futures = {}

    try:
        next_chunk = 0
        initial = min(cfg.workers, len(chunks))
        for _ in range(initial):
            s, e = chunks[next_chunk]
            progress.chunk_started(next_chunk)
            future = executor.submit(
                search_chunk, s, e, observations, effective_max, needs_domain, needs_full_rng
            )
            futures[future] = next_chunk
            next_chunk += 1

        while futures:
            completed_future = next(as_completed(futures))
            chunk_index = futures.pop(completed_future)
            chunk_results = completed_future.result()
            found.extend(chunk_results)
            progress.chunk_completed(chunk_index, len(chunk_results))
            if len(found) >= effective_max:
                # Got it. Cancel anything still queued so we don't keep grinding.
                for f in futures:
                    f.cancel()
                break

            if next_chunk < len(chunks):
                s, e = chunks[next_chunk]
                progress.chunk_started(next_chunk)
                future = executor.submit(
                    search_chunk, s, e, observations, effective_max, needs_domain, needs_full_rng
                )
                futures[future] = next_chunk
                next_chunk += 1
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    found = sorted(set(found))
    progress.final_summary()
    return found[: effective_max]


def parse_args() -> argparse.Namespace:
    cpu_count = os.cpu_count() or 1

    parser = argparse.ArgumentParser(
        description="Crack Grey Hack world seed using full-seed observables"
    )
    parser.add_argument("input", help="Path to JSON observations file")
    parser.add_argument(
        "--start",
        type=int,
        default=-(2**31),
        help="Start of world-seed range (inclusive)",
    )
    parser.add_argument(
        "--end", type=int, default=2**31 - 1, help="End of world-seed range (inclusive)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(cpu_count, 8)),
        help="Parallel workers for CPU phases",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2_000_000,
        help="Seeds per worker chunk (CPU-only mode)",
    )
    parser.add_argument(
        "--max-results", type=int, default=16, help="Stop after this many matches"
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Disable CUDA prefilter and use pure CPU search",
    )
    parser.add_argument(
        "--cuda-buffer-size",
        type=int,
        default=0,
        help="Override CUDA survivor buffer size (default: auto)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observations = load_observations(args.input)

    cfg = SearchConfig(
        start_seed=int(args.start),
        end_seed=int(args.end),
        workers=max(1, int(args.workers)),
        chunk_size=max(1, int(args.chunk_size)),
        max_results=max(1, int(args.max_results)),
        use_cuda=not args.no_cuda,
        cuda_buffer_size=max(0, int(args.cuda_buffer_size)),
    )

    print(f"Loaded {len(observations)} observations")
    print(f"CUDA available: {_CUDA_AVAILABLE}; CUDA enabled: {cfg.use_cuda and _CUDA_AVAILABLE}")
    print(
        f"Searching seeds in [{cfg.start_seed}, {cfg.end_seed}] "
        f"with workers={cfg.workers}, chunk_size={cfg.chunk_size}, max_results={cfg.max_results}"
    )

    matches = crack_world_seed(observations, cfg)
    if not matches:
        print("No matching world seed found in range")
        return

    print(f"Found {len(matches)} matching seed(s):")
    for seed in matches:
        print(seed)


if __name__ == "__main__":
    main()