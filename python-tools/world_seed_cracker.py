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
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

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
            num4 = num3 - num4
            if num4 < 0:
                num4 += MBIG
            num3 = self._seed_array[num]

        for _ in range(1, 5):
            for k in range(1, 56):
                num5 = k + 30
                if num5 >= 55:
                    num5 -= 55
                self._seed_array[k] -= self._seed_array[1 + num5]
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


def search_chunk(
    start_seed: int,
    end_seed: int,
    observations: Sequence[Observation],
    max_results: int,
    needs_domain: bool,
) -> List[int]:
    results: List[int] = []
    for seed in range(start_seed, end_seed + 1):
        if candidate_matches(seed, observations, needs_domain):
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


def crack_world_seed(
    observations: Sequence[Observation], cfg: SearchConfig
) -> List[int]:
    if cfg.start_seed > cfg.end_seed:
        raise ValueError("start_seed cannot be greater than end_seed")

    needs_domain = any(o.domain is not None for o in observations)
    chunks = build_chunks(cfg.start_seed, cfg.end_seed, cfg.chunk_size)

    if cfg.workers <= 1:
        found: List[int] = []
        for s, e in chunks:
            found.extend(
                search_chunk(
                    s, e, observations, cfg.max_results - len(found), needs_domain
                )
            )
            if len(found) >= cfg.max_results:
                break
        return found

    found = []
    with ProcessPoolExecutor(max_workers=cfg.workers) as executor:
        futures = {
            executor.submit(
                search_chunk, s, e, observations, cfg.max_results, needs_domain
            ): (s, e)
            for s, e in chunks
        }

        for future in as_completed(futures):
            res = future.result()
            if res:
                found.extend(res)
                if len(found) >= cfg.max_results:
                    break

    found = sorted(set(found))
    return found[: cfg.max_results]


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
        help="Parallel workers",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2_000_000,
        help="Seeds per worker chunk",
    )
    parser.add_argument(
        "--max-results", type=int, default=16, help="Stop after this many matches"
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
    )

    print(f"Loaded {len(observations)} observations")
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
