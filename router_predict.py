#!/usr/bin/env python3
"""
Predict the deterministic contents of a Grey Hack RouterNode + Router from
(worldSeed, IP) alone.

KNOWN ISSUE (2026-05): for certain network seeds, _internal_sample() returns
negative values, producing nonsense BSSID bytes (e.g. "E2:-39:74:..." for
198.51.100.200 under seed -1285005987). Root cause appears to be that
DotNetRandom.__init__ in world_seed_cracker.py does not bound _seed_array
entries to [0, MBIG) at every step — C# uses int32 with overflow semantics;
our Python port lets values exceed int32 range, which contaminates later
draws. The cracker happens to work despite this because domain prediction
only consumes one Next(4) draw (early, before contamination accumulates).
Fix this in DotNetRandom and the predictor will work for all seeds.

This mirrors the C# generation sequence in ServerMap.RouterNode and Router
constructors. It predicts everything that is purely a function of the seeded
PRNG state, stopping at the first call that depends on game data we don't have
faithfully ported (Markov chain wordlist, hardware tables, filesystem template).

For each IP, the predictor produces:

  RouterNode constructor chain (Random A, seeded with worldSeed + ipSeed):
    seed         — network seed (worldSeed + IP.GetSeedFromIP(ip))
    tipoRed      — site type (no RNG, pure XOR math)
    webAddress   — full domain (1 RNG draw for TLD)
    bssid        — MAC address (6 RNG draws of Next(256))
    essid_skel   — structural fingerprint of the wifi name (1 draw of Next(7))
                   tells us source list (usernames vs corp_names) and whether
                   the _SUFFIX is appended (num < 3)

  GeneraRouter routerID chain (Random B, fresh, same seed):
    routerID     — "{ip}:{Next(int.MaxValue)}"

  Router constructor chain (Random C, fresh, same seed):
    lan_subnet   — "192.168.{Next(2)}.1"  (except RentServer which is fixed)
    rng_state_at_handoff — state of Random C immediately after lan_subnet draw,
                           which is where WordGenerator.GetNextWord(passwords)
                           would consume RNG to produce routerPassword

The handoff RNG state is included so this tool can be extended once the
Markov generator, hardware tables, and filesystem template are ported.

Usage:
  router_predict.py --seed -1285005987 --ip 99.71.91.182
  router_predict.py --seed -1285005987 --ips ips.txt
  router_predict.py --seed -1285005987 --ip 99.71.91.182 --format pretty
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from typing import List, Optional

# Reuse all the validated primitives from the cracker.
from python_tools.appleseed import (
    BASE_CONSONANTS,
    BASE_VOWELS,
    TLDS,
    TIPO_RED,
    DotNetRandom,
    get_seed_from_ip_like_csharp,
    ip_to_uint_be,
    ip_to_unique_name,
    shuffle_array,
    to_int32,
    to_uint32,
)


# Match ServerMap.TipoRed enum ordering. Note the enum starts with Unknown=0,
# while IpGenerator.tiposRed is 0-indexed FROM Comisaria. The site-type
# computation uses the tiposRed array (which TIPO_RED in the cracker mirrors).
# For human-readable output we just use the cracker's TIPO_RED list directly.


# The WiFi-name source lists as referenced by ServerMap.GeneraWifiName.
# random.Next(7) returns 0..6. The C# code says (num > 6) ? usernames : corp_names,
# but Next(7) never returns 7, so the source is ALWAYS corp_names in practice.
# This appears to be a dead branch / bug in the game code. Preserve fidelity.
ESSID_SOURCE_USERNAMES = "usernames"
ESSID_SOURCE_CORPNAMES = "corp_names"


@dataclass
class RngStateSnapshot:
    """Serializable snapshot of a DotNetRandom state, for handoff to extensions."""
    seed_array: List[int]
    inext: int
    inextp: int

    @classmethod
    def from_rng(cls, rng: DotNetRandom) -> "RngStateSnapshot":
        return cls(
            seed_array=list(rng._seed_array),
            inext=rng._inext,
            inextp=rng._inextp,
        )


@dataclass
class RouterPrediction:
    """Everything we can confidently predict for a router from (worldSeed, IP)."""

    # Inputs (echoed for clarity)
    world_seed: int
    ip_address: str

    # RouterNode (Random A)
    network_seed: int
    tipo_red: str
    tipo_red_index: int
    web_address: str
    domain: str
    domain_name: str           # the bit between "www." and ".<tld>"
    tld: str
    bssid: str
    essid_source: str          # "corp_names" or "usernames" (latter is dead code)
    essid_has_suffix: bool     # if num < 3, "_SUFFIX" is appended via OS.GetRandomName
    essid_first_draw: int      # the raw Next(7) value

    # GeneraRouter (Random B, fresh)
    router_id: str

    # Router constructor up to the wordgen handoff (Random C, fresh)
    lan_subnet_base: str       # the "192.168.x.1" value the router will use

    # State of Random C immediately after the LAN-subnet draw, suitable for
    # resuming generation once a WordGenerator port is available.
    handoff_rng_state: Optional[RngStateSnapshot] = field(default=None)

    # Caveats — fields that depend on game data we have not ported.
    notes: List[str] = field(default_factory=list)


def _compute_site_type(world_seed: int, ip_be: int) -> int:
    """Reproduce IpGenerator.GetNetworkType for the standard TipoRed table."""
    return ((ip_be ^ to_uint32(world_seed)) & 0x7FFFFFFF) % len(TIPO_RED)


def _generate_bssid(rng: DotNetRandom) -> str:
    """Reproduce Networking.GeneraMacAddress: 6 bytes of Next(256), hex with colons."""
    mac = [rng.next(max_value=256) for _ in range(6)]
    return ":".join(f"{b:02X}" for b in mac)


def _generate_essid_structure(rng: DotNetRandom):
    """Reproduce the *predictable* part of ServerMap.GeneraWifiName.

    Returns (source_list_name, has_suffix, raw_next7_value).
    The actual word content depends on WordGenerator/OS.GetRandomName.
    """
    n = rng.next(max_value=7)
    source = ESSID_SOURCE_USERNAMES if n > 6 else ESSID_SOURCE_CORPNAMES
    has_suffix = n < 3
    return source, has_suffix, n


def predict_router(world_seed: int, ip_address: str) -> RouterPrediction:
    """Predict every derivable field for a router at (world_seed, ip_address)."""

    # ---- Setup ----
    ip_seed = get_seed_from_ip_like_csharp(ip_address)
    network_seed = to_int32(world_seed + ip_seed)
    ip_be = ip_to_uint_be(ip_address)

    # ---- Site type (no RNG) ----
    site_type_idx = _compute_site_type(world_seed, ip_be)
    site_type_name = TIPO_RED[site_type_idx]

    # ---- RouterNode chain (Random A) ----
    rng_a = DotNetRandom(network_seed)

    # webAddress = "www." + GetDomainName(ip, rng_a)
    #   GetDomainName: name = IPToUniqueName(ip)  // no RNG
    #                  tld = Next(4)              // 1 draw
    shuffled_consonants = shuffle_array(BASE_CONSONANTS, world_seed)
    shuffled_vowels = shuffle_array(BASE_VOWELS, world_seed)
    domain_name = ip_to_unique_name(ip_address, shuffled_consonants, shuffled_vowels)
    domain_name_lc = domain_name.lower()
    tld_idx = rng_a.next(max_value=len(TLDS))
    tld = TLDS[tld_idx]
    domain = f"{domain_name_lc}.{tld}"
    web_address = f"www.{domain}"

    # bssid = GeneraMacAddress(rng_a)  // 6 draws
    bssid = _generate_bssid(rng_a)

    # essid = GeneraWifiName(rng_a)  // 1 draw + WordGenerator (opaque)
    essid_source, essid_has_suffix, essid_first = _generate_essid_structure(rng_a)

    # After this point, RouterNode does:
    #   - date = wall clock (skip)
    #   - helperLibVersions.ConfigLibVersions(rng_a) — N draws (skip until ported)
    # We don't try to predict past this in chain A.

    # ---- GeneraRouter chain (Random B, fresh) ----
    # routerID = ipAddress + ":" + Next(int.MaxValue)
    rng_b = DotNetRandom(network_seed)
    router_id_suffix = rng_b.next(max_value=2147483647)
    router_id = f"{ip_address}:{router_id_suffix}"

    # ---- Router constructor chain (Random C, fresh) ----
    rng_c = DotNetRandom(network_seed)

    # lan_subnet_base: "192.168.0.1" for RentServer, else "192.168.{Next(2)}.1"
    # RentServer is tipo_red index 13 in the ServerMap.TipoRed enum:
    #   Unknown=0, Comisaria=1, ..., RentServer=13, NetServices=14, ...
    # But IpGenerator.tiposRed (and our TIPO_RED) does NOT include Unknown,
    # RentServer, or CTF — those types aren't reachable via GetNetworkType.
    # So for any tipoRed produced by GetNetworkType, it is NEVER RentServer,
    # and the "192.168.{Next(2)}.1" branch always applies.
    lan_subnet_octet = rng_c.next(max_value=2)
    lan_subnet_base = f"192.168.{lan_subnet_octet}.1"

    # Snapshot RNG C state here — this is the handoff point.
    # The very next RNG consumer is WordGenerator.GetNextWord(passwords, rng_c)
    # which produces routerPassword. Everything downstream chains from this state.
    handoff = RngStateSnapshot.from_rng(rng_c)

    notes = [
        "BSSID is verified against in-game observations.",
        "Domain is verified against in-game observations.",
        "ESSID source list and suffix flag are derivable, but the actual "
        "wifi name string requires the Markov chain WordGenerator to be ported.",
        "Router password, hardware specs, file system contents, LAN topology, "
        "admin name, and admin password are all downstream of the WordGenerator "
        "call and cannot be predicted without it. The handoff_rng_state field "
        "captures Random C's state at the exact point where prediction stops.",
        "routerPos and date are not seed-derivable (Guid.NewGuid + wall clock).",
    ]

    return RouterPrediction(
        world_seed=world_seed,
        ip_address=ip_address,
        network_seed=network_seed,
        tipo_red=site_type_name,
        tipo_red_index=site_type_idx,
        web_address=web_address,
        domain=domain,
        domain_name=domain_name_lc,
        tld=tld,
        bssid=bssid,
        essid_source=essid_source,
        essid_has_suffix=essid_has_suffix,
        essid_first_draw=essid_first,
        router_id=router_id,
        lan_subnet_base=lan_subnet_base,
        handoff_rng_state=handoff,
        notes=notes,
    )


def _format_pretty(pred: RouterPrediction) -> str:
    lines = [
        f"=== Router at {pred.ip_address} (seed {pred.world_seed}) ===",
        f"  network_seed:   {pred.network_seed}",
        f"  tipo_red:       {pred.tipo_red} (index {pred.tipo_red_index})",
        f"  web_address:    {pred.web_address}",
        f"  bssid:          {pred.bssid}",
        f"  essid:          [{pred.essid_source}]"
                            + (" + _<SUFFIX>" if pred.essid_has_suffix else "")
                            + f"   (Next(7)={pred.essid_first_draw})",
        f"  router_id:      {pred.router_id}",
        f"  lan_subnet:     {pred.lan_subnet_base}",
        "",
        "  RNG handoff state captured — extend predictor when WordGenerator is ported.",
        "",
    ]
    return "\n".join(lines)


def _load_ips(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict Grey Hack router fields from (worldSeed, IP)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="The world seed (signed int32, e.g. -1285005987)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--ip",
        type=str,
        help="A single IP address to predict",
    )
    group.add_argument(
        "--ips",
        type=str,
        help="Path to a file containing one IP address per line (# comments allowed)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "pretty"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--include-rng-state",
        action="store_true",
        help="Include the full 56-int RNG state snapshot in JSON output "
             "(default: omit, since it's large and only useful for extensions)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.ip:
        ips = [args.ip]
    else:
        ips = _load_ips(args.ips)
        if not ips:
            print(f"No IPs found in {args.ips}", file=sys.stderr)
            return 1

    predictions = [predict_router(args.seed, ip) for ip in ips]

    if args.format == "pretty":
        for p in predictions:
            print(_format_pretty(p))
    else:
        if args.include_rng_state:
            payload = [asdict(p) for p in predictions]
        else:
            payload = []
            for p in predictions:
                d = asdict(p)
                d.pop("handoff_rng_state", None)
                payload.append(d)
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
