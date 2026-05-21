#!/usr/bin/env python3
"""
Predict RouterB's position given RouterA's exact (x, y) and the Power
percentage that RouterA reports for RouterB.

Game formula (from RouterNode source):
    Power = 100 - int(distance * 100f / 0.5f)
          = 100 - int(distance * 200.0f)       -- all float32

Inversion (k = 100 - Power):
    k / 200.0  <=  distance  <  (k+1) / 200.0

Each router's seed is new System.Random(Guid.NewGuid().GetHashCode()), which
is an int32 (~4.3 billion candidates). We brute-force the full int32 space on
the GPU using the _dn_init/_dn_sample device functions from appleseed.py.

Cell layout: cells are 40x40 game units; positions come from data/MapConfig.xml.
The RNG sequence per seed is:
    sample 0  ->  cell index         (int(s * cell_count))
    sample 1  ->  x offset in [-20, 20]
    sample 2  ->  y offset in [-20, 20]

Usage:
    python predict_router.py <router_a_x> <router_a_y> <power>

Example:
    python predict_router.py 150.3 -320.5 73
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from numba import cuda
from numba import float32 as nb_f32
from numba import float64 as nb_f64
from numba import int32 as nb_i32

sys.path.insert(0, str(Path(__file__).resolve().parent))
from appleseed import _dn_init, _dn_sample

MAPCONFIG = Path(r".\serialCeldas.json")

# 1.0 / 2_147_483_647  — converts a DotNetRandom internal sample to [0, 1)
_MBIG_INV = 4.656612875245797e-10


def load_cell_anchors() -> np.ndarray:
    with open(MAPCONFIG, "r") as f:
        data = json.load(f)
    cells = [
        (float(c["anchoredPosition"]["x"]), float(c["anchoredPosition"]["y"]))
        for c in data
    ]
    return np.array(cells, dtype=np.float32)  # shape (N, 2)


@cuda.jit
def _router_kernel(
    cell_anchors,  # float32[N, 2]: cell anchor positions
    cell_count,  # int32
    ax,  # float32: RouterA x
    ay,  # float32: RouterA y
    target_k,  # int32: 100 - Power
    start_signed,  # int64: seed at tid == 0
    seed_limit,  # int64: number of seeds this launch covers
    out_seeds,  # int32[B]
    out_bx,  # float32[B]
    out_by,  # float32[B]
    counter,  # int32[1]: atomic write index
):
    tid = cuda.grid(1)
    if tid >= seed_limit:
        return

    seed = nb_i32(start_signed + tid)

    state = cuda.local.array(56, dtype=nb_i32)
    _dn_init(state, seed)
    inext = nb_i32(0)
    inextp = nb_i32(21)

    # sample 0 -> cell index: int(sample() * cell_count)
    s0, inext, inextp = _dn_sample(state, inext, inextp)
    cell_idx = nb_i32(nb_f64(s0) * _MBIG_INV * nb_f64(cell_count))
    if cell_idx < 0:
        cell_idx = nb_i32(0)
    elif cell_idx >= cell_count:
        cell_idx = cell_count - nb_i32(1)

    # sample 1 -> x offset: (float)(nextDouble * 40.0 - 20.0)
    s1, inext, inextp = _dn_sample(state, inext, inextp)
    x_off = nb_f32(nb_f64(s1) * _MBIG_INV * 40.0 - 20.0)

    # sample 2 -> y offset
    s2, inext, inextp = _dn_sample(state, inext, inextp)
    y_off = nb_f32(nb_f64(s2) * _MBIG_INV * 40.0 - 20.0)

    bx = cell_anchors[cell_idx, 0] + x_off
    by = cell_anchors[cell_idx, 1] + y_off

    # distance in float32 — mirrors Unity Mathf.Sqrt / Vector2.Distance
    dx = bx - ax
    dy = by - ay
    dist = nb_f32(math.sqrt(dx * dx + dy * dy))

    # mirror game formula: (int)(dist * 200.0f)
    if nb_i32(dist * nb_f32(200.0)) != target_k:
        return

    idx = cuda.atomic.add(counter, 0, 1)
    if idx < out_seeds.shape[0]:
        out_seeds[idx] = seed
        out_bx[idx] = bx
        out_by[idx] = by


def search(
    ax: float, ay: float, power: int, buf: int = 50_000
) -> list[tuple[int, float, float]]:
    anchors = load_cell_anchors()
    n_cells = len(anchors)
    k = 100 - power

    # print(f"Cells: {n_cells}", flush=True)
    print(f"RouterA: ({ax}, {ay})", flush=True)
    print(
        f"Power: {power}%  ->  k={k}  ->  dist in [{(k)/200:.5f}, {(k+1)/200:.5f})",
        flush=True,
    )

    d_anchors = cuda.to_device(anchors)
    d_seeds = cuda.to_device(np.zeros(buf, dtype=np.int32))
    d_bx = cuda.to_device(np.zeros(buf, dtype=np.float32))
    d_by = cuda.to_device(np.zeros(buf, dtype=np.float32))
    d_counter = cuda.to_device(np.zeros(1, dtype=np.int32))

    THREADS = 256
    ax32 = np.float32(ax)
    ay32 = np.float32(ay)

    t0 = time.time()
    # Split at zero: each half fits in int64 thread indexing
    for r_start, r_end in [(-(2**31), -1), (0, 2**31 - 1)]:
        width = r_end - r_start + 1
        blocks = (width + THREADS - 1) // THREADS
        _router_kernel[blocks, THREADS](
            d_anchors,
            np.int32(n_cells),
            ax32,
            ay32,
            np.int32(k),
            np.int64(r_start),
            np.int64(width),
            d_seeds,
            d_bx,
            d_by,
            d_counter,
        )

    elapsed = time.time() - t0
    total = int(d_counter.copy_to_host()[0])
    stored = min(total, buf)

    seeds = d_seeds.copy_to_host()[:stored]
    bxs = d_bx.copy_to_host()[:stored]
    bys = d_by.copy_to_host()[:stored]

    if total > buf:
        print(
            f"WARNING: {total} matches exceeded buffer ({buf}); rerun with buf=<larger>",
            flush=True,
        )

    print(f"Search complete in {elapsed:.2f}s — {total} candidate(s)", flush=True)
    return list(zip(seeds.tolist(), bxs.tolist(), bys.tolist()))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("router_a_x", type=float, help="RouterA X coordinate")
    ap.add_argument("router_a_y", type=float, help="RouterA Y coordinate")
    ap.add_argument(
        "power", type=int, help="Power %% that RouterA reports for RouterB (0-100)"
    )
    args = ap.parse_args()

    if not 0 <= args.power <= 100:
        sys.exit("error: power must be in [0, 100]")

    results = search(args.router_a_x, args.router_a_y, args.power)

    if not results:
        print("No candidates found.")
        return

    print()
    print(f"{'Seed':>12}  {'RouterB X':>12}  {'RouterB Y':>12}  {'Distance':>10}")
    print("-" * 54)
    for seed, bx, by in sorted(results, key=lambda t: t[0]):
        dx = bx - args.router_a_x
        dy = by - args.router_a_y
        dist = math.sqrt(dx * dx + dy * dy)
        print(f"{seed:>12}  {bx:>12.4f}  {by:>12.4f}  {dist:>10.6f}")


    average_x = sum(bx for _, bx, _ in results) / len(results)
    average_y = sum(by for _, _, by in results) / len(results)
    print(f"\nAverage position: ({average_x:.4f}, {average_y:.4f})")


if __name__ == "__main__":
    main()
