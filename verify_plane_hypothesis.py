#!/usr/bin/env python3
"""
verify_plane_hypothesis.py

Hypothesis
----------
For every new single-day trip the engine:
1. Picks *three* past single-day trips.
2. Treats the plane through those (miles, receipts, expected) points
   as the local model.
3. Reads the new trip’s (miles, receipts) on that plane to get
   expected_output.

This script checks how well that idea can explain the data.

Method
------
* Take the first half (⌊N/2⌋) of the 92 points as the **reference pool**.
* Use the remaining points as **tests**.
* For each test point, exhaustively try every 3-point combination
  from the pool (46 choose 3 = 15 180 planes).
* Plug the test trip’s (miles, receipts) into each plane, predict
  expected_output, and keep the plane with the smallest absolute error.
* Collect error statistics.

Run
---
$ python verify_plane_hypothesis.py public_cases.json

"""
import itertools
import json
import pathlib
import sys

import numpy as np
from tqdm import tqdm


def load_day1(path: pathlib.Path) -> np.ndarray:
    with path.open() as fh:
        raw = json.load(fh)
    pts = [
        (
            float(r["input"]["miles_traveled"]),
            float(r["input"]["total_receipts_amount"]),
            float(r["expected_output"]),
        )
        for r in raw
        if r["input"]["trip_duration_days"] == 1
    ]
    if not pts:
        raise ValueError("No day-1 rows found")
    return np.asarray(pts, dtype=float)


def plane_from_three(p):  # p = (3,3) array
    v1, v2 = p[1] - p[0], p[2] - p[0]
    n = np.cross(v1, v2)
    n /= np.linalg.norm(n)
    d = -np.dot(n, p[0])
    return n, d


def predict_z(n, d, x, y):
    if abs(n[2]) < 1e-12:
        return None  # vertical plane
    a, b, c = -n[0] / n[2], -n[1] / n[2], -d / n[2]
    return a * x + b * y + c


def main(path):
    pts = load_day1(path)
    N = len(pts)
    half = N // 2
    pool, tests = pts[:half], pts[half:]

    best_errs = []
    exact_hits = 0

    print(f"{N} day-1 points → pool {len(pool)}, tests {len(tests)}")

    combos = list(itertools.combinations(range(len(pool)), 3))
    planes = []
    for idxs in combos:
        p = pool[list(idxs)]
        n, d = plane_from_three(p)
        planes.append((idxs, n, d))

    test_infos = []  # collect detailed info for every test point

    for local_idx, (x, y, z) in enumerate(tqdm(tests, desc="testing"), start=0):
        best = float("inf")
        best_idxs = None
        best_pred = None

        for idxs, n, d in planes:
            z_hat = predict_z(n, d, x, y)
            if z_hat is None:
                continue
            err = abs(z_hat - z)
            if err < best:
                best = err
                best_idxs = idxs
                best_pred = z_hat
            if best < 1e-9:  # exact to the cent
                break

        best_errs.append(best)
        if best < 0.1:
            exact_hits += 1

        # Save reporting info
        global_idx = half + local_idx  # index relative to full dataset
        test_infos.append(
            {
                "global_idx": global_idx,
                "coords": (x, y, z),
                "pred": best_pred,
                "err": best,
                "support_idxs": best_idxs,
            }
        )

    best_errs = np.asarray(best_errs)
    print("\n=== results ===")
    print(f"tests solved to the cent (≤ 0.1): {exact_hits}/{len(tests)}")
    print(f"mean |error| : {best_errs.mean():.4f}")
    print(f"median error: {np.median(best_errs):.4f}")
    print(f"max   error : {best_errs.max():.4f}")


    print("\n=== per‑test best supporting triplets ===")
    for info in test_infos:
        gi = info["global_idx"]
        x, y, z = info["coords"]
        pred = info["pred"]
        err = info["err"]
        idxs = info["support_idxs"]

        print(f"Test[{gi}] (miles={x:.2f}, receipts={y:.2f}, expected={z:.2f})")
        print(f"  best plane prediction = {pred:.2f}  (|err| = {err:.4f})")
        print(f"  supporting pool indices: {idxs}")
        for i in idxs:
            sx, sy, sz = pool[i]
            print(f"    pool[{i}] = ({sx:.2f}, {sy:.2f}, {sz:.2f})")
        print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python verify_plane_hypothesis.py public_cases.json")
        sys.exit(1)
    main(pathlib.Path(sys.argv[1]))