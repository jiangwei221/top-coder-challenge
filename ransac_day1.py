#!/usr/bin/env python3
"""
ransac_day1.py

A quick‑and‑dirty RANSAC‑style plane‑finder for the Top‑Coder trip dataset.

What it does
------------
1. Reads *public_cases.json*.
2. Keeps only rows where ``input.trip_duration_days == 1``.
3. Builds a 3‑D point cloud of
      x = miles_traveled
      y = total_receipts_amount
      z = expected_output
4. Runs a user‑specified number (-n / --n_iter) of random 3‑point samples,
   turning each sample into a candidate plane.
5. Counts how many other points fall on that plane within a perpendicular
   distance threshold (-e / --eps).
6. Deduplicates “nearly identical” planes and prints a summary.

Example
-------
$ python ransac_day1.py -f public_cases.json -n 5000 -e 2.0

"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
from collections import defaultdict
from typing import Dict, List, Tuple


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3‑D proj)
import numpy as np  # (already present but safe if duplicated)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def load_day1_points(path: pathlib.Path) -> np.ndarray:
    """Return an (N, 3) array of (miles, receipts, expected)."""
    with path.open() as fh:
        raw = json.load(fh)

    pts: List[Tuple[float, float, float]] = []
    for row in raw:
        if row["input"]["trip_duration_days"] != 1:
            continue
        pts.append(
            (
                float(row["input"]["miles_traveled"]),
                float(row["input"]["total_receipts_amount"]),
                float(row["expected_output"]),
            )
        )
    if not pts:
        raise ValueError("No rows found with trip_duration_days == 1")
    return np.asarray(pts, dtype=float)


def plane_from_three(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Given three non‑collinear points, return (normal, d) s.t.
        normal • p + d = 0
    and ‖normal‖ == 1.
    Raises ValueError if the points are collinear.
    """
    v1, v2 = p2 - p1, p3 - p1
    normal = np.cross(v1, v2)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-9:
        raise ValueError("Points are collinear")
    normal = normal / norm_len

    # Fix sign to make comparison easier: first non‑zero component positive
    for comp in normal:
        if abs(comp) > 1e-12:
            if comp < 0:
                normal *= -1
            break

    d = -np.dot(normal, p1)
    return normal, float(d)


def point_plane_distance(normal: np.ndarray, d: float, pts: np.ndarray) -> np.ndarray:
    """Return perpendicular distances of pts (N,3) to plane normal•p + d = 0."""
    return np.abs(np.dot(pts, normal) + d)


def near_same_plane(
    n1: np.ndarray,
    d1: float,
    n2: np.ndarray,
    d2: float,
    ang_tol: float = 1e-3,
    d_tol: float = 1e-1,
) -> bool:
    """Rudimentary check if two planes are effectively the same."""
    # Compare normals (angle between them)
    cosang = np.clip(np.dot(n1, n2), -1.0, 1.0)
    if 1.0 - cosang > ang_tol:
        return False
    return abs(d1 - d2) < d_tol


# ----------------------------------------------------------------------
# Main RANSAC loop
# ----------------------------------------------------------------------
def find_planes(
    pts: np.ndarray,
    n_iter: int = 1_000,
    eps: float = 1.0,
    min_inliers: int = 4,
    rng: random.Random | None = None,
) -> Dict[int, Tuple[np.ndarray, float, List[int]]]:
    """
    Run a crude RANSAC plane search.

    Returns
    -------
    planes : dict[int, (normal, d, inlier_indices)]
        Keyed by plane id. Each entry has the unit normal,
        the offset d, and the list of inlier row indices.
    """
    if rng is None:
        rng = random.Random()

    n_pts = pts.shape[0]
    planes: Dict[int, Tuple[np.ndarray, float, List[int]]] = {}
    plane_id = 0

    for _ in range(n_iter):
        idx_sample = rng.sample(range(n_pts), 3)
        p1, p2, p3 = pts[idx_sample]

        try:
            normal, d = plane_from_three(p1, p2, p3)
        except ValueError:
            continue  # collinear, skip

        dist = point_plane_distance(normal, d, pts)
        inliers = np.where(dist <= eps)[0]

        if len(inliers) < min_inliers:
            continue

        # Check if this plane (normal, d) is already known
        duplicate = False
        for n_exist, d_exist, _ in planes.values():
            if near_same_plane(normal, d, n_exist, d_exist):
                duplicate = True
                break
        if duplicate:
            continue

        planes[plane_id] = (normal, d, inliers.tolist())
        plane_id += 1

    return planes


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Find distinct planes among day‑1 trip points "
        "that fit other points within a distance threshold."
    )
    p.add_argument("-f", "--file", required=True, type=pathlib.Path, help="Path to public_cases.json")
    p.add_argument("-n", "--n_iter", type=int, default=2000, help="Number of random samples")
    p.add_argument("-e", "--eps", type=float, default=1.0, help="Perpendicular distance threshold")
    p.add_argument(
        "--min_inliers",
        type=int,
        default=4,
        help="Minimum #points (incl. the 3 sample points) required to accept a plane",
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    print(rng)

    pts = load_day1_points(args.file)
    print(f"Loaded {pts.shape[0]} day‑1 points.")

    planes = find_planes(
        pts, n_iter=args.n_iter, eps=args.eps, min_inliers=args.min_inliers, rng=rng
    )
    
    # import IPython; IPython.embed()

    print(f"\nFound {len(planes)} distinct planes (eps = {args.eps}).\n")

    coeffs = []
    for pid, (n, d, inliers) in planes.items():
        print(f"Plane {pid:02d}:")
        print(f"  normal = {n}")
        print(f"  d      = {d:.6f}")
        print(f"  # inliers (including the 3 seed pts) = {len(inliers)}")
        print(f"  indices = {inliers}")
        print("  points (miles_traveled, total_receipts_amount, expected_output):")
        # found_outlier = False
        for idx in inliers:
            x, y, z = pts[idx]
            print(f"    ({x:.2f}, {y:.2f}, {z:.2f})")
            # if z == 446.94:
            #     found_outlier = True
        
        if abs(n[2]) > 1e-9:  # n_z not ~0, so we can solve for expected_output
            a = -n[0] / n[2]
            b = -n[1] / n[2]
            c = -d / n[2]
            print(
                "  equation (expected_output ≈ a*miles + b*receipts + c):\n"
                f"    a = {a:.6f},  b = {b:.6f},  c = {c:.6f}"
            )
            coeffs.append((a, b, c))
        else:
            print("  Plane is nearly vertical in z; cannot express z as a function of x,y.")
        # if found_outlier:
        #     import IPython; IPython.embed()

    # ------------------------------------------------------------------
    # How many 4‑point planes does each day‑1 record belong to?
    # ------------------------------------------------------------------
    point_plane_count = [0] * len(pts)
    point_plane_ids = [[] for _ in range(len(pts))]
    for pid_loop, (_n, _d, inliers) in planes.items():
        for idx in inliers:
            point_plane_count[idx] += 1
            point_plane_ids[idx].append(pid_loop)

    # print("\n=== Coplane membership per point (sorted by index) ===")
    # for idx, (x, y, z) in enumerate(pts):
    #     print(f"Idx {idx:02d}: ({x:.2f}, {y:.2f}, {z:.2f})  planes = {point_plane_count[idx]}")
        
    # ------------------------------------------------------------------
    # Points that belong to exactly ONE 4‑point plane AND whose three
    # supporting points each lie on MORE THAN ONE 4‑point plane.
    # ------------------------------------------------------------------
    print(
        "\n=== Points with exactly ONE coplane where the other three "
        "inliers each belong to >1 planes ==="
    )
    for idx, (x, y, z) in enumerate(pts):
        if point_plane_count[idx] != 1:
            continue

        pid = point_plane_ids[idx][0]
        inlier_idxs = planes[pid][2]  # list of 4 indices
        supporting = [i for i in inlier_idxs if i != idx]

        if all(point_plane_count[si] > 1 for si in supporting):
            print(
                f"Idx {idx:02d}: ({x:.2f}, {y:.2f}, {z:.2f})  -> plane {pid:02d}"
            )
            print("  Supporting triplet (each on >1 planes):")
            for si in supporting:
                sx, sy, sz = pts[si]
                print(
                    f"    pool[{si}] = ({sx:.2f}, {sy:.2f}, {sz:.2f}) "
                    f"[planes = {point_plane_count[si]}]"
                )
            print()

# ------------------------------------------------------------------
    # Points that belong to exactly ONE 4‑point plane
    # ------------------------------------------------------------------
    print("\n=== Points with exactly ONE supporting 4‑point plane ===")
    for idx, (x, y, z) in enumerate(pts):
        if point_plane_count[idx] == 1:
            pid = point_plane_ids[idx][0]
            inlier_idxs = planes[pid][2]  # list of 4 indices
            # separate the three supporting points from the point itself
            supporting = [i for i in inlier_idxs if i != idx]

            print(
                f"Idx {idx:02d}: ({x:.2f}, {y:.2f}, {z:.2f})  -> plane {pid:02d}"
            )
            print("  Supporting triplet:")
            for si in supporting:
                sx, sy, sz = pts[si]
                print(f"    pool[{si}] = ({sx:.2f}, {sy:.2f}, {sz:.2f})")
            print("  Test point itself:")
            print(f"    self[{idx}] = ({x:.2f}, {y:.2f}, {z:.2f})\n")
    # 3‑D scatter of (a, b, c) for all detected planes
    # ------------------------------------------------------------------
    if coeffs:
        coeffs_arr = np.asarray(coeffs, dtype=float)
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(coeffs_arr[:, 0], coeffs_arr[:, 1], coeffs_arr[:, 2], s=60)
        ax.set_xlabel("a (miles coefficient)")
        ax.set_ylabel("b (receipts coefficient)")
        ax.set_zlabel("c (intercept)")
        ax.set_title("Planes in coefficient space")
        plt.tight_layout()
        plt.show()

        # ------------------------------------------------------------------
        # Visualise every accepted plane as a semi‑transparent surface in
        # (miles_traveled, total_receipts_amount, expected_output) space,
        # along with all the original day‑1 points.
        # ------------------------------------------------------------------
        if planes:
            fig2 = plt.figure(figsize=(7, 6))
            ax2 = fig2.add_subplot(111, projection="3d")

            # Scatter the raw points for context
            ax2.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                s=15,
                marker="o",
                alpha=0.6,
                label="data points",
            )

            x_min, x_max = 0, pts[:, 0].max()
            y_min, y_max = 0, pts[:, 1].max()
            z_min, z_max = 0, pts[:, 2].max()

            # Build a common x‑y grid spanning the data range
            x_mesh = np.linspace(pts[:, 0].min(), pts[:, 0].max(), 10)
            y_mesh = np.linspace(pts[:, 1].min(), pts[:, 1].max(), 10)
            xx, yy = np.meshgrid(x_mesh, y_mesh)

            for pid, (n, d, _) in planes.items():
                if abs(n[2]) < 1e-9:
                    continue  # skip nearly vertical planes
                a = -n[0] / n[2]
                b = -n[1] / n[2]
                c = -d / n[2]
                zz = a * xx + b * yy + c
                ax2.plot_surface(xx, yy, zz, alpha=0.3, linewidth=0)

            ax2.set_xlabel("Miles traveled")
            ax2.set_ylabel("Total receipts amount")
            ax2.set_zlabel("Expected output")
            # Cap axes to the data extents
            ax2.set_xlim(x_min, x_max)
            ax2.set_ylim(y_min, y_max)
            ax2.set_zlim(z_min, z_max)
            ax2.set_title("Detected planes in original data space")
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    main()