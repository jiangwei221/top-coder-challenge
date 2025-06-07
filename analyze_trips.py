#!/usr/bin/env python3
"""
analyze_trips.py

Usage
-----
$ python analyze_trips.py -f public_cases.json -d 5

Arguments
---------
-f, --file     Path to the JSON file that looks like:
               [
                 {"input": {...}, "expected_output": ...},
                 ...
               ]
-d, --days     The value of `trip_duration_days` you want to keep.

What it does
------------
1. Loads the JSON.
2. Extracts records whose `input["trip_duration_days"]` equals the
   days value you pass with -d / --days.
3. Builds a scatter-and-line plot of expected_output vs miles_traveled
   for that subset.
4. Saves the figure alongside the JSON (same folder) as
   expected_vs_miles_<days>d.png and also shows it on screen.
"""

import argparse
import json
import pathlib

import matplotlib.pyplot as plt


def load_data(path: pathlib.Path):
    """Return a list of dicts with duration, miles, and expected."""
    with path.open() as fh:
        raw = json.load(fh)

    records = []
    for row in raw:
        rec = {
            "trip_duration_days": row["input"]["trip_duration_days"],
            "miles_traveled":    row["input"]["miles_traveled"],
            "expected_output":   row["expected_output"],
        }
        records.append(rec)
    return records


def filter_by_duration(records, duration):
    """Keep only records matching the desired trip_duration_days."""
    return [r for r in records if r["trip_duration_days"] == duration]


def plot(records, duration, out_path):
    """Scatter + line plot of expected vs miles for the chosen duration."""
    # Sort by miles so the line is monotone left-to-right
    records = sorted(records, key=lambda r: r["miles_traveled"])
    miles   = [r["miles_traveled"]  for r in records]
    exp_out = [r["expected_output"] for r in records]

    plt.figure(figsize=(8, 5))
    plt.scatter(miles, exp_out, marker="o")
    plt.plot(miles, exp_out, linewidth=1)
    plt.title(
        f"Expected reimbursement vs. miles (trip_duration_days = {duration})",
        pad=15,
    )
    plt.xlabel("Miles traveled")
    plt.ylabel("Expected output")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze trip cases JSON")
    parser.add_argument(
        "-f",
        "--file",
        # required=True,
        type=pathlib.Path,
        default='public_cases.json',
        help="Path to public_cases.json (or similar)",
    )
    parser.add_argument(
        "-d",
        "--days",
        # required=True,
        type=int,
        default=1,
        help="Value of trip_duration_days to keep",
    )

    args = parser.parse_args()

    records = load_data(args.file)

    # ---------------------------------------------------------------------
    # Compute and display the maximum expected_output for every distinct
    # trip_duration_days present in the JSON.
    # ---------------------------------------------------------------------
    max_by_duration = {}
    for rec in records:
        d = rec["trip_duration_days"]
        max_by_duration[d] = max(
            max_by_duration.get(d, float("-inf")),
            rec["expected_output"],
        )

    print("\nMaximum expected_output by trip_duration_days:")
    for d in sorted(max_by_duration):
        print(f"  {d}: {max_by_duration[d]:.2f}")
    subset  = filter_by_duration(records, args.days)

    if not subset:
        raise SystemExit(
            f"No rows with trip_duration_days == {args.days}! "
            "Did you choose an existing value?"
        )

    out_png = args.file.with_name(f"expected_vs_miles_{args.days}d.png")
    plot(subset, args.days, out_png)
    print(f"Plot saved to {out_png.resolve()}")


if __name__ == "__main__":
    main()