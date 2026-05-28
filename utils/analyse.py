"""
Analyze paired_arena.py results.

Reads the CSV produced by paired_arena.py and reports, for one A-vs-B match:

  1. Decisive win rate (A) with a Wilson 95% interval — the headline.
  2. Paired margin: per-seed mean ship differential with a 95% interval.
     Because every map is played both slot-ways, this is the LOW-VARIANCE
     signal — it cancels each map's position asymmetry, so it moves on a
     real skill difference long before the win rate's interval tightens.
  3. A position/luck diagnostic: how often the winner flipped just by
     swapping slots on the same map.

By default it analyzes the most recent run_id in the file (so A and B are
unambiguous). Use --run-id or --all-runs to override.

Usage:
  python analyze.py
  python analyze.py --csv runs.csv
  python analyze.py --run-id 20260528-141233
  python analyze.py --all-runs
"""

import argparse
import math

import pandas as pd


def wilson(k, n, z=1.96):
    """Wilson score interval for k successes in n Bernoulli trials.

    Used instead of the textbook p +/- z*sqrt(p(1-p)/n) because that
    normal approximation is unreliable at the game counts you'll run.
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, center - half, center + half


def mean_ci(values, z=1.96):
    """Normal-approx 95% CI for a mean. Slightly optimistic for very small
    samples; fine once you have a few dozen seeds."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    m = sum(values) / n
    if n == 1:
        return m, m, m
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    return m, m - z * se, m + z * se


def analyze(df):
    # Keep only completed games (drop agent-crash rows).
    df = df[df["winner"].isin(["A", "B", "TIE"])].copy()
    if df.empty:
        print("No valid games to analyze.")
        return

    label_a = df["label_a"].iloc[0]
    label_b = df["label_b"].iloc[0]
    hash_a = df["hash_a"].iloc[0]
    hash_b = df["hash_b"].iloc[0]

    n_games = len(df)
    n_seeds = df["seed"].nunique()
    ties = int((df["winner"] == "TIE").sum())

    print(f"\n  {label_a} ({hash_a})  vs  {label_b} ({hash_b})")
    print(f"  {n_games} games over {n_seeds} maps   (ties: {ties})")
    print("  " + "-" * 58)

    # 1. Headline: decisive win rate with Wilson interval.
    dec = df[df["winner"].isin(["A", "B"])]
    k = int((dec["winner"] == "A").sum())
    n = len(dec)
    p, lo, hi = wilson(k, n)
    print(f"  WIN RATE (A, decisive games)   {p*100:5.1f}%   "
          f"95% CI [{lo*100:.1f}%, {hi*100:.1f}%]   (n={n})")
    if lo < 0.5 < hi:
        print("    -> interval straddles 50%: not yet distinguishable from a coin flip.")
    elif lo >= 0.5:
        print(f"    -> A is better with 95% confidence (lower bound {lo*100:.1f}% > 50%).")
    else:
        print(f"    -> B is better with 95% confidence (upper bound {hi*100:.1f}% < 50%).")

    # 2. Paired margin: average A-margin per map, then CI across maps.
    per_seed_margin = df.groupby("seed")["ship_margin"].mean().tolist()
    m, mlo, mhi = mean_ci(per_seed_margin)
    print(f"\n  PAIRED SHIP MARGIN (A - B)     {m:+6.1f}   "
          f"95% CI [{mlo:+.1f}, {mhi:+.1f}]   (maps={len(per_seed_margin)})")
    if mlo > 0:
        print("    -> positive with 95% confidence: A wins the economy on average.")
    elif mhi < 0:
        print("    -> negative with 95% confidence: B wins the economy on average.")
    else:
        print("    -> interval crosses 0: no clear margin edge yet.")

    # 3. Position/luck diagnostic over maps played both slot-ways.
    paired = df.groupby("seed")["winner"].agg(set)
    paired = paired[df.groupby("seed").size() >= 2]
    a_both = sum(w == {"A"} for w in paired)
    b_both = sum(w == {"B"} for w in paired)
    split = len(paired) - a_both - b_both
    if len(paired):
        print(f"\n  POSITION EFFECT (maps played both ways, n={len(paired)})")
        print(f"    A won both slots:  {a_both:3d}    B won both slots: {b_both:3d}    "
              f"slot decided it: {split:3d} ({split/len(paired)*100:.0f}%)")
        print("    -> high 'slot decided it' means map/position luck is large; "
              "trust the paired margin over the raw win rate.")
    print()


def main():
    p = argparse.ArgumentParser(description="Analyze paired_arena.py CSV output")
    p.add_argument("--csv", default="results.csv", help="CSV path (default results.csv)")
    p.add_argument("--run-id", default=None, help="Analyze a specific run_id")
    p.add_argument("--all-runs", action="store_true",
                   help="Pool every run in the file (assumes consistent A/B orientation)")
    args = p.parse_args()

    df = pd.read_csv(args.csv)

    if args.run_id:
        df = df[df["run_id"] == args.run_id]
        if df.empty:
            print(f"No rows with run_id == {args.run_id}")
            return
    elif not args.all_runs:
        latest = df["run_id"].iloc[-1]
        df = df[df["run_id"] == latest]
        print(f"(analyzing most recent run_id: {latest} — use --all-runs to pool everything)")

    analyze(df)


if __name__ == "__main__":
    main()