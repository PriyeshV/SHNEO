"""Summarize a SLURM array of AntMaze runs.

Success-rate AUC on AntMaze is bimodal: a run either reaches the good mode or
stalls in a poor one, with little in between.  A mean over a handful of runs
therefore reports which modes happened to be sampled rather than how good an
arm is, and looks deceptively precise when every sample lands on one side.

So this reports the fraction of runs in the upper mode with a Wilson interval,
alongside the usual mean, and picks the mode boundary from the largest gap in
the pooled values unless one is given.

Usage, from the directory holding the slurm-*.out files::

    python src/mains/ant_maze/summarize_runs.py            # newest array
    python src/mains/ant_maze/summarize_runs.py --job 10522714
    python src/mains/ant_maze/summarize_runs.py --threshold 0.19
"""

import argparse
import glob
import math
import os
import re
from collections import defaultdict

AUC_RE = re.compile(r"auc_normalized=([0-9.eE+-]+)")
TASK_RE = re.compile(r"^task=(\d+)\s+tag=(\S+)\s+seed=(\d+)", re.M)
SUCCESS_RE = re.compile(r"success=([0-9.]+)")
CURVE_RE = re.compile(r"env_step=(\d+) \| return=[0-9.eE+-]+ \| success=([0-9.]+)")


def newest_job(directory):
    candidates = glob.glob(os.path.join(directory, "slurm-*_0.out"))
    if not candidates:
        raise SystemExit(f"no slurm-*_0.out in {directory!r}; pass --job")
    newest = max(candidates, key=os.path.getmtime)
    return re.search(r"slurm-(\d+)_0\.out", os.path.basename(newest)).group(1)


def wilson(successes, total, z=1.96):
    """Wilson score interval: sane at 0/n and n/n, unlike the normal one."""

    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def largest_gap(values):
    """Midpoint of the widest gap between consecutive sorted values."""

    if len(values) < 2:
        return None
    ordered = sorted(values)
    gaps = [(b - a, (a + b) / 2.0) for a, b in zip(ordered, ordered[1:])]
    width, midpoint = max(gaps)
    return midpoint, width


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=".", help="directory with slurm-*.out")
    parser.add_argument("--job", default=None, help="array id (default: newest)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="upper-mode boundary (default: widest gap)")
    parser.add_argument("--curves", action="store_true",
                        help="compare the mean evaluation curve of each mode")
    args = parser.parse_args()

    job = args.job or newest_job(args.dir)
    runs = []
    for path in sorted(glob.glob(os.path.join(args.dir, f"slurm-{job}_*.out"))):
        text = open(path, errors="replace").read()
        header = TASK_RE.search(text)
        auc = AUC_RE.findall(text)
        success = SUCCESS_RE.findall(text)
        if header is None:
            continue
        task, tag, seed = int(header.group(1)), header.group(2), int(header.group(3))
        runs.append({
            "task": task, "tag": tag, "seed": seed,
            "auc": float(auc[-1]) if auc else None,
            "final": float(success[-1]) if success else None,
            "curve": [(int(a), float(b)) for a, b in CURVE_RE.findall(text)],
        })

    done = [r for r in runs if r["auc"] is not None]
    print(f"array {job}: {len(done)}/{len(runs)} runs finished\n")
    if not done:
        raise SystemExit("no completed runs yet")

    values = [r["auc"] for r in done]
    if args.threshold is not None:
        threshold, width = args.threshold, None
    else:
        threshold, width = largest_gap(values)
        print(f"mode boundary {threshold:.4f} (widest gap {width:.4f}); "
              f"override with --threshold\n")

    print(f"{'TASK':>4} {'TAG':<10} {'SEED':>4} {'AUC':>9} {'FINAL':>7}  MODE")
    for r in sorted(done, key=lambda r: r["auc"]):
        mode = "high" if r["auc"] >= threshold else "low"
        print(f"{r['task']:>4} {r['tag']:<10} {r['seed']:>4} "
              f"{r['auc']:>9.4f} {r['final']:>7.3f}  {mode}")

    by_tag = defaultdict(list)
    for r in done:
        by_tag[r["tag"]].append(r["auc"])

    print(f"\n{'TAG':<10} {'N':>3} {'MEAN':>8} {'MIN':>8} {'MAX':>8} "
          f"{'HIGH':>7}  {'FRACTION':>8}  95% CI")
    for tag in sorted(by_tag, key=lambda t: -sum(
            v >= threshold for v in by_tag[t]) / len(by_tag[t])):
        vals = by_tag[tag]
        high = sum(v >= threshold for v in vals)
        lo, hi = wilson(high, len(vals))
        print(f"{tag:<10} {len(vals):>3} {sum(vals)/len(vals):>8.4f} "
              f"{min(vals):>8.4f} {max(vals):>8.4f} {high:>3}/{len(vals):<3} "
              f"{high/len(vals):>8.2f}  [{lo:.2f}, {hi:.2f}]")

    print("\nOverlapping intervals mean the arms are not separated at this "
          "sample size.")

    if args.curves:
        _print_curves(done, threshold)


def _print_curves(runs, threshold):
    """Mean evaluation curve for each mode, to locate where runs diverge.

    If the low mode is already behind at the first evaluations, the split is
    decided early and the cause is in the initial learning dynamics rather
    than in anything an exploration bonus does later.
    """

    groups = {"high": [], "low": []}
    for run in runs:
        if run["curve"]:
            groups["high" if run["auc"] >= threshold else "low"].append(run["curve"])

    if not groups["high"] or not groups["low"]:
        print("\nboth modes need at least one run for the curve comparison")
        return

    length = min(len(c) for group in groups.values() for c in group)
    picks = [round(i * (length - 1) / 7.0) for i in range(8)]

    print(f"\nmean success by evaluation ({len(groups['high'])} high, "
          f"{len(groups['low'])} low)")
    header = "".join(f"{groups['high'][0][i][0] / 1e6:>9.1f}M" for i in picks)
    print(f"{'step':<7}{header}")
    for name in ("high", "low"):
        cells = ""
        for i in picks:
            values = [curve[i][1] for curve in groups[name]]
            cells += f"{sum(values) / len(values):>10.3f}"
        print(f"{name:<7}{cells}")

    first = picks[1]
    high_early = sum(c[first][1] for c in groups["high"]) / len(groups["high"])
    low_early = sum(c[first][1] for c in groups["low"]) / len(groups["low"])
    step = groups["high"][0][first][0]
    if low_early < 0.5 * high_early:
        print(f"\nThe modes have already separated by {step/1e6:.1f}M steps: "
              "the split is decided early.")
    else:
        print(f"\nThe modes are still together at {step/1e6:.1f}M steps: "
              "the split happens later in training.")


if __name__ == "__main__":
    main()
