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


if __name__ == "__main__":
    main()
