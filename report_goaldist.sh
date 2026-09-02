#!/bin/bash
# Success against goal distance, from a run_antmaze.sh array.
#
#   bash report_goaldist.sh            # newest array
#   bash report_goaldist.sh 10623430   # a specific one
#
# Peak rather than final success: every configuration in this project declines
# after roughly 4M steps, so the final evaluation understates what the agent
# reached.

set -uo pipefail
cd "$(dirname "$0")"

JOB="${1:-}"
if [ -z "$JOB" ]; then
  JOB=$(ls -t slurm-*_0.out 2>/dev/null | head -1 | sed 's/slurm-\([0-9]*\)_0\.out/\1/')
fi
[ -n "$JOB" ] || { echo "no slurm-*_0.out found; pass a job id" >&2; exit 1; }
echo "array $JOB"
echo
printf "%-11s %s\n" "goal dist" "  1       2       3       4       5       6"

for arm in SAC NEO_xy NEO_pixel; do
  row=""
  for d in 1 2 3 4 5 6; do
    best=""
    for f in slurm-${JOB}_*.out; do
      [ -f "$f" ] || continue
      grep -q "tag=${arm}_d${d} " "$f" || continue
      peak=$(grep -o 'success=[0-9.]*' "$f" | cut -d= -f2 | sort -g | tail -1)
      [ -n "$peak" ] || continue
      # mean over the seeds present for this cell
      best="${best}${peak} "
    done
    if [ -z "$best" ]; then
      row="${row}    -   "
    else
      mean=$(echo "$best" | tr ' ' '\n' | grep . | awk '{s+=$1; n++} END {printf "%.3f", s/n}')
      row="${row}  ${mean} "
    fi
  done
  printf "%-11s %s\n" "$arm" "$row"
done

echo
echo "peak evaluation success, averaged over seeds"
echo "a column that is high for every method is not testing exploration;"
echo "a column that is zero for every method is where the method should show up"
