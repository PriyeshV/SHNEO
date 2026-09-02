#!/bin/bash
# Success against goal distance on ant_u_maze (Mila).
#
#   sbatch run_antmaze.sh
#   bash report_goaldist.sh <jobid>
#
# The multi-goal task cannot ask the exploration question.  u_maze has six
# goals at path distances 1 to 6 cells from the reset, and the reported
# success rate is their average, so it is dominated by the nearest ones:
# a fixed near goal is solved 4/4 seeds to 100%, the far goal 0/4 to 0%, and
# the "bimodal" multi-goal AUC turns out to be whether a run masters one goal
# (1/6 = 0.167) or two (2/6 = 0.333).
#
# So sweep the goal instead of averaging over it.  Three methods x six goal
# distances x two seeds = 36 runs, giving a success-versus-distance profile
# per method.  With the goal fixed, both start and goal are deterministic and
# the run-to-run variance that plagued the multi-goal grids largely vanishes,
# so two seeds carry more information here than twelve did there.

#SBATCH --job-name=shneo_goaldist
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --array=0-35
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail

VENV="${SHNEO_VENV:-$HOME/virtualenvs/pybrax_uv}"
REPO="${SHNEO_REPO:-$HOME/SHNEO}"
source "$VENV/bin/activate"
cd "$REPO"

ARMS=(SAC NEO_xy NEO_pixel)
# u_maze goal cells (i, j) at world (4i, 4j), ordered by path distance from
# the reset at cell (1, 1).  All six are 'g' in the layout.
GOAL_X=(4  4  8  12 12 12)
GOAL_Y=(8  12 12 12 8  4)
DIST=(1 2 3 4 5 6)

ARM_INDEX=$(( SLURM_ARRAY_TASK_ID / 12 ))
WITHIN=$((   SLURM_ARRAY_TASK_ID % 12 ))
GOAL_INDEX=$(( WITHIN / 2 ))
SEED=$((      WITHIN % 2 ))

TAG="${ARMS[$ARM_INDEX]}"
GX="${GOAL_X[$GOAL_INDEX]}"
GY="${GOAL_Y[$GOAL_INDEX]}"
D="${DIST[$GOAL_INDEX]}"
NAME="${TAG}_d${D}"

COMMON=(
  --e_id ant_u_maze
  --e_fixed_goal "$GX" "$GY"
  --wb_on True
  --e_n_test_envs 64
  --a_n_evals 40
  --x_seed "$SEED"
  --x_file "$NAME"
  --wb_model "$NAME"
)

case "$TAG" in
  SAC)       SCRIPT=src/mains/ant_maze/RND.py
             EXTRA=(--e_pixel_obs False --r_input xy --a_int_coef 0) ;;
  NEO_xy)    SCRIPT=src/mains/ant_maze/NEO_LapXY.py
             EXTRA=(--e_pixel_obs False --r_input xy) ;;
  NEO_pixel) SCRIPT=src/mains/ant_maze/NEO.py
             EXTRA=(--e_pixel_obs True) ;;
  *) echo "unexpected tag $TAG" >&2; exit 1 ;;
esac

echo "task=$SLURM_ARRAY_TASK_ID tag=$NAME seed=$SEED goal=($GX,$GY) dist=$D"
exec python "$SCRIPT" "${COMMON[@]}" "${EXTRA[@]}"
