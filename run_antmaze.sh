#!/bin/bash
# SLURM array for the AntMaze continuous case (Mila).
#
#   sbatch run_antmaze.sh
#   python src/mains/ant_maze/summarize_runs.py
#
# Three arms x 12 seeds on ant_u_maze, each at its own default configuration:
#
#   SAC_RND    SAC-RND baseline, a_int_coef at the sac_rnd default of 1.0.
#   NEO_xy     SAC-NEO, vector Laplacian, every NEO knob at its default
#              (epsilon floor 0.01, neo_rnd_coef 0, lap weighting on).
#   NEO_pixel  SAC-NEO with the pixel Laplacian, the original formulation.
#
# Together with the SAC, NEO_e010 and NEO_e030 arms already collected, this
# gives six arms x 12 seeds on one maze.
#
# Twelve seeds because success-rate AUC here is bimodal -- a run either
# reaches ~0.26 or stalls at ~0.14 -- and four runs of one command at one seed
# spanned 0.140 to 0.274.  The statistic to report is the fraction of runs
# reaching the upper mode, not a mean over a handful.
#
# Checkpoints only for seed 0 of each arm; 40 evaluations x 36 runs of full
# checkpoints would be several GB for no extra insight.

#SBATCH --job-name=shneo_report
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --requeue
#SBATCH --array=0-35
# Relative to the submission directory (the repo), which .gitignore already
# excludes.  An absolute path built from %u is wrong on Mila, where homes are
# /home/mila/<first-letter>/<user>/ and %u expands to the username alone.
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail

VENV="${SHNEO_VENV:-$HOME/virtualenvs/pybrax_uv}"
REPO="${SHNEO_REPO:-$HOME/SHNEO}"
source "$VENV/bin/activate"
cd "$REPO"

SEEDS_PER_ARM=12
ARMS=(SAC_RND NEO_xy NEO_pixel)

ARM_INDEX=$(( SLURM_ARRAY_TASK_ID / SEEDS_PER_ARM ))
SEED=$(( SLURM_ARRAY_TASK_ID % SEEDS_PER_ARM ))
TAG="${ARMS[$ARM_INDEX]}"

COMMON=(
  --e_id ant_u_maze
  --wb_on True
  --e_n_test_envs 64
  --a_n_evals 40
  --x_seed "$SEED"
  --x_file "$TAG"
  --wb_model "$TAG"
)

if [ "$SEED" -eq 0 ]; then
  CKPT="$REPO/ckpt/${TAG}_seed0"
  mkdir -p "$CKPT"
  COMMON+=(--a_checkpoint_logdir "$CKPT")
fi

case "$TAG" in
  SAC_RND)   SCRIPT=src/mains/ant_maze/RND.py
             EXTRA=(--e_pixel_obs False --r_input xy) ;;
  NEO_xy)    SCRIPT=src/mains/ant_maze/NEO_LapXY.py
             EXTRA=(--e_pixel_obs False --r_input xy) ;;
  NEO_pixel) SCRIPT=src/mains/ant_maze/NEO.py
             EXTRA=(--e_pixel_obs True) ;;
  *) echo "unexpected tag $TAG" >&2; exit 1 ;;
esac

echo "task=$SLURM_ARRAY_TASK_ID tag=$TAG seed=$SEED script=$SCRIPT"
exec python "$SCRIPT" "${COMMON[@]}" "${EXTRA[@]}"
