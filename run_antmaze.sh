#!/bin/bash
# SLURM array for the AntMaze continuous case (Mila).
#
#   sbatch run_antmaze.sh
#
# Four arms on ant_u_maze, 16 tasks:
#
#   SAC_rep   x4   SAC baseline, seed 0 every time.  A reproducibility probe:
#                  identical command, identical seed.  Their spread is the
#                  real run-to-run noise floor, which every arm comparison
#                  has to clear before it means anything.  Two earlier grids
#                  gave this same configuration AUC 0.274 and 0.139.
#   SAC       x4   seeds 0-3, the baseline proper.
#   NEO_e010  x4   seeds 0-3, eigen-options at a 10% epsilon floor.
#   NEO_e030  x4   seeds 0-3, 30% floor.
#
# The inherited floor of 0.01 leaves options generating 1% of transitions once
# the schedule decays, i.e. NEO is SAC for 90% of training.  Higher floors keep
# them in the behaviour mix.
#
# a_n_evals is 40 rather than 20: the trapezoidal AUC is computed over the
# evaluation points, so doubling them halves its discretization.
#
# Checkpoints land in ckpt/<tag>_seed<N>/ so the learned eigenfunctions survive
# the job for src/mains/ant_maze/plot_eigenfunctions.py.

#SBATCH --job-name=shneo_repro
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --requeue
#SBATCH --array=0-15
# Relative to the submission directory (the repo), which .gitignore already
# excludes.  An absolute path built from %u is wrong on Mila, where homes are
# /home/mila/<first-letter>/<user>/ and %u expands to the username alone --
# SLURM then cannot open the file and fails every task before it starts.
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail

VENV="${SHNEO_VENV:-$HOME/virtualenvs/pybrax_uv}"
REPO="${SHNEO_REPO:-$HOME/SHNEO}"
source "$VENV/bin/activate"
cd "$REPO"

#            0        1        2        3       4   5   6   7
TAGS=(   SAC_rep  SAC_rep  SAC_rep  SAC_rep   SAC SAC SAC SAC \
         NEO_e010 NEO_e010 NEO_e010 NEO_e010  NEO_e030 NEO_e030 NEO_e030 NEO_e030 )
SEEDS=(     0        0        0        0       0   1   2   3 \
            0        1        2        3       0   1   2   3 )

TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
CKPT="$REPO/ckpt/${TAG}_seed${SEED}_t${SLURM_ARRAY_TASK_ID}"
mkdir -p "$CKPT"

COMMON=(
  --e_id ant_u_maze
  --e_pixel_obs False
  --r_input xy
  --wb_on True
  --e_n_test_envs 64
  --a_n_evals 40
  --a_checkpoint_logdir "$CKPT"
  --x_seed "$SEED"
  --x_file "$TAG"
  --wb_model "$TAG"
)

case "$TAG" in
  SAC|SAC_rep) SCRIPT=src/mains/ant_maze/RND.py;       EXTRA=(--a_int_coef 0) ;;
  NEO_e010)    SCRIPT=src/mains/ant_maze/NEO_LapXY.py; EXTRA=(--a_eps_end 0.10) ;;
  NEO_e030)    SCRIPT=src/mains/ant_maze/NEO_LapXY.py; EXTRA=(--a_eps_end 0.30) ;;
  *) echo "unexpected tag $TAG" >&2; exit 1 ;;
esac

echo "task=$SLURM_ARRAY_TASK_ID tag=$TAG seed=$SEED script=$SCRIPT"
exec python "$SCRIPT" "${COMMON[@]}" "${EXTRA[@]}"
