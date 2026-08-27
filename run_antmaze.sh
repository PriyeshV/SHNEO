#!/bin/bash
# SLURM array for the AntMaze continuous case (Mila).
#
#   sbatch run_antmaze.sh
#   python src/mains/ant_maze/summarize_runs.py
#
# Three arms x 12 seeds on ant_u_maze.
#
# Twelve rather than three or four because success-rate AUC here is bimodal:
# a run either reaches ~0.26 or stalls at ~0.14, with a wide empty gap between.
# A reproducibility probe -- four runs of one command at one seed -- spanned
# 0.140 to 0.274, as wide as the spread across different seeds.  So the seed
# explains almost none of the variance, small samples look precise whenever
# they land in one mode, and the statistic worth reporting is the fraction of
# runs reaching the upper mode rather than a mean.  Separating 100% from 60%
# needs roughly a dozen runs per arm.
#
# Checkpoints are written only for seed 0 of each arm; 40 evaluations x 36 runs
# of full checkpoints would be several GB for no extra insight.

#SBATCH --job-name=shneo_modes
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --requeue
#SBATCH --array=0-35
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

SEEDS_PER_ARM=12
ARMS=(SAC NEO_e010 NEO_e030)

ARM_INDEX=$(( SLURM_ARRAY_TASK_ID / SEEDS_PER_ARM ))
SEED=$(( SLURM_ARRAY_TASK_ID % SEEDS_PER_ARM ))
TAG="${ARMS[$ARM_INDEX]}"

COMMON=(
  --e_id ant_u_maze
  --e_pixel_obs False
  --r_input xy
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
  SAC)      SCRIPT=src/mains/ant_maze/RND.py;       EXTRA=(--a_int_coef 0) ;;
  NEO_e010) SCRIPT=src/mains/ant_maze/NEO_LapXY.py; EXTRA=(--a_eps_end 0.10) ;;
  NEO_e030) SCRIPT=src/mains/ant_maze/NEO_LapXY.py; EXTRA=(--a_eps_end 0.30) ;;
  *) echo "unexpected tag $TAG" >&2; exit 1 ;;
esac

echo "task=$SLURM_ARRAY_TASK_ID tag=$TAG seed=$SEED script=$SCRIPT"
exec python "$SCRIPT" "${COMMON[@]}" "${EXTRA[@]}"
