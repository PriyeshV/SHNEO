#!/bin/bash
# AntMaze results for the paper (Mila).
#
#   sbatch run_antmaze.sh
#   python src/mains/ant_maze/summarize_runs.py
#
# Four arms x 12 seeds on ant_u_maze, all with the PixelAntMaze observation so
# every arm sees an identical environment, matching the paper's pixel-based
# setting.  The SAC actor and critics consume the vector observation in every
# arm; only the novelty model and the Laplacian differ.
#
#   SAC           the project's designated baseline, SAC.py on Brax's own
#                 maintained SAC trainer.
#   SAC_RND       RND on the primitive policy, a_int_coef at its 1.0 default.
#   NEO           eigen-options WITHOUT the novelty bonus (neo_rnd_coef 0).
#                 This is the "direct extension of NEO" baseline.
#   NEO_reliable  eigen-options WITH the novelty bonus (neo_rnd_coef 1.0).
#                 This is the method.
#
# The headline comparison is NEO_reliable against NEO: it isolates the novelty
# bonus, which is the contribution.  SAC and SAC_RND are context.
#
# Twelve seeds because success-rate AUC on this task is bimodal -- a run either
# reaches ~0.26 or stalls at ~0.14 -- and four runs of one command at one seed
# spanned 0.140 to 0.274.  Report the fraction of runs reaching the upper mode
# with a binomial interval, not a mean over three seeds.
#
# Everything goes to a fresh W&B project so the reportable set is
# self-contained: earlier arrays reused group names across code versions and
# only the timestamp separated them.

#SBATCH --job-name=shneo_paper
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --array=0-47
# Relative to the submission directory (the repo), which .gitignore excludes.
# An absolute path built from %u is wrong on Mila, where homes are
# /home/mila/<first-letter>/<user>/ and %u expands to the username alone.
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail

VENV="${SHNEO_VENV:-$HOME/virtualenvs/pybrax_uv}"
REPO="${SHNEO_REPO:-$HOME/SHNEO}"
source "$VENV/bin/activate"
cd "$REPO"

SEEDS_PER_ARM=12
ARMS=(SAC SAC_RND NEO NEO_reliable)

ARM_INDEX=$(( SLURM_ARRAY_TASK_ID / SEEDS_PER_ARM ))
SEED=$((      SLURM_ARRAY_TASK_ID % SEEDS_PER_ARM ))
TAG="${ARMS[$ARM_INDEX]}"

COMMON=(
  --e_id ant_u_maze
  --e_pixel_obs True
  --wb_on True
  --wb_proj SHNEO_antmaze_final
  --e_n_test_envs 64
  --a_n_evals 40
  --x_seed "$SEED"
  --x_file "$TAG"
  --wb_model "$TAG"
)

if [ "$SEED" -eq 0 ]; then
  CKPT="$REPO/ckpt/final_${TAG}_seed0"
  mkdir -p "$CKPT"
  COMMON+=(--a_checkpoint_logdir "$CKPT")
fi

case "$TAG" in
  SAC)          SCRIPT=src/mains/ant_maze/SAC.py; EXTRA=() ;;
  SAC_RND)      SCRIPT=src/mains/ant_maze/RND.py; EXTRA=() ;;
  NEO)          SCRIPT=src/mains/ant_maze/NEO.py; EXTRA=(--neo_rnd_coef 0) ;;
  NEO_reliable) SCRIPT=src/mains/ant_maze/NEO.py; EXTRA=() ;;
  *) echo "unexpected tag $TAG" >&2; exit 1 ;;
esac

echo "task=$SLURM_ARRAY_TASK_ID tag=$TAG seed=$SEED script=$SCRIPT"
exec python "$SCRIPT" "${COMMON[@]}" "${EXTRA[@]}"
