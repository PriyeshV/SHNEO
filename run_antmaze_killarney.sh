#!/bin/bash
# AntMaze results for the paper, on Killarney (Alliance Canada).
#
#   sbatch run_antmaze_killarney.sh
#   # different allocation:
#   sbatch --account=aip-dprecup run_antmaze_killarney.sh
#
# Then, from a LOGIN node once the array finishes:
#   wandb login                      # once per cluster
#   wandb sync --sync-all $SCRATCH/wandb
#
# Compute nodes on Alliance clusters have no outbound internet, so W&B runs
# offline and is uploaded afterwards.  Without this the jobs stall or lose
# their logs entirely.
#
# Same four arms as the Mila script:
#   SAC           the project's designated baseline, SAC.py on Brax's trainer.
#   SAC_RND       RND on the primitive policy, a_int_coef at its 1.0 default.
#   NEO           eigen-options WITHOUT the novelty bonus (neo_rnd_coef 0):
#                 the "direct extension of NEO" baseline.
#   NEO_reliable  eigen-options WITH the novelty bonus: the method.
#
# The headline comparison is NEO_reliable against NEO, which isolates the
# novelty bonus.  SAC and SAC_RND are context.

#SBATCH --job-name=shneo_paper
#SBATCH --account=aip-isabeau
#SBATCH --partition=gpubase_l40s_b2
#SBATCH --time=11:59:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --array=0-47
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail

module load python/3.11.5

VENV="${SHNEO_VENV:-/project/aip-isabeau/$USER/venvs/pybrax}"
REPO="${SHNEO_REPO:-$HOME/projects/SHNEO}"
source "$VENV/bin/activate"
cd "$REPO"

# Keep W&B run directories and checkpoints off /home: its inode quota is the
# binding limit on this cluster, not its capacity.
export WANDB_MODE=offline
export WANDB_DIR="${SCRATCH}/wandb"
mkdir -p "$WANDB_DIR"

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
  CKPT="${SCRATCH}/shneo_ckpt/final_${TAG}_seed0"
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
