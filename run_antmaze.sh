#!/bin/bash
# SLURM array for the AntMaze continuous case (Mila).
#
#   sbatch run_antmaze.sh
#
# Four arms x 3 seeds on ant_u_maze: a SAC baseline plus SAC-NEO at three
# epsilon floors.  The floor is the interesting knob because it sets what
# fraction of transitions eigen-options generate once the schedule has decayed.
# At the inherited default of 0.01 that is 1%, i.e. NEO is SAC for 90% of
# training -- which is what the first 21 runs measured.
#
# Checkpoints go to ckpt/armN_seedM/ so the learned eigenfunctions survive the
# job and can be drawn with src/mains/ant_maze/plot_eigenfunctions.py.

#SBATCH --job-name=shneo_eps
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --requeue
#SBATCH --array=0-11
#SBATCH --output=/home/mila/%u/slurm-%A_%a.out

set -euo pipefail

VENV="${SHNEO_VENV:-$HOME/virtualenvs/pybrax_uv}"
REPO="${SHNEO_REPO:-$HOME/SHNEO}"
source "$VENV/bin/activate"
cd "$REPO"

ARM=$(( SLURM_ARRAY_TASK_ID / 3 ))
SEED=$(( SLURM_ARRAY_TASK_ID % 3 ))
CKPT="$REPO/ckpt/arm${ARM}_seed${SEED}"
mkdir -p "$CKPT"

COMMON=(
  --e_id ant_u_maze
  --e_pixel_obs False
  --r_input xy
  --wb_on True
  --e_n_test_envs 64
  --a_checkpoint_logdir "$CKPT"
  --x_seed "$SEED"
)

case "$ARM" in
  0) SCRIPT=src/mains/ant_maze/RND.py;       EXTRA=(--a_int_coef 0   --x_file SAC        --wb_model SAC) ;;
  1) SCRIPT=src/mains/ant_maze/NEO_LapXY.py; EXTRA=(--a_eps_end 0.01 --x_file NEO_eps001 --wb_model NEO_eps001) ;;
  2) SCRIPT=src/mains/ant_maze/NEO_LapXY.py; EXTRA=(--a_eps_end 0.10 --x_file NEO_eps010 --wb_model NEO_eps010) ;;
  3) SCRIPT=src/mains/ant_maze/NEO_LapXY.py; EXTRA=(--a_eps_end 0.30 --x_file NEO_eps030 --wb_model NEO_eps030) ;;
  *) echo "unexpected arm $ARM" >&2; exit 1 ;;
esac

echo "task=$SLURM_ARRAY_TASK_ID arm=$ARM seed=$SEED script=$SCRIPT"
exec python "$SCRIPT" "${COMMON[@]}" "${EXTRA[@]}"
