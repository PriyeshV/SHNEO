# Scalable Hierarchical Novel Eigen Options (SHNEO)

JAX implementations of **Scalable Novel Eigen Options (SNEO)** and **Scalable Hierarchical Novel Eigen Options (SHNEO)** for exploration in Atari and Procgen environments.

The agents combine Parallelized Q-Networks (PQN), online Laplacian representation learning, eigenoptions, and Random Network Distillation (RND). SNEO learns novelty-driven exploratory options; SHNEO adds a learned hierarchy that selects between primitive actions and options. The repository also includes PQN, RND, eigenoption, and hierarchical eigenoption baselines.

## Setup

```bash
git clone https://github.com/PriyeshV/SHNEO.git
cd SHNEO
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cuda12.txt
python -m pip check
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

Run the commands below from the repository root. Setting `PYTHONPATH` makes the `src` imports available to both domains' entry points.

The training entry points explicitly select a JAX GPU device. Use Linux x86_64 with an NVIDIA GPU and a driver compatible with CUDA 12. Python 3.10 is a suggested starting environment.

`requirements.txt` contains the shared Atari/Procgen runtime. `requirements-cuda12.txt` adds the archived JAX `0.4.23` CUDA 12 build and its wheel source; use this file for training. Install in a fresh environment to avoid mixing the monolithic CUDA JAX build with the separate plugin/PJRT packages from the old dependency dump. `constraints.txt` keeps SciPy, Orbax, and CUDA libraries compatible with the older stack. These files are reduced dependency specifications.

## Running experiments

Keep run-specific choices on the command line and fixed settings in the configuration files:

| Argument | Purpose                                                                                                                                                                          |
| --- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--x_seed` | Experiment random seed.                                                                                                                                                          |
| `--wb_model` | W&B group label; the script selects the algorithm.                                                                                                                               |
| `--neo_rnd_coef` | RND intrinsic-reward coefficient used by the novelty-driven options.                                                                                                             |
| `--e_id` | Environment name, such as `MontezumaRevenge` or `Venture` for ATARI games. <br/>`Climber`, `Jumper`, `Heist`, `Ninja`, `Leaper`, `Fruitbot` and `Bossfight` for Procgen Hard-7 games. |
| `--a_tot_steps` | Training environment-step budget; scientific notation is accepted.                                                                                                               |

### Atari: SNEO on Montezuma's Revenge

```bash
python src/mains/atari/PQN_NEO.py \
    --x_seed 0 \
    --wb_model SNEO \
    --neo_rnd_coef 100 \
    --e_id MontezumaRevenge \
    --a_tot_steps 1e9
```

### Atari: SHNEO on Venture

```bash
python src/mains/atari/PQN_HNEO.py \
    --x_seed 0 \
    --wb_model SHNEO \
    --neo_rnd_coef 100 \
    --e_id Venture \
    --a_tot_steps 5e7
```

Atari environment names omit the `-v5` suffix, which the environment factory appends automatically. Training uses EnvPool with action repeat 4, episodic life, no sticky actions, and up to 30 initial no-ops. The step budget counts training environment transitions, not raw emulator frames; updates use complete rollout batches.

### Procgen: SHNEO on Climber

```bash
python src/mains/procgen/PQN_HNEO.py \
    --x_seed 0 \
    --wb_model SHNEO \
    --neo_rnd_coef 100 \
    --e_id Climber \
    --a_tot_steps 2.5e7
```

The Procgen factory maps `Climber` with distribution mode `hard` to `ClimberHard-v0`. It currently fixes `num_levels=1` and uses `e_start_level` (default `0`). These runs therefore use a single training level.

Change the seed to run independent trials. Use the environment-specific novelty coefficient and training horizon for each experiment.

## Fixed experiment configuration

| File | Responsibility |
| --- | --- |
| `src/config/common/arg_helpers.py` | Shared parser defaults for optimization, exploration, Laplacian learning, options, RND, and logging. |
| `src/config/atari/env_args.py` | Atari environment arguments. |
| `src/config/procgen/env_args.py` | Procgen environment arguments and distribution settings. |
| `src/config/atari/pqn_laplacian.py` | Atari parser assembly and config conversion. |
| `src/config/procgen/pqn_laplacian.py` | Procgen parser assembly and config conversion. |
| `src/config/common/dataclass_args.py` | Typed configuration structures. |
| `src/config/common/build_config.py` | Derived rollout sizes, nested config construction, and W&B initialization. |

The intended SNEO/SHNEO settings are:

| Parameter | Atari | Procgen |
| --- | --- | --- |
| `lap_br_val` | **0.5** | **0.5** |
| `lap_average` | `False` | `False` |
| `lap_br` / `lap_dual` | `False` / `True` | `False` / `True` |
| `a_int_coef` | `0.0` | `0.0` |
| `e_gamma` / `r_int_gamma` | `0.99` / `0.99` | `0.99` / `0.99` |
| `a_td_lambda`, `o_td_lambda`, `o_meta_td_lambda` | `0.65` | `0.95` |
| `o_gamma` | `0.95` | `0.95` |
| `o_n_options` | `10` | `10` |
| `o_opt_dur` | `10` | `5` |
| `o_policy_greedy` / `o_policy_temp` | `False` / `0.25` | `False` / `0.25` |
| `o_mu_prob` | `0.9` | `0.5` |
| `a_eps_end` | `0.1` | `0.1` |
| `e_n_frames` | `4` | `1` |
| `a_rew_scale` | `True` | `True` |
| `o_stall_coef` | `0.0` | `0.0` |
| `a_use_rnd_eps` | `False` | `False` |
| `e_distribution_mode` | — | `hard` |
| `e_start_level` | — | `0` |
| `e_n_test_envs` | `8` | `8` |

## Algorithms and repository layout

Both `src/mains/atari/` and `src/mains/procgen/` provide the following entry points:

| Script | Method |
| --- | --- |
| `PQN.py` | PQN baseline. |
| `PQN_RND.py` | PQN with RND exploration. |
| `PQN_EO.py` | Eigenoptions. |
| `PQN_HEO.py` | Hierarchical eigenoptions. |
| `PQN_NEO.py` | SNEO. |
| `PQN_HNEO.py` | SHNEO. |

```text
src/
├── config/
│   ├── common/        # Shared arguments and typed configuration
│   ├── atari/         # Atari parser and environment settings
│   └── procgen/       # Procgen parser and environment settings
├── losses/           # Laplacian objectives
├── mains/
│   ├── agents/       # Shared PQN and option-agent training cores
│   ├── algorithms/   # PQN, RND, NEO, EO, HEO, and Laplacian components
│   ├── common/       # Shared environment, network, logging, and video helpers
│   ├── atari/        # Atari entry points and environment factory
│   └── procgen/      # Procgen entry points and environment factory
├── networks/         # Encoders, value networks, and neural-network layers
└── utils/            # JAX helpers and EnvPool wrappers
```
## Logging and evaluation

Configure `wb_entity` and `wb_proj` in the config, then authenticate with `wandb login`. Replace the archived entity default with your own account or team. `wb_model` controls the W&B group, and the seed becomes the job type.

## Citation

If you use this code, please cite the paper accepted at **NeurIPS 2026**:

[Addressing Sparse-Rewards in RL with Scalable Hierarchical Novel Eigen Options](https://neurips.cc/virtual/2026/poster/150071).

```bibtex
@inproceedings{vijayan2026shneo,
  title = {Addressing Sparse-Rewards in {RL} with Scalable Hierarchical Novel Eigen Options},
  author = {Vijayan, Priyesh and Coté-Gauthier, Élodie and Reymond, Mathieu and Chandar, Sarath and Precup, Doina and Prémont-Schwarz, Isabeau},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2026},
  url = {https://neurips.cc/virtual/2026/poster/150071}
}
```
