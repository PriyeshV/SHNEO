"""Offline ALLO on the AntMaze U-corridor: the official recipe vs SHNEO's loss, same data.

Assembled from the validated pieces of helpers/allo_official_precompute.py (maze, uniform random
walk, official samplers, official train step, per-mode cosine) with ONE addition: a train step
that calls SHNEO's src/losses/laplacian.py multi_step_ALLO + update_ALLO_parameters, read-only,
on the identical walk, encoder and evaluation.  Only the loss (and optionally the optimiser)
differs between arms, so the comparison isolates the objective.

The maze is ANT_MAZE_LAYOUTS["u_maze"] with each cell subdivided SUB x SUB, the same 112-node
graph check_lap_xy.py scores against, so avg |cos| here is directly comparable to those runs.
Positions are world coordinates (cell (x, y) is centred on (4x, 4y)) scaled by the trainer's
xy_scale = 20, so the encoder sees exactly what NEO_LapXY feeds it.

Two data sources:
  --data uniform   a long uniform random walk over the graph, sampled i.i.d. (the ALLO paper's
                   setting; what the official recipe is validated on).
  --data online    the trainer's stream: e_n_envs continuous walkers that all reset at the start
                   cell, move --sigma world units per step, end their episode every
                   --episode-len steps, and are read as a rolling window of their last
                   lap_seq_len states every lap_seq_len steps, --epochs gradient steps per window.
                   Nothing is sampled from a buffer.

    JAX_PLATFORMS=cpu python allo_harness.py --loss official --d 11
    JAX_PLATFORMS=cpu python allo_harness.py --loss shneo --d 32 --zeta 0.5       # as the trainer runs it
    JAX_PLATFORMS=cpu python allo_harness.py --loss shneo --d 11 --zeta 0         # 0.998 on the first 5
    JAX_PLATFORMS=cpu python allo_harness.py --loss shneo --d 11 --zeta 0 --data online \
        --encoder trainer --batch-norm --sigma 0.3                                # the online run
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.getcwd())
import numpy as np

# ANT_MAZE_LAYOUTS["u_maze"] indexed [y][x] so that cell (x, y) sits at world (4x, 4y): the start
# is cell (1, 1) = (4, 4) and the goals d1..d6 run (1,2) (1,3) (2,3) (3,3) (3,2) (3,1), exactly as
# check_lap_xy.py's VISITED cells (1,1) (1,2) (1,3) (2,3) read.
U_MAZE = ["XXXXX",
          "X X X",
          "X X X",
          "X   X",
          "XXXXX"]
DELTAS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
CELL = 4.0             # _MAZE_SIZE_SCALING
XY_SCALE = 5 * CELL    # NEO_LapXY: len(layout) * 4.0
START = np.array([4.0, 4.0])


# ---------------------------------------------------------------- maze + truth (from the harness)
class Maze:
    def __init__(self, rows, sub):
        self.cells = [(x, y) for y, r in enumerate(rows) for x, c in enumerate(r) if c != "X"]
        self.free = np.zeros((len(rows[0]), len(rows)), bool)
        for x, y in self.cells:
            self.free[x, y] = True
        self.sub = sub
        self.valid_xy = sorted((x * sub + a, y * sub + b) for (x, y) in self.cells
                               for a in range(sub) for b in range(sub))
        self.xy_to_idx = {xy: i for i, xy in enumerate(self.valid_xy)}
        self.S = len(self.valid_xy)
        self.T = np.zeros((self.S, 4), dtype=np.int32)
        for i, (x, y) in enumerate(self.valid_xy):
            for a, (dx, dy) in enumerate(DELTAS):
                self.T[i, a] = self.xy_to_idx.get((x + dx, y + dy), i)
        self.grid = -np.ones((len(rows[0]) * sub, len(rows) * sub), np.int64)
        for (x, y), i in self.xy_to_idx.items():
            self.grid[x, y] = i

    def world(self):
        """[S, 2] world-coordinate centre of every sub-cell."""
        return (np.array(self.valid_xy, np.float64) + 0.5) * (CELL / self.sub) - CELL / 2

    def is_free(self, p):
        """p: [N, 2] world positions -> [N] bool, inside a free cell."""
        ix = np.floor((p + CELL / 2) / CELL).astype(int)
        inside = (ix >= 0).all(1) & (ix[:, 0] < self.free.shape[0]) & (ix[:, 1] < self.free.shape[1])
        ix = np.clip(ix, 0, np.array(self.free.shape) - 1)
        return inside & self.free[ix[:, 0], ix[:, 1]]

    def sub_index(self, p):
        """p: [N, 2] world positions -> [N] sub-cell index (-1 outside the corridor)."""
        s = np.floor((p + CELL / 2) / (CELL / self.sub)).astype(int)
        s = np.clip(s, 0, np.array(self.grid.shape) - 1)
        return self.grid[s[:, 0], s[:, 1]]


def exact_basis(maze, d):
    P = np.zeros((maze.S, maze.S))
    for s in range(maze.S):
        for a in range(4):
            P[s, maze.T[s, a]] += 0.25
    L = np.eye(maze.S) - 0.5 * (P + P.T)
    mu, E = np.linalg.eigh(L)
    return mu[:d], E[:, :d]


def per_mode_cosine(U, E, mu, tol=1e-8):
    U = np.asarray(U, dtype=np.float64)
    U = U / (np.linalg.norm(U, axis=0, keepdims=True) + 1e-10)
    groups, cur = [], [0]
    for i in range(1, len(mu)):
        (cur.append(i) if abs(mu[i] - mu[i - 1]) < tol else (groups.append(cur), cur := [i]))
    groups.append(cur)
    out = np.zeros(U.shape[1])
    for g in groups:
        for i in g:
            out[i] = np.linalg.norm(E[:, g].T @ U[:, i])
    return out


# ---------------------------------------------------------------- random walk + samplers (from the harness)
class Walk:
    def __init__(self, maze, num_steps, episode_len, rng):
        H = episode_len + 1
        n_ep = max(1, int(np.ceil(num_steps / H)))
        self.H, self.n_ep = H, n_ep
        self.pos = np.zeros((n_ep, H), dtype=np.int32)
        s = rng.integers(maze.S, size=n_ep)
        for h in range(H):
            self.pos[:, h] = s
            s = maze.T[s, rng.integers(4, size=n_ep)]
        self.pos_flat = self.pos.reshape(-1)

    def sample_pairs(self, rng, batch, discount):
        H = self.H
        ep = rng.integers(self.n_ep, size=batch)
        step1 = np.floor(rng.uniform(size=batch) * (H - 1)).astype(np.int64)
        n = H - step1 - 1
        u = rng.uniform(size=batch)
        i = np.floor(np.log(1.0 - (1.0 - np.power(discount, n)) * u) / np.log(discount))
        i = np.clip(i.astype(np.int64), 0, n - 1)
        return ep * H + step1, ep * H + step1 + i + 1

    def sample_steps(self, rng, batch):
        ep = rng.integers(self.n_ep, size=batch)
        return ep * self.H + np.floor(rng.uniform(size=batch) * self.H).astype(np.int64)

    def sample_sequences(self, rng, batch, n_steps):
        """[batch, n_steps+1] consecutive states inside one episode -- SHNEO's window."""
        ep = rng.integers(self.n_ep, size=batch)
        start = rng.integers(self.H - n_steps, size=batch)
        return (ep * self.H)[:, None] + start[:, None] + np.arange(n_steps + 1)[None, :]


class AntWalk:
    """The trainer's stream: n_envs continuous walkers in world coordinates.

    Every walker starts at the start cell, takes a Gaussian step of scale sigma each actor step
    (a step into a wall leaves it where it is), and its episode ends after episode_len steps, on
    which it is put back at the start -- Brax's AutoResetWrapper.  Episodes are synchronous unless
    stagger is set, in which case the initial ages are random so the resets spread out.
    ``push`` returns what NEO_LapXY.fresh_update appends to its window: xy(s_t) and done_t.
    """

    def __init__(self, maze, n_envs, sigma, episode_len, rng, stagger):
        self.maze, self.n, self.sigma, self.H, self.rng = maze, n_envs, sigma, episode_len, rng
        self.pos = self._spawn(n_envs)
        self.age = rng.integers(episode_len, size=n_envs) if stagger else np.zeros(n_envs, int)
        self.visited = np.zeros(maze.S, np.int64)

    def _spawn(self, n):
        return START[None, :] + 0.1 * self.rng.normal(size=(n, 2))

    def push(self):
        xy = self.pos.copy()
        idx = self.maze.sub_index(xy)
        self.visited += np.bincount(idx[idx >= 0], minlength=self.maze.S)
        cand = self.pos + self.sigma * self.rng.normal(size=(self.n, 2))
        ok = self.maze.is_free(cand)
        self.pos = np.where(ok[:, None], cand, self.pos)
        self.age += 1
        done = self.age >= self.H
        if done.any():
            self.pos[done] = self._spawn(int(done.sum()))
            self.age[done] = 0
        return xy, done.astype(np.float32)


# ---------------------------------------------------------------- the two train steps
@dataclass(frozen=True)
class BarrierConfig:
    on: bool; lr: float; clip: bool; value: float


@dataclass(frozen=True)
class DualConfig:
    on: bool; lr: float; clip: bool


@dataclass(frozen=True)
class RateConfig:
    error: float; q_error: float


def make_apply(encoder):
    """encoder.apply with the batch_stats collection threaded through when the encoder has one."""

    def apply(params, batch_stats, x, train):
        if batch_stats is None:
            return encoder.apply({"params": params}, x), None
        variables = {"params": params, "batch_stats": batch_stats}
        if not train:
            return encoder.apply(variables, x, train=False), batch_stats
        out, new = encoder.apply(variables, x, train=True, mutable=["batch_stats"])
        return out, new["batch_stats"]

    return apply


def make_official_step(encoder, optimizer, bank, d, cfg):
    """Verbatim recipe: graph term on geometric pairs, two INDEPENDENT constraint batches,
    barrier sum(e1*e2) unclipped, duals lr 1e-4 in [-100,100], barrier from 0.1."""
    import jax, jax.numpy as jnp, optax
    eye = jnp.eye(d)
    apply = make_apply(encoder)

    def gram_error(u):
        return jnp.tril(jnp.einsum("bi,bj->ij", u, jax.lax.stop_gradient(u)) / u.shape[0] - eye)

    @jax.jit
    def step(params, batch_stats, opt_state, betas, barrier, quad_ema, key, i1, i2, iu1, iu2):
        x1, x2, xu1, xu2 = bank[i1], bank[i2], bank[iu1], bank[iu2]
        B = x1.shape[0]

        def loss_fn(p):
            # One forward pass so a batch-norm encoder sees one set of statistics.
            out, new_bs = apply(p, batch_stats, jnp.concatenate([x1, x2, xu1, xu2]), True)
            phi1, phi2, u1, u2 = out[:B], out[B:2 * B], out[2 * B:3 * B], out[3 * B:]
            graph = ((phi1 - phi2) ** 2).mean(0).sum()
            e1, e2 = gram_error(u1), gram_error(u2)
            errors, quad = 0.5 * (e1 + e2), e1 * e2
            loss = graph + (jax.lax.stop_gradient(betas) * errors).sum() \
                + jax.lax.stop_gradient(barrier) * quad.sum()
            return loss, (graph, errors, quad, new_bs)

        (loss, (graph, errors, quad, batch_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        first = jnp.isclose(jnp.linalg.norm(quad_ema), 0.0, atol=1e-13).astype(jnp.float32)
        quad_ema = quad_ema + (first + (1 - first) * 0.1) * (quad - quad_ema)
        betas = jnp.tril(jnp.clip(betas + cfg["dual_lr"] * jnp.tril(errors), -100.0, 100.0))
        barrier = jnp.clip(barrier + cfg["barrier_lr"] * jnp.clip(quad_ema, 0.0).mean(), 0.0, 1e4)
        metrics = jnp.stack([loss, graph, jnp.linalg.norm(errors), jnp.linalg.norm(betas), barrier])
        return params, batch_stats, opt_state, betas, barrier, quad_ema, key, metrics

    return step


def make_shneo_step(encoder, optimizer, d, cfg):
    """SHNEO's src/losses/laplacian.py, called exactly as NEO_LapXY.fresh_update calls it, on
    [B, n_steps+1] windows x of scaled xy with done flags.  Read-only import; nothing in the repo
    is touched."""
    import jax, jax.numpy as jnp, optax
    import flax.struct
    from src.losses.laplacian import multi_step_ALLO, update_ALLO_parameters

    @flax.struct.dataclass
    class ALLOState:
        barrier_coeffs: jnp.ndarray
        betas: jnp.ndarray
        errors: jnp.ndarray
        quadratic_errors: jnp.ndarray

    n = cfg["n_steps"]
    rate = RateConfig(1.0, 0.1)
    barrier_cfg = BarrierConfig(True, cfg["barrier_lr"], True, cfg["barrier_init"])
    dual_cfg = DualConfig(True, cfg["dual_lr"], True)
    init = ALLOState(jnp.ones((1, 1)) * cfg["barrier_init"], jnp.zeros((d, d)),
                     jnp.zeros((d, d)), jnp.zeros((d, d)))
    apply = make_apply(encoder)

    @jax.jit
    def step(params, batch_stats, opt_state, allo, key, x, done):
        key, lkey = jax.random.split(key)
        B = x.shape[0]                                      # x [B, n+1, 2], done [B, n+1]
        masks = 1.0 - done
        sample_mask, future_masks = masks[:, 0].astype(bool), masks[:, 1:].astype(bool)

        def loss_fn(p):
            phi_flat, new_bs = apply(p, batch_stats, x.reshape(-1, 2), True)
            phis = phi_flat.reshape(B, n + 1, d)
            loss, (pos, quad, _), (_, err) = multi_step_ALLO(
                lkey, phis[:, 0], phis[:, 1:], jnp.zeros((B, n), jnp.int32),
                sample_mask, future_masks,
                allo.barrier_coeffs, allo.betas, None,
                novelty_values=jnp.ones((B, n)), zeta_values=jnp.full((B, n + 1), cfg["zeta"]),
                use_novelty=False, use_zeta=cfg["zeta"] > 0, normalize_zeta=False,
                gamma=cfg["gamma"], state_action=False, uni_policy=False, n_steps=n,
                use_dual=True, use_weights=False, average_loss=True, rms_norm=False, barrier_dims=1)
            return loss, (pos, err, new_bs)

        (loss, (pos, err, batch_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        allo = update_ALLO_parameters(allo, err, rate, barrier_cfg, dual_cfg,
                                      average_loss=True, barrier_dims=1)
        metrics = jnp.stack([loss, pos, jnp.linalg.norm(allo.errors),
                             jnp.linalg.norm(allo.betas), allo.barrier_coeffs[0, 0]])
        return params, batch_stats, opt_state, allo, key, metrics

    return step, init


def trainer_encoder():
    """NEO_LapXY.LaplacianMLP, imported the way plot_eigenfunctions.py does: the trainer parses
    the SAC command line at import, so hand it an empty one for the duration."""
    saved, sys.argv = sys.argv, [sys.argv[0], "--wb_on", "False"]
    try:
        from src.mains.ant_maze.NEO_LapXY import LaplacianMLP
    finally:
        sys.argv = saved
    return LaplacianMLP


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=["official", "shneo"], required=True)
    ap.add_argument("--opt", choices=["official", "shneo"], default=None,
                    help="official: Adam 1e-3, no clip.  shneo: clip 10 + adamw 2.5e-4 (the trainer). "
                         "Default follows --loss.")
    ap.add_argument("--encoder", choices=["official", "trainer"], default="official",
                    help="official: relu MLP --hidden with output bias.  trainer: NEO_LapXY.LaplacianMLP "
                         "(no output bias, optional batch norm) with --hidden, default [512].")
    ap.add_argument("--batch-norm", action="store_true", help="trainer encoder: --lap_batch_norm True")
    ap.add_argument("--d", type=int, default=11)
    ap.add_argument("--hidden", type=int, nargs="+", default=None)
    ap.add_argument("--sub", type=int, default=4)
    ap.add_argument("--data", choices=["uniform", "online"], default="uniform")
    ap.add_argument("--num-samples", type=int, default=400_000, help="uniform: walk length")
    ap.add_argument("--episode-len", type=int, default=None,
                    help="uniform: 50 (episodes of the walk).  online: 1000 (e_max_episode_steps)")
    ap.add_argument("--num-updates", type=int, default=20_000, help="uniform: gradient steps")
    ap.add_argument("--eval-every", type=int, default=None,
                    help="gradient steps between reports: uniform 2000, online 500")
    ap.add_argument("--batch", type=int, default=1024, help="official: pairs per step")
    ap.add_argument("--discount", type=float, default=0.9, help="official: geometric pair lag")
    ap.add_argument("--seq-batch", type=int, default=128, help="shneo: windows per step (= e_n_envs)")
    ap.add_argument("--n-steps", type=int, default=31, help="shneo: futures per window (= lap_seq_len-1)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="shneo: gradient steps per window (lap_epochs): uniform 1, online 2 as the trainer")
    ap.add_argument("--gamma", type=float, default=0.99, help="shneo: e_gamma")
    ap.add_argument("--zeta", type=float, default=0.5,
                    help="shneo: constant zeta fed to the sink term; 0 disables it. The trainer's "
                         "NEO/zeta_mean was 0.28 with max 0.5")
    ap.add_argument("--barrier-init", type=float, default=None, help="official 0.1 / shneo 100")
    ap.add_argument("--barrier-lr", type=float, default=0.01)
    ap.add_argument("--dual-lr", type=float, default=None, help="official 1e-4 / shneo 0.01")
    ap.add_argument("--actor-steps", type=int, default=78_125,
                    help="online: vectorised env steps (a_tot_steps / e_n_envs = 10M / 128)")
    ap.add_argument("--sigma", type=float, default=0.3,
                    help="online: world units moved per step; the corridor is 4 wide")
    ap.add_argument("--stagger", action="store_true", help="online: random initial ages, so episodes "
                                                           "do not all reset together")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.opt = a.opt or a.loss
    a.hidden = a.hidden or ([512] if a.encoder == "trainer" else [256, 256, 256])
    a.episode_len = a.episode_len or (1000 if a.data == "online" else 50)
    a.eval_every = a.eval_every or (500 if a.data == "online" else 2000)
    a.epochs = a.epochs or (2 if a.data == "online" else 1)
    a.barrier_init = a.barrier_init if a.barrier_init is not None else (0.1 if a.loss == "official" else 100.0)
    a.dual_lr = a.dual_lr if a.dual_lr is not None else (1e-4 if a.loss == "official" else 0.01)
    if a.data == "online" and a.loss == "official":
        ap.error("--data online is the trainer's window stream; it only feeds --loss shneo")
    if a.batch_norm and a.encoder != "trainer":
        ap.error("--batch-norm needs --encoder trainer")

    import jax, jax.numpy as jnp, optax
    from flax import linen as nn

    rng = np.random.default_rng(a.seed)
    maze = Maze(U_MAZE, a.sub)
    mu, E = exact_basis(maze, a.d)
    bank = jnp.asarray(maze.world() / XY_SCALE, jnp.float32)   # [S, 2] as NEO_LapXY._lap_obs scales it
    print(f"[maze] u_maze x{a.sub}: {maze.S} nodes | true mu[:8] = {np.round(mu[:8], 4).tolist()}")

    if a.encoder == "trainer":
        enc = trainer_encoder()(hidden_dims=tuple(a.hidden), output_dim=a.d, batch_norm=a.batch_norm)
    else:
        class MLP(nn.Module):
            @nn.compact
            def __call__(self, x, train=False):
                for h in a.hidden:
                    x = nn.relu(nn.Dense(h)(x))
                return nn.Dense(a.d)(x)
        enc = MLP()
    variables = enc.init(jax.random.PRNGKey(a.seed), bank[:2])
    params, batch_stats = variables["params"], variables.get("batch_stats")
    if a.opt == "official":
        opt = optax.adam(1e-3)
    else:
        opt = optax.chain(optax.clip_by_global_norm(10.0), optax.adamw(2.5e-4, eps=1.5e-4))
    opt_state = opt.init(params)
    cfg = dict(dual_lr=a.dual_lr, barrier_lr=a.barrier_lr, barrier_init=a.barrier_init,
               n_steps=a.n_steps, gamma=a.gamma, zeta=a.zeta)
    print(f"[run] loss={a.loss} opt={a.opt} data={a.data} encoder={a.encoder} hidden={a.hidden} "
          f"bn={a.batch_norm} d={a.d} barrier_init={a.barrier_init} dual_lr={a.dual_lr}"
          + (f" n_steps={a.n_steps} epochs={a.epochs} zeta={a.zeta}" if a.loss == "shneo" else "")
          + (f" sigma={a.sigma} episode_len={a.episode_len} stagger={a.stagger}" if a.data == "online" else ""))

    key = jax.random.PRNGKey(a.seed)
    if a.loss == "official":
        step = make_official_step(enc, opt, bank, a.d, cfg)
        betas, barrier = jnp.zeros((a.d, a.d)), jnp.asarray(a.barrier_init)
        quad_ema = jnp.zeros((a.d, a.d))
    else:
        step, allo = make_shneo_step(enc, opt, a.d, cfg)

    def report(it, m, t0, extra=""):
        Phi = np.asarray(make_apply(enc)(params, batch_stats, bank, False)[0])
        cos = per_mode_cosine(Phi, E, mu)
        G = np.diag(Phi.T @ Phi / maze.S)
        print(f"[{it:6d}] avg|cos| {cos.mean():.3f}  first5 {cos[:5].mean():.3f}  min {cos.min():.3f} | "
              f"loss {m[0]:.3f} graph {m[1]:.3f} |err| {m[2]:.3f} |beta| {m[3]:.2f} b {m[4]:.3f} | "
              f"Gram diag [{G.min():.3g}, {G.max():.3g}] | {time.time() - t0:.0f}s{extra}")
        print("          per-mode: " + " ".join(f"{c:.2f}" for c in cos[:min(a.d, 12)]))

    t0 = time.time()
    m = np.zeros(5)
    report(0, m, t0)

    if a.data == "uniform":
        walk = Walk(maze, a.num_samples, a.episode_len, rng)
        print(f"[walk] {walk.n_ep} episodes x {walk.H} states")
        for it in range(1, a.num_updates + 1):
            if a.loss == "official":
                i1, i2 = walk.sample_pairs(rng, a.batch, a.discount)
                iu1, iu2 = walk.sample_steps(rng, a.batch), walk.sample_steps(rng, a.batch)
                params, batch_stats, opt_state, betas, barrier, quad_ema, key, m = step(
                    params, batch_stats, opt_state, betas, barrier, quad_ema, key,
                    jnp.asarray(walk.pos_flat[i1]), jnp.asarray(walk.pos_flat[i2]),
                    jnp.asarray(walk.pos_flat[iu1]), jnp.asarray(walk.pos_flat[iu2]))
            else:
                seq = walk.sample_sequences(rng, a.seq_batch, a.n_steps)
                x = bank[jnp.asarray(walk.pos_flat[seq])]
                for _ in range(a.epochs):
                    params, batch_stats, opt_state, allo, key, m = step(
                        params, batch_stats, opt_state, allo, key, x, jnp.zeros(x.shape[:2]))
            if it % a.eval_every == 0:
                report(it, np.asarray(m), t0)
        return

    # --data online: the trainer's rolling window, updated once per full window.
    K = a.n_steps + 1
    ant = AntWalk(maze, a.seq_batch, a.sigma, a.episode_len, rng, a.stagger)
    seq_xy = np.zeros((a.seq_batch, K, 2), np.float32)
    seq_done = np.ones((a.seq_batch, K), np.float32)      # SNEOState init: everything masked
    updates = 0
    for t in range(1, a.actor_steps + 1):
        xy, done = ant.push()
        seq_xy = np.concatenate([seq_xy[:, 1:], (xy / XY_SCALE)[:, None, :].astype(np.float32)], 1)
        seq_done = np.concatenate([seq_done[:, 1:], done[:, None]], 1)
        if t % K:
            continue
        x, dn = jnp.asarray(seq_xy), jnp.asarray(seq_done)
        for _ in range(a.epochs):
            params, batch_stats, opt_state, allo, key, m = step(
                params, batch_stats, opt_state, allo, key, x, dn)
            updates += 1
            if updates % a.eval_every == 0:
                span = np.linalg.norm(seq_xy[:, -1] - seq_xy[:, 0], axis=1).mean() * XY_SCALE
                report(updates, np.asarray(m), t0,
                       f" | env step {t * a.seq_batch:,} visited {int((ant.visited > 0).sum())}/{maze.S} "
                       f"window span {span:.2f}")
    span = np.linalg.norm(seq_xy[:, -1] - seq_xy[:, 0], axis=1).mean() * XY_SCALE
    report(updates, np.asarray(m), t0,
           f" | env step {a.actor_steps * a.seq_batch:,} visited {int((ant.visited > 0).sum())}/{maze.S} "
           f"window span {span:.2f}")


if __name__ == "__main__":
    main()
