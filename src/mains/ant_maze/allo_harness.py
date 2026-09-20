"""Offline ALLO on the AntMaze U-corridor: the official recipe vs SHNEO's loss, same data.

Assembled from the validated pieces of helpers/allo_official_precompute.py (maze, uniform random
walk, official samplers, official train step, per-mode cosine) with ONE addition: a train step
that calls SHNEO's src/losses/laplacian.py multi_step_ALLO + update_ALLO_parameters, read-only,
on the identical walk, encoder and evaluation.  Only the loss (and optionally the optimiser)
differs between arms, so the comparison isolates the objective.

The maze is ANT_MAZE_LAYOUTS["u_maze"] with each cell subdivided SUB x SUB, the same 112-node
graph check_lap_xy.py scores against, so avg |cos| here is directly comparable to those runs.

    JAX_PLATFORMS=cpu python allo_harness.py --loss official --d 11
    JAX_PLATFORMS=cpu python allo_harness.py --loss shneo --d 32 --zeta 0.5       # as the trainer runs it
    JAX_PLATFORMS=cpu python allo_harness.py --loss shneo --d 32 --zeta 0         # sink term off
    JAX_PLATFORMS=cpu python allo_harness.py --loss shneo --d 11 --zeta 0 --opt official
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.getcwd())
import numpy as np

U_MAZE = ["XXXXX",
          "X   X",
          "XXX X",
          "X   X",
          "XXXXX"]
DELTAS = [(1, 0), (0, 1), (-1, 0), (0, -1)]


# ---------------------------------------------------------------- maze + truth (from the harness)
class Maze:
    def __init__(self, rows, sub):
        cells = [(x, y) for y, r in enumerate(rows) for x, c in enumerate(r) if c != "X"]
        self.valid_xy = sorted((x * sub + a, y * sub + b) for (x, y) in cells
                               for a in range(sub) for b in range(sub))
        self.xy_to_idx = {xy: i for i, xy in enumerate(self.valid_xy)}
        self.S = len(self.valid_xy)
        self.width, self.height = len(rows[0]) * sub, len(rows) * sub
        self.T = np.zeros((self.S, 4), dtype=np.int32)
        for i, (x, y) in enumerate(self.valid_xy):
            for a, (dx, dy) in enumerate(DELTAS):
                self.T[i, a] = self.xy_to_idx.get((x + dx, y + dy), i)


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


def make_official_step(encoder, optimizer, bank, d, cfg):
    """Verbatim recipe: graph term on geometric pairs, two INDEPENDENT constraint batches,
    barrier sum(e1*e2) unclipped, duals lr 1e-4 in [-100,100], barrier from 0.1."""
    import jax, jax.numpy as jnp, optax
    eye = jnp.eye(d)

    def gram_error(u):
        return jnp.tril(jnp.einsum("bi,bj->ij", u, jax.lax.stop_gradient(u)) / u.shape[0] - eye)

    @jax.jit
    def step(params, opt_state, betas, barrier, quad_ema, key, i1, i2, iu1, iu2):
        x1, x2, xu1, xu2 = bank[i1], bank[i2], bank[iu1], bank[iu2]

        def loss_fn(p):
            phi1, phi2 = encoder.apply({"params": p}, x1), encoder.apply({"params": p}, x2)
            u1, u2 = encoder.apply({"params": p}, xu1), encoder.apply({"params": p}, xu2)
            graph = ((phi1 - phi2) ** 2).mean(0).sum()
            e1, e2 = gram_error(u1), gram_error(u2)
            errors, quad = 0.5 * (e1 + e2), e1 * e2
            loss = graph + (jax.lax.stop_gradient(betas) * errors).sum() \
                + jax.lax.stop_gradient(barrier) * quad.sum()
            return loss, (graph, errors, quad)

        (loss, (graph, errors, quad)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        first = jnp.isclose(jnp.linalg.norm(quad_ema), 0.0, atol=1e-13).astype(jnp.float32)
        quad_ema = quad_ema + (first + (1 - first) * 0.1) * (quad - quad_ema)
        betas = jnp.tril(jnp.clip(betas + cfg["dual_lr"] * jnp.tril(errors), -100.0, 100.0))
        barrier = jnp.clip(barrier + cfg["barrier_lr"] * jnp.clip(quad_ema, 0.0).mean(), 0.0, 1e4)
        metrics = jnp.stack([loss, graph, jnp.linalg.norm(errors), jnp.linalg.norm(betas), barrier])
        return params, opt_state, betas, barrier, quad_ema, key, metrics

    return step


def make_shneo_step(encoder, optimizer, bank, d, cfg):
    """SHNEO's src/losses/laplacian.py, called exactly as NEO_LapXY.fresh_update calls it, on
    [B, n_steps+1] windows of the same walk.  Read-only import; nothing in the repo is touched."""
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

    @jax.jit
    def step(params, opt_state, allo, key, seq):
        key, lkey = jax.random.split(key)
        x = bank[seq]                                       # [B, n+1, 2]
        B = x.shape[0]

        def loss_fn(p):
            phis = encoder.apply({"params": p}, x.reshape(-1, 2)).reshape(B, n + 1, d)
            loss, (pos, quad, _), (_, err) = multi_step_ALLO(
                lkey, phis[:, 0], phis[:, 1:], jnp.zeros((B, n), jnp.int32),
                jnp.ones((B,), bool), jnp.ones((B, n), bool),
                allo.barrier_coeffs, allo.betas, None,
                novelty_values=jnp.ones((B, n)), zeta_values=jnp.full((B, n + 1), cfg["zeta"]),
                use_novelty=False, use_zeta=cfg["zeta"] > 0, normalize_zeta=False,
                gamma=cfg["gamma"], state_action=False, uni_policy=False, n_steps=n,
                use_dual=True, use_weights=False, average_loss=True, rms_norm=False, barrier_dims=1)
            return loss, (pos, err)

        (loss, (pos, err)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        allo = update_ALLO_parameters(allo, err, rate, barrier_cfg, dual_cfg,
                                      average_loss=True, barrier_dims=1)
        metrics = jnp.stack([loss, pos, jnp.linalg.norm(allo.errors),
                             jnp.linalg.norm(allo.betas), allo.barrier_coeffs[0, 0]])
        return params, opt_state, allo, key, metrics

    return step, init


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=["official", "shneo"], required=True)
    ap.add_argument("--opt", choices=["official", "shneo"], default=None,
                    help="official: Adam 1e-3, no clip.  shneo: clip 10 + adamw 2.5e-4 (the trainer). "
                         "Default follows --loss.")
    ap.add_argument("--d", type=int, default=11)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 256])
    ap.add_argument("--sub", type=int, default=4)
    ap.add_argument("--num-samples", type=int, default=400_000)
    ap.add_argument("--episode-len", type=int, default=50)
    ap.add_argument("--num-updates", type=int, default=20_000)
    ap.add_argument("--eval-every", type=int, default=2_000)
    ap.add_argument("--batch", type=int, default=1024, help="official: pairs per step")
    ap.add_argument("--discount", type=float, default=0.9, help="official: geometric pair lag")
    ap.add_argument("--seq-batch", type=int, default=128, help="shneo: windows per step (= e_n_envs)")
    ap.add_argument("--n-steps", type=int, default=31, help="shneo: futures per window (= lap_seq_len-1)")
    ap.add_argument("--gamma", type=float, default=0.99, help="shneo: e_gamma")
    ap.add_argument("--zeta", type=float, default=0.5,
                    help="shneo: constant zeta fed to the sink term; 0 disables it. The trainer's "
                         "NEO/zeta_mean was 0.28 with max 0.5")
    ap.add_argument("--barrier-init", type=float, default=None, help="official 0.1 / shneo 100")
    ap.add_argument("--barrier-lr", type=float, default=0.01)
    ap.add_argument("--dual-lr", type=float, default=None, help="official 1e-4 / shneo 0.01")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.opt = a.opt or a.loss
    a.barrier_init = a.barrier_init if a.barrier_init is not None else (0.1 if a.loss == "official" else 100.0)
    a.dual_lr = a.dual_lr if a.dual_lr is not None else (1e-4 if a.loss == "official" else 0.01)

    import jax, jax.numpy as jnp, optax
    from flax import linen as nn

    rng = np.random.default_rng(a.seed)
    maze = Maze(U_MAZE, a.sub)
    mu, E = exact_basis(maze, a.d)
    xy = np.array(maze.valid_xy, np.float32)
    bank = jnp.asarray(np.stack([xy[:, 0] / maze.width - 0.5, xy[:, 1] / maze.height - 0.5], 1))
    walk = Walk(maze, a.num_samples, a.episode_len, rng)
    print(f"[maze] u_maze x{a.sub}: {maze.S} nodes | true mu[:8] = {np.round(mu[:8], 4).tolist()}")
    print(f"[walk] {walk.n_ep} episodes x {walk.H} states")

    class MLP(nn.Module):
        @nn.compact
        def __call__(self, x):
            for h in a.hidden:
                x = nn.relu(nn.Dense(h)(x))
            return nn.Dense(a.d)(x)

    enc = MLP()
    params = enc.init(jax.random.PRNGKey(a.seed), bank[:2])["params"]
    if a.opt == "official":
        opt = optax.adam(1e-3)
    else:
        opt = optax.chain(optax.clip_by_global_norm(10.0), optax.adamw(2.5e-4, eps=1.5e-4))
    opt_state = opt.init(params)
    cfg = dict(dual_lr=a.dual_lr, barrier_lr=a.barrier_lr, barrier_init=a.barrier_init,
               n_steps=a.n_steps, gamma=a.gamma, zeta=a.zeta)
    print(f"[run] loss={a.loss} opt={a.opt} d={a.d} hidden={a.hidden} barrier_init={a.barrier_init} "
          f"dual_lr={a.dual_lr}" + (f" n_steps={a.n_steps} zeta={a.zeta}" if a.loss == "shneo" else ""))

    key = jax.random.PRNGKey(a.seed)
    if a.loss == "official":
        step = make_official_step(enc, opt, bank, a.d, cfg)
        betas, barrier = jnp.zeros((a.d, a.d)), jnp.asarray(a.barrier_init)
        quad_ema = jnp.zeros((a.d, a.d))
    else:
        step, allo = make_shneo_step(enc, opt, bank, a.d, cfg)

    def report(it, m, t0):
        Phi = np.asarray(enc.apply({"params": params}, bank))
        cos = per_mode_cosine(Phi, E, mu)
        G = np.diag(Phi.T @ Phi / maze.S)
        print(f"[{it:6d}] avg|cos| {cos.mean():.3f}  first5 {cos[:5].mean():.3f}  min {cos.min():.3f} | "
              f"loss {m[0]:.3f} graph {m[1]:.3f} |err| {m[2]:.3f} |beta| {m[3]:.2f} b {m[4]:.3f} | "
              f"Gram diag [{G.min():.3g}, {G.max():.3g}] | {time.time() - t0:.0f}s")
        print("          per-mode: " + " ".join(f"{c:.2f}" for c in cos[:min(a.d, 12)]))

    t0 = time.time()
    m = np.zeros(5)
    report(0, m, t0)
    for it in range(1, a.num_updates + 1):
        if a.loss == "official":
            i1, i2 = walk.sample_pairs(rng, a.batch, a.discount)
            iu1, iu2 = walk.sample_steps(rng, a.batch), walk.sample_steps(rng, a.batch)
            params, opt_state, betas, barrier, quad_ema, key, m = step(
                params, opt_state, betas, barrier, quad_ema, key,
                jnp.asarray(walk.pos_flat[i1]), jnp.asarray(walk.pos_flat[i2]),
                jnp.asarray(walk.pos_flat[iu1]), jnp.asarray(walk.pos_flat[iu2]))
        else:
            seq = walk.sample_sequences(rng, a.seq_batch, a.n_steps)
            params, opt_state, allo, key, m = step(params, opt_state, allo, key,
                                                   jnp.asarray(walk.pos_flat[seq]))
        if it % a.eval_every == 0:
            report(it, np.asarray(m), t0)


if __name__ == "__main__":
    main()
