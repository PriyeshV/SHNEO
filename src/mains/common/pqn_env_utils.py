import time
from functools import partial
from pathlib import Path
from typing import Union

import imageio.v2 as imageio
import jax
import chex
import flax
import optax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training.train_state import TrainState

from src.utils.jax_utils import learning_rate_fn
from src.networks.networks import FFNetwork, Conv_Encoder


# ---------------------------------------------------------------------
# Common TrainState wrappers
# ---------------------------------------------------------------------
class Enc_State(TrainState):
    batch_stats: Union[None, flax.core.FrozenDict] = None


class FFN_State(TrainState):
    batch_stats: Union[None, flax.core.FrozenDict] = None


# ---------------------------------------------------------------------
# Common observation helpers
# ---------------------------------------------------------------------
def obs_chw_to_hwc(obs):
    return jnp.transpose(obs, (0, 2, 3, 1))


def single_obs_chw_to_hwc(obs0):
    return jnp.expand_dims(np.transpose(obs0, (1, 2, 0)), 0)


def chw_frame_to_hwc_uint8(obs_chw):
    x = np.asarray(obs_chw)
    if x.ndim != 3:
        raise ValueError(f"Expected CHW frame, got shape={x.shape}")
    x = np.transpose(x, (1, 2, 0))
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    return x


# ---------------------------------------------------------------------
# Common network builders
# ---------------------------------------------------------------------
def make_common_encoder_kwargs(args):
    return dict(
        min_scale=1,
        width_mul=1,
        impala_cnn=getattr(args.net, "impala", False),
        batch_norm=args.net.batch_norm,
        layer_norm=args.net.layer_norm,
    )


def build_encoder_network(args, name="PQN:Enc"):
    common_encoder_kwargs = make_common_encoder_kwargs(args)
    assert len(args.net.c_dims) > 0
    return Conv_Encoder(
        name=name,
        layer_dims=list(args.net.c_dims),
        activations=[nn.relu] * len(args.net.c_dims),
        **common_encoder_kwargs,
    )


def build_q_network(args, n_actions, name="PQN:Q"):
    return FFNetwork(
        name=name,
        stop_grad=False,
        ip_batch_norm=False,
        batch_norm=args.net.batch_norm,
        layer_norm=args.net.layer_norm,
        layer_dims=list(args.net.q_dims) + [n_actions],
        activations=[nn.relu] * len(args.net.q_dims) + [None],
    )


def build_lr_schedule(args):
    return learning_rate_fn(
        init_lr=args.net.lr,
        alpha=1.0,
        num_epochs=args.algo.n_upds,
        warmup_epochs=max(1, args.algo.n_upds // 4),
        steps_per_epoch=args.algo.n_mini_batches * args.algo.n_epochs,
    )


def build_enc_state(key, args, obs, name="PQN:Enc"):
    enc_network = build_encoder_network(args, name=name)
    obs_init = single_obs_chw_to_hwc(obs[0])

    enc_variables = enc_network.init(key, obs_init, train=False)
    enc_optimizer = optax.chain(
        optax.clip_by_global_norm(args.net.max_grad_norm),
        optax.adamw(learning_rate=build_lr_schedule(args), eps=1.5e-4),
    )
    enc_batch_stats = enc_variables["batch_stats"] if args.net.batch_norm else None
    enc_state = Enc_State.create(
        apply_fn=enc_network.apply,
        tx=enc_optimizer,
        params=enc_variables["params"],
        batch_stats=enc_batch_stats,
    )
    return enc_state


def build_ffn_state(key, args, n_actions, name="PQN:Q"):
    q_network = build_q_network(args, n_actions, name=name)
    dummy_embedding = jnp.ones((1, args.net.c_dims[-1]), dtype=jnp.float32)

    q_variables = q_network.init(key, dummy_embedding, train=False)
    q_optimizer = optax.chain(
        optax.clip_by_global_norm(args.net.max_grad_norm),
        optax.adamw(learning_rate=build_lr_schedule(args), eps=1.5e-4),
    )
    q_batch_stats = q_variables["batch_stats"] if args.net.batch_norm else None
    q_state = FFN_State.create(
        apply_fn=q_network.apply,
        tx=q_optimizer,
        params=q_variables["params"],
        batch_stats=q_batch_stats,
    )
    return q_state


# ---------------------------------------------------------------------
# Common action selection
# ---------------------------------------------------------------------
@jax.jit
def eps_greedy_exploration(rng, q_vals, eps):
    rng_a, rng_e = jax.random.split(rng)
    greedy_actions = jnp.argmax(q_vals, axis=-1)
    chosen_actions = jnp.where(
        jax.random.uniform(rng_e, greedy_actions.shape) < eps,
        jax.random.randint(rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]),
        greedy_actions,
    )
    return chosen_actions


# ---------------------------------------------------------------------
# Common Q forward helpers
# ---------------------------------------------------------------------
@partial(jax.jit, static_argnums=(3, 4))
def get_q_values_generic(obs, enc_state, q_state, use_enc_bn, use_q_bn):
    obs = obs_chw_to_hwc(obs)

    enc_vars = {"params": enc_state.params}
    if use_enc_bn:
        enc_vars["batch_stats"] = enc_state.batch_stats
    embeddings = enc_state.apply_fn(enc_vars, obs)

    q_vars = {"params": q_state.params}
    if use_q_bn:
        q_vars["batch_stats"] = q_state.batch_stats
    q_values = q_state.apply_fn(q_vars, embeddings)
    return q_values


@partial(jax.jit, static_argnums=(4, 5, 6))
def get_dual_q_values_generic(obs, enc_state, q_ext_state, q_int_state, use_enc_bn, use_qext_bn, use_qint_bn):
    obs = obs_chw_to_hwc(obs)

    enc_vars = {"params": enc_state.params}
    if use_enc_bn:
        enc_vars["batch_stats"] = enc_state.batch_stats
    embeddings = enc_state.apply_fn(enc_vars, obs)

    qext_vars = {"params": q_ext_state.params}
    if use_qext_bn:
        qext_vars["batch_stats"] = q_ext_state.batch_stats

    qint_vars = {"params": q_int_state.params}
    if use_qint_bn:
        qint_vars["batch_stats"] = q_int_state.batch_stats

    q_ext_vals = q_ext_state.apply_fn(qext_vars, embeddings)
    q_int_vals = q_int_state.apply_fn(qint_vars, embeddings)
    return q_ext_vals, q_int_vals


@partial(jax.jit, static_argnums=range(2, 6))
def combine_dual_q_values(q_ext, q_int, int_coef: float, use_value_rescale: bool, value_rescale_fn, inv_value_rescale_fn):
    if use_value_rescale:
        q_sel_raw = inv_value_rescale_fn(q_ext) + int_coef * inv_value_rescale_fn(q_int)
        q_sel = value_rescale_fn(q_sel_raw)
    else:
        q_sel = q_ext + int_coef * q_int
    return q_sel


def _stack_state_field(states, field_name: str):
    vals = [getattr(s, field_name) for s in states]
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *vals)


def get_q_values_generic_controller(
    obs,
    enc_state,
    use_enc_bn: bool,
    use_q_bn: bool,
    *,
    q_state=None,
    q_ext_state=None,
    q_int_state=None,
    int_coef: float = 0.0,
    use_value_rescale: bool = False,
    value_rescale_fn=None,
    inv_value_rescale_fn=None,
):
    if q_state is not None:
        q = get_q_values_generic(obs, enc_state, q_state, use_enc_bn, use_q_bn)
        return None, None, q

    assert q_ext_state is not None and q_int_state is not None
    q_ext = get_q_values_generic(obs, enc_state, q_ext_state, use_enc_bn, use_q_bn)
    q_int = get_q_values_generic(obs, enc_state, q_int_state, use_enc_bn, use_q_bn)

    q_sel = combine_dual_q_values(
        q_ext, q_int,
        int_coef=int_coef,
        use_value_rescale=use_value_rescale,
        value_rescale_fn=value_rescale_fn,
        inv_value_rescale_fn=inv_value_rescale_fn,
    )
    return q_ext, q_int, q_sel


def compute_option_qs_generic(
    obs,
    opt_enc_state,
    use_enc_bn: bool,
    use_q_bn: bool,
    *,
    opt_q_states=None,
    opt_q_ext_states=None,
    opt_q_int_states=None,
    int_coef: float = 0.0,
    use_value_rescale: bool = False,
    value_rescale_fn=None,
    inv_value_rescale_fn=None,
):
    obs_hwc = obs_chw_to_hwc(obs)

    enc_vars = {"params": opt_enc_state.params}
    if use_enc_bn:
        enc_vars["batch_stats"] = opt_enc_state.batch_stats
    emb = opt_enc_state.apply_fn(enc_vars, obs_hwc, train=False)

    if opt_q_states is not None:
        q_params = _stack_state_field(opt_q_states, "params")
        q_bs = _stack_state_field(opt_q_states, "batch_stats") if use_q_bn else None
        q_apply = opt_q_states[0].apply_fn

        def q_one(p_q, bs_q):
            vars_q = {"params": p_q}
            if use_q_bn:
                vars_q["batch_stats"] = bs_q
            return q_apply(vars_q, emb, train=False)

        q_sel = jax.vmap(q_one, in_axes=(0, 0 if use_q_bn else None))(q_params, q_bs)
        return None, None, q_sel

    assert opt_q_ext_states is not None and opt_q_int_states is not None

    qext_params = _stack_state_field(opt_q_ext_states, "params")
    qint_params = _stack_state_field(opt_q_int_states, "params")

    qext_bs = _stack_state_field(opt_q_ext_states, "batch_stats") if use_q_bn else None
    qint_bs = _stack_state_field(opt_q_int_states, "batch_stats") if use_q_bn else None

    qext_apply = opt_q_ext_states[0].apply_fn
    qint_apply = opt_q_int_states[0].apply_fn

    def qext_one(p_q, bs_q):
        vars_q = {"params": p_q}
        if use_q_bn:
            vars_q["batch_stats"] = bs_q
        return qext_apply(vars_q, emb, train=False)

    def qint_one(p_q, bs_q):
        vars_q = {"params": p_q}
        if use_q_bn:
            vars_q["batch_stats"] = bs_q
        return qint_apply(vars_q, emb, train=False)

    q_ext = jax.vmap(qext_one, in_axes=(0, 0 if use_q_bn else None))(qext_params, qext_bs)
    q_int = jax.vmap(qint_one, in_axes=(0, 0 if use_q_bn else None))(qint_params, qint_bs)

    q_sel = combine_dual_q_values(
        q_ext, q_int,
        int_coef=int_coef,
        use_value_rescale=use_value_rescale,
        value_rescale_fn=value_rescale_fn,
        inv_value_rescale_fn=inv_value_rescale_fn,
    )
    return q_ext, q_int, q_sel


# ---------------------------------------------------------------------
# Common evaluation head picker
# ---------------------------------------------------------------------
def pick_eval_head(q_ext, q_int, q_sel, mode: str):
    mode = mode.lower()
    if mode == "sel":
        return q_sel
    if mode == "ext":
        return q_sel if q_ext is None else q_ext
    if mode == "int":
        return q_sel if q_int is None else q_int
    raise ValueError(f"Unknown eval mode: {mode}")


# ---------------------------------------------------------------------
# Common option action selector for videos
# ---------------------------------------------------------------------
@partial(jax.jit, static_argnums=(2,))
def select_option_action_video(q_opt_sel, key, option_policy_greedy: bool = True, option_policy_temp: float = 1.0):
    if option_policy_greedy:
        act = jnp.argmax(q_opt_sel, axis=-1)
        dist = jax.nn.one_hot(act, q_opt_sel.shape[-1], dtype=jnp.float32)
        prob = jnp.ones((q_opt_sel.shape[0],), dtype=jnp.float32)
    else:
        temp = jnp.maximum(jnp.float32(option_policy_temp), 1e-6)
        logits = q_opt_sel / temp
        dist = jax.nn.softmax(logits, axis=-1)
        act = jax.random.categorical(key, logits, axis=-1)
        prob = jnp.take_along_axis(dist, act[:, None], axis=-1).squeeze(-1)
        prob = jnp.clip(prob, 1e-8, 1.0)
    return act, prob, dist


# ---------------------------------------------------------------------
# Common generic video rollout
# ---------------------------------------------------------------------
def collect_video_rollout_generic(
    *,
    env,
    enc_state,
    use_enc_bn: bool,
    use_q_bn: bool,
    n_steps: int = 50,
    has_options: bool = False,
    n_total_options: int = 0,
    opt_dur_table=None,
    forced_option_id: int | None = None,
    forced_option_duration: int = 50,
    meta_q_state=None,
    meta_q_ext_state=None,
    meta_q_int_state=None,
    meta_int_coef: float = 0.0,
    option_enc_state=None,
    opt_q_states=None,
    opt_q_ext_states=None,
    opt_q_int_states=None,
    option_int_coef: float = 0.0,
    use_value_rescale: bool = False,
    value_rescale_fn=None,
    inv_value_rescale_fn=None,
    meta_eval_mode: str = "ext",
    option_eval_mode: str = "ext",
    option_policy_greedy: bool = True,
    option_policy_temp: float = 1.0,
):
    init_obs, env_state = env.reset()

    if int(init_obs.shape[0]) != 1:
        raise ValueError(
            f"Video rollout expects num_envs=1, got batch={init_obs.shape[0]}."
        )

    if forced_option_id is not None and not has_options:
        raise ValueError("forced_option_id was given but has_options=False.")

    obs = init_obs
    key = jax.random.PRNGKey(0)

    frames = [chw_frame_to_hwc_uint8(obs[0])]
    actions = []
    meta_actions = []
    rewards = []
    dones = []
    option_ids = []

    last_meta = jnp.array([-1], dtype=jnp.int32)
    last_term = jnp.array([True])
    opt_count = jnp.array([0], dtype=jnp.int32)

    for _ in range(int(n_steps)):
        q_meta_ext, q_meta_int, q_meta_sel = get_q_values_generic_controller(
            obs,
            enc_state,
            use_enc_bn,
            use_q_bn,
            q_state=meta_q_state,
            q_ext_state=meta_q_ext_state,
            q_int_state=meta_q_int_state,
            int_coef=meta_int_coef,
            use_value_rescale=use_value_rescale,
            value_rescale_fn=value_rescale_fn,
            inv_value_rescale_fn=inv_value_rescale_fn,
        )

        q_meta_eval = pick_eval_head(q_meta_ext, q_meta_int, q_meta_sel, meta_eval_mode)
        A_meta = int(q_meta_eval.shape[-1])

        if has_options:
            A = A_meta - int(n_total_options)
            O = int(n_total_options)
            if A <= 0:
                raise ValueError(f"Invalid action split: A_meta={A_meta}, n_total_options={n_total_options}")
        else:
            A = A_meta
            O = 0

        if forced_option_id is not None:
            curr_meta = jnp.array([A + int(forced_option_id)], dtype=jnp.int32)
        elif has_options:
            greedy_meta = jnp.argmax(q_meta_eval, axis=-1).astype(jnp.int32)
            curr_meta = jnp.where(last_term, greedy_meta, last_meta)
        else:
            curr_meta = jnp.argmax(q_meta_eval, axis=-1).astype(jnp.int32)

        if has_options:
            is_prim = curr_meta < A
            sel_opt = jnp.clip(curr_meta - A, 0, O - 1)

            q_opt_ext, q_opt_int, q_opt_sel = compute_option_qs_generic(
                obs,
                option_enc_state,
                use_enc_bn,
                use_q_bn,
                opt_q_states=opt_q_states,
                opt_q_ext_states=opt_q_ext_states,
                opt_q_int_states=opt_q_int_states,
                int_coef=option_int_coef,
                use_value_rescale=use_value_rescale,
                value_rescale_fn=value_rescale_fn,
                inv_value_rescale_fn=inv_value_rescale_fn,
            )

            q_opt_eval = pick_eval_head(q_opt_ext, q_opt_int, q_opt_sel, option_eval_mode)
            q_opt_curr = q_opt_eval[sel_opt, jnp.arange(1), :]

            key, k_opt = jax.random.split(key)
            opt_act, _, _ = select_option_action_video(
                q_opt_curr,
                k_opt,
                option_policy_greedy=option_policy_greedy,
                option_policy_temp=option_policy_temp,
            )
            opt_act = opt_act.astype(jnp.int32)
            final_act = jnp.where(is_prim, curr_meta, opt_act)
        else:
            is_prim = jnp.array([True])
            sel_opt = jnp.array([-1], dtype=jnp.int32)
            final_act = curr_meta

        next_obs, env_state, reward, done, info = env.step(env_state, final_act)

        actions.append(int(final_act[0]))
        meta_actions.append(int(curr_meta[0]))
        rewards.append(float(reward[0]))
        dones.append(bool(done[0]))
        option_ids.append(int(sel_opt[0]) if (has_options and not bool(is_prim[0])) else -1)
        frames.append(chw_frame_to_hwc_uint8(next_obs[0]))

        if forced_option_id is not None:
            opt_count_new = jnp.where(last_term, 1, opt_count + 1)
            curr_term = (opt_count_new >= int(forced_option_duration)) | done
            next_opt_count = jnp.where(curr_term, 0, opt_count_new)
        elif has_options:
            opt_count_new = jnp.where(last_term, 1, opt_count + 1)
            if opt_dur_table is None:
                option_dur = jnp.full((1,), 1, dtype=jnp.int32)
            else:
                option_dur = jnp.asarray(opt_dur_table)[jnp.clip(sel_opt, 0, O - 1)]
            curr_term = is_prim | (opt_count_new >= option_dur) | done
            next_opt_count = jnp.where(curr_term, 0, opt_count_new)
        else:
            curr_term = jnp.ones_like(done, dtype=bool) | done
            next_opt_count = jnp.zeros_like(opt_count)

        obs = next_obs
        last_meta = curr_meta
        last_term = curr_term
        opt_count = next_opt_count

        if bool(done[0]):
            break

    return {
        "frames": frames,
        "actions": np.asarray(actions, dtype=np.int32),
        "meta_actions": np.asarray(meta_actions, dtype=np.int32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=bool),
        "option_ids": np.asarray(option_ids, dtype=np.int32),
    }


def save_video_generic(frames, path: str, fps: int = 10):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)
    return str(path)


def record_policy_and_option_videos_generic(
    *,
    out_dir: str,
    env,
    enc_state,
    use_enc_bn: bool,
    use_q_bn: bool,
    n_steps_policy: int = 200,
    n_steps_option: int = 50,
    fps: int = 10,
    prefix: str = "run",
    has_options: bool = False,
    n_total_options: int = 0,
    opt_dur_table=None,
    option_ids=None,
    meta_q_state=None,
    meta_q_ext_state=None,
    meta_q_int_state=None,
    meta_int_coef: float = 0.0,
    option_enc_state=None,
    opt_q_states=None,
    opt_q_ext_states=None,
    opt_q_int_states=None,
    option_int_coef: float = 0.0,
    use_value_rescale: bool = False,
    value_rescale_fn=None,
    inv_value_rescale_fn=None,
    meta_eval_mode: str = "ext",
    option_eval_mode: str = "ext",
    option_policy_greedy: bool = True,
    option_policy_temp: float = 1.0,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = {"greedy": None, "options": {}}

    greedy_rollout = collect_video_rollout_generic(
        env=env,
        enc_state=enc_state,
        use_enc_bn=use_enc_bn,
        use_q_bn=use_q_bn,
        n_steps=n_steps_policy,
        has_options=has_options,
        n_total_options=n_total_options,
        opt_dur_table=opt_dur_table,
        forced_option_id=None,
        forced_option_duration=n_steps_option,
        meta_q_state=meta_q_state,
        meta_q_ext_state=meta_q_ext_state,
        meta_q_int_state=meta_q_int_state,
        meta_int_coef=meta_int_coef,
        option_enc_state=option_enc_state,
        opt_q_states=opt_q_states,
        opt_q_ext_states=opt_q_ext_states,
        opt_q_int_states=opt_q_int_states,
        option_int_coef=option_int_coef,
        use_value_rescale=use_value_rescale,
        value_rescale_fn=value_rescale_fn,
        inv_value_rescale_fn=inv_value_rescale_fn,
        meta_eval_mode=meta_eval_mode,
        option_eval_mode=option_eval_mode,
        option_policy_greedy=option_policy_greedy,
        option_policy_temp=option_policy_temp,
    )

    greedy_path = out_dir / f"{prefix}_greedy.mp4"
    outputs["greedy"] = save_video_generic(greedy_rollout["frames"], greedy_path, fps=fps)

    if has_options:
        if option_ids is None:
            option_ids = list(range(int(n_total_options)))

        for oid in option_ids:
            opt_rollout = collect_video_rollout_generic(
                env=env,
                enc_state=enc_state,
                use_enc_bn=use_enc_bn,
                use_q_bn=use_q_bn,
                n_steps=n_steps_option,
                has_options=True,
                n_total_options=n_total_options,
                opt_dur_table=opt_dur_table,
                forced_option_id=int(oid),
                forced_option_duration=n_steps_option,
                meta_q_state=meta_q_state,
                meta_q_ext_state=meta_q_ext_state,
                meta_q_int_state=meta_q_int_state,
                meta_int_coef=meta_int_coef,
                option_enc_state=option_enc_state,
                opt_q_states=opt_q_states,
                opt_q_ext_states=opt_q_ext_states,
                opt_q_int_states=opt_q_int_states,
                option_int_coef=option_int_coef,
                use_value_rescale=use_value_rescale,
                value_rescale_fn=value_rescale_fn,
                inv_value_rescale_fn=inv_value_rescale_fn,
                meta_eval_mode=meta_eval_mode,
                option_eval_mode=option_eval_mode,
                option_policy_greedy=option_policy_greedy,
                option_policy_temp=option_policy_temp,
            )
            opt_path = out_dir / f"{prefix}_option_{int(oid)}.mp4"
            outputs["options"][int(oid)] = save_video_generic(opt_rollout["frames"], opt_path, fps=fps)

    return outputs


# ---------------------------------------------------------------------
# Common WandB helper
# ---------------------------------------------------------------------
def make_wandb_log_callback(args, start_time):
    def _log_to_wandb(m):
        import wandb
        elapsed = time.time() - start_time
        total_env = m["update_steps"] * args.env.n_steps * args.env.n_envs
        m["SPS"] = total_env / max(elapsed, 1e-6)
        wandb.log(m)

    @jax.jit
    def _pass_and_log(m):
        return jax.debug.callback(_log_to_wandb, m)

    return _pass_and_log