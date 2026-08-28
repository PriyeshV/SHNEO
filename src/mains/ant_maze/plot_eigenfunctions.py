"""Render the learned source eigenfunctions of a SAC_NEO AntMaze run.

The vector Laplacian in ``NEO_LapXY.py`` is an MLP over torso ``(x, y)``, so
its eigenfunctions can be evaluated on a dense grid and drawn directly over the
maze layout.  That answers the question a success curve cannot: whether the
representation the options are built on has any maze structure at all.

A healthy phi_m is smooth inside corridors and changes sign across the maze
rather than across a wall.  Noise, a constant field, or structure that ignores
the walls all mean the Laplacian never converged, and no amount of option
tuning will help.

Usage::

    python src/mains/ant_maze/plot_eigenfunctions.py \\
        --checkpoint runs/sac_neo_explore_10000768.pkl \\
        --maze u_maze --out eigenfunctions.png

The checkpoint is the ``*_explore_*.pkl`` file written when a NEO run is given
``--a_checkpoint_logdir``.
"""

import argparse
import glob
import os
import re
import sys

sys.path.append(os.getcwd())

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from flax.serialization import msgpack_restore  # noqa: E402

from src.mains.ant_maze.pixel_ant_maze import ANT_MAZE_LAYOUTS  # noqa: E402


def _wall_mask(layout, scale):
    """Boolean wall grid plus the world extent of the layout."""

    cells = np.asarray(layout, dtype=object)
    wall = (cells == 1)
    rows, columns = wall.shape
    # JaxGCRL places cell (i, j) at (i * scale, j * scale); confirmed against a
    # live env, where the u_maze reset cell (1, 1) reports obs[:2] == [4, 4].
    extent = (
        -0.5 * scale, (rows - 0.5) * scale,      # x
        -0.5 * scale, (columns - 0.5) * scale,   # y
    )
    return wall, extent


def _grid(extent, resolution):
    xs = np.linspace(extent[0], extent[1], resolution, dtype=np.float32)
    ys = np.linspace(extent[2], extent[3], resolution, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return gx, gy, np.stack([gx.ravel(), gy.ravel()], axis=-1)


def _is_wall(points, wall, scale):
    rows, columns = wall.shape
    i = np.clip(np.rint(points[:, 0] / scale).astype(int), 0, rows - 1)
    j = np.clip(np.rint(points[:, 1] / scale).astype(int), 0, columns - 1)
    return wall[i, j]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                    help="*_explore_*.pkl file, or a checkpoint directory")
    parser.add_argument("--maze", default="u_maze", choices=tuple(ANT_MAZE_LAYOUTS))
    parser.add_argument("--scale", type=float, default=4.0, help="maze_size_scaling")
    parser.add_argument("--n_show", type=int, default=5, help="eigenfunctions to draw")
    parser.add_argument("--resolution", type=int, default=220)
    parser.add_argument("--out", default="eigenfunctions.png")
    args = parser.parse_args()

    # A directory is accepted so callers need not know the final step number,
    # which moves with a_n_evals.
    checkpoint = args.checkpoint
    if os.path.isdir(checkpoint):
        candidates = glob.glob(os.path.join(checkpoint, "*explore*.pkl"))
        if not candidates:
            raise SystemExit(f"no *explore*.pkl in {checkpoint!r}")
        checkpoint = max(candidates, key=lambda p: int(
            re.search(r"(\d+)\.pkl$", os.path.basename(p)).group(1)))
        print(f"using {os.path.basename(checkpoint)}")

    with open(checkpoint, "rb") as handle:
        payload = msgpack_restore(handle.read())
    if "lap_params" not in payload:
        raise SystemExit(
            f"{checkpoint} has no 'lap_params'. Point this at the "
            "*_explore_*.pkl file, not the policy checkpoint."
        )
    lap_params = jax.tree_util.tree_map(jnp.asarray, payload["lap_params"])

    # Rebuild the encoder from the checkpoint rather than the config, so the
    # script stays correct if the run used non-default widths.
    dense = sorted(
        (name for name in lap_params if name.startswith("Dense_")),
        key=lambda name: int(name.split("_")[1]),
    )
    widths = [lap_params[name]["kernel"].shape[1] for name in dense]
    output_dim = widths[-1]
    print(f"encoder: 2 -> {' -> '.join(str(w) for w in widths)}")

    def apply(xy):
        h = xy
        for name in dense[:-1]:
            p = lap_params[name]
            h = jax.nn.relu(h @ p["kernel"] + p["bias"])
        p = lap_params[dense[-1]]
        return h @ p["kernel"] + p["bias"]

    layout = ANT_MAZE_LAYOUTS[args.maze]
    wall, extent = _wall_mask(layout, args.scale)
    gx, gy, points = _grid(extent, args.resolution)
    phi = np.asarray(jax.jit(apply)(jnp.asarray(points)))

    blocked = _is_wall(points, wall, args.scale)
    phi[blocked] = np.nan  # walls are unreachable, so leave them blank

    n_show = min(args.n_show, output_dim)
    fig, axes = plt.subplots(1, n_show, figsize=(3.1 * n_show, 3.4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for m in range(n_show):
        field = phi[:, m].reshape(gx.shape)
        limit = np.nanmax(np.abs(field))
        ax = axes[m]
        # Transposed so the image axes match the maze picture: world x runs up.
        ax.imshow(
            field.T,
            origin="lower",
            extent=(extent[0], extent[1], extent[2], extent[3]),
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        ax.contour(gx, gy, np.nan_to_num(field), levels=[0.0], colors="k", linewidths=0.8)
        ax.set_title(f"$\\phi_{{{m}}}$   |max| = {limit:.2f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"Source eigenfunctions over torso xy - {args.maze} - "
        f"{os.path.basename(checkpoint)}",
        fontsize=11,
    )
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")

    # A constant field is the classic failure: the barrier is satisfied by a
    # single direction and every eigenfunction collapses onto it.
    spread = np.nanstd(phi[:, :n_show], axis=0)
    print("per-eigenfunction std over the maze:", np.array2string(spread, precision=3))
    if np.all(spread < 1e-2):
        print("WARNING: all eigenfunctions are near-constant; the Laplacian did not converge")


if __name__ == "__main__":
    main()
