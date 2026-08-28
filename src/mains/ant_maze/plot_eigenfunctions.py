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
from brax.io import model  # noqa: E402

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


def ascii_sign_maps(phi, wall_grid, n_show, rows, cols):
    """Print each eigenfunction's sign structure as text.

    The nodal line -- where phi changes sign -- is what matters for an
    eigen-option: a genuine maze Laplacian puts it across a corridor, so the
    two signs are regions the agent must travel between.  A function that
    ignores the walls is only a smooth function of position.
    """

    for m in range(n_show):
        field = phi[:, m].reshape(rows, cols)
        free = ~wall_grid
        positive = int(np.sum((field > 0) & free))
        total = int(np.sum(free))
        print(f"\nphi_{m}   positive over {positive / total:5.1%} of free space")
        for r in range(rows):
            line = "".join(
                "#" if wall_grid[r, c] else ("+" if field[r, c] > 0 else "-")
                for c in range(cols)
            )
            print("  " + line)


def cell_magnitudes(phi, points, wall, scale, n_show):
    """Mean |phi| per maze cell.

    A sign map cannot tell high-frequency structure from |phi| ~ 0, where the
    sign is numerical noise.  This does: the eigen-reward is
    |phi(s')| - |phi(s)|, so any cell whose magnitude is near zero contributes
    only noise to the option that is defined by that eigenfunction.
    """

    rows, columns = wall.shape
    i = np.clip(np.rint(points[:, 0] / scale).astype(int), 0, rows - 1)
    j = np.clip(np.rint(points[:, 1] / scale).astype(int), 0, columns - 1)
    magnitude = np.abs(phi)

    for m in range(n_show):
        print(f"
mean |phi_{m}| per maze cell (large x on top)")
        for r in range(rows - 1, -1, -1):
            cells = []
            for c in range(columns):
                if wall[r, c]:
                    cells.append("    ##")
                    continue
                selected = (i == r) & (j == c)
                value = np.nanmean(magnitude[selected, m]) if selected.any() else np.nan
                cells.append(f"{value:6.2f}")
            print("  " + "".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                    help="*_explore_*.pkl file, or a checkpoint directory")
    parser.add_argument("--maze", default="u_maze", choices=tuple(ANT_MAZE_LAYOUTS))
    parser.add_argument("--scale", type=float, default=4.0, help="maze_size_scaling")
    parser.add_argument("--n_show", type=int, default=5, help="eigenfunctions to draw")
    parser.add_argument("--resolution", type=int, default=220)
    parser.add_argument("--out", default="eigenfunctions.png")
    parser.add_argument("--ascii", action="store_true",
                        help="also print sign maps as text, no transfer needed")
    parser.add_argument("--cells", action="store_true",
                        help="print mean |phi| per maze cell")
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

    # Read it back the same way the trainer wrote it: brax.io.model uses
    # pickle, so a flax msgpack reader sees trailing bytes and raises.
    payload = model.load_params(checkpoint)
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
    # np.array, not np.asarray: a jax array converts to a read-only view and
    # the wall masking below writes into it.
    phi = np.array(jax.jit(apply)(jnp.asarray(points)))

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

    if args.ascii:
        # Character cells are about twice as tall as wide, so use twice as
        # many columns to keep the maze square on screen.
        rows, cols = 21, 42
        axis_x = np.linspace(extent[1], extent[0], rows, dtype=np.float32)
        axis_y = np.linspace(extent[2], extent[3], cols, dtype=np.float32)
        cx, cy = np.meshgrid(axis_x, axis_y, indexing="ij")
        coarse = np.stack([cx.ravel(), cy.ravel()], axis=-1)
        coarse_phi = np.array(jax.jit(apply)(jnp.asarray(coarse)))
        coarse_wall = _is_wall(coarse, wall, args.scale).reshape(rows, cols)
        ascii_sign_maps(coarse_phi, coarse_wall, n_show, rows, cols)

    if args.cells:
        cell_magnitudes(phi, points, wall, args.scale, n_show)


if __name__ == "__main__":
    main()
