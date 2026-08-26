"""JIT-compatible top-down pixel observations for JaxGCRL AntMaze.

This wrapper deliberately renders from the Brax pipeline state with JAX
operations.  It therefore remains inside ``jit``/``vmap``/``scan`` and does
not perform a host callback to MuJoCo's OpenGL renderer.

The default ``observation_mode="state"`` preserves JaxGCRL's vector
observation for the SAC actor and critics and adds an ``H x W x 3`` float32
image to ``state.info["pixels"]``.  This is the intended mode for SAC-RND:
the control problem remains identical to the vector baaseline while only the
novelty model changes its input.

``observation_mode="pixels"`` is provided for a future pixel-control agent.
It replaces ``state.obs`` with the image, so it is not compatible with
Brax's stock MLP SAC networks.
"""

from __future__ import annotations

from typing import Final, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from brax import envs


_RESET: Final = "r"
_GOAL: Final = "g"

# These are the layouts in jaxgcrl.envs.ant_maze.  Keeping the small static
# arrays here makes rendering independent of JaxGCRL's package initializer,
# which imports legacy brax.v1 agents in some releases.
ANT_MAZE_LAYOUTS: Mapping[str, Sequence[Sequence[object]]] = {
    "u_maze": (
        (1, 1, 1, 1, 1),
        (1, _RESET, _GOAL, _GOAL, 1),
        (1, 1, 1, _GOAL, 1),
        (1, _GOAL, _GOAL, _GOAL, 1),
        (1, 1, 1, 1, 1),
    ),
    "big_maze": (
        (1, 1, 1, 1, 1, 1, 1, 1),
        (1, _RESET, _GOAL, 1, 1, _GOAL, _GOAL, 1),
        (1, _GOAL, _GOAL, 1, _GOAL, _GOAL, _GOAL, 1),
        (1, 1, _GOAL, _GOAL, _GOAL, 1, 1, 1),
        (1, _GOAL, _GOAL, 1, _GOAL, _GOAL, _GOAL, 1),
        (1, _GOAL, 1, _GOAL, _GOAL, 1, _GOAL, 1),
        (1, _GOAL, _GOAL, _GOAL, 1, _GOAL, _GOAL, 1),
        (1, 1, 1, 1, 1, 1, 1, 1),
    ),
    "hardest_maze": (
        (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        (1, _RESET, _GOAL, _GOAL, _GOAL, 1, _GOAL, _GOAL, _GOAL, _GOAL, _GOAL, 1),
        (1, _GOAL, 1, 1, _GOAL, 1, _GOAL, 1, _GOAL, 1, _GOAL, 1),
        (1, _GOAL, _GOAL, _GOAL, _GOAL, _GOAL, _GOAL, 1, _GOAL, _GOAL, _GOAL, 1),
        (1, _GOAL, 1, 1, 1, 1, _GOAL, 1, 1, 1, _GOAL, 1),
        (1, _GOAL, _GOAL, 1, _GOAL, 1, _GOAL, _GOAL, _GOAL, _GOAL, _GOAL, 1),
        (1, 1, _GOAL, 1, _GOAL, 1, _GOAL, 1, _GOAL, 1, 1, 1),
        (1, _GOAL, _GOAL, 1, _GOAL, _GOAL, _GOAL, 1, _GOAL, _GOAL, _GOAL, 1),
        (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    ),
}


def _static_maze_image(
    layout: Sequence[Sequence[object]], height: int, width: int
) -> np.ndarray:
    """Creates the unchanging floor/wall portion of the RGB image."""

    cells = np.asarray(layout, dtype=object)
    wall = cells == 1
    rows, columns = wall.shape

    # Pixel centres are mapped to maze cells.  Image y is reversed so larger
    # JaxGCRL x coordinates appear toward the top of the displayed frame.
    row_index = np.floor(
        (np.arange(height, dtype=np.float32) + 0.5) * rows / height
    ).astype(np.int32)
    column_index = np.floor(
        (np.arange(width, dtype=np.float32) + 0.5) * columns / width
    ).astype(np.int32)
    row_index = (rows - 1) - np.clip(row_index, 0, rows - 1)
    column_index = np.clip(column_index, 0, columns - 1)
    wall_pixels = wall[row_index[:, None], column_index[None, :]]

    image = np.empty((height, width, 3), dtype=np.float32)
    image[...] = np.asarray((0.08, 0.10, 0.12), dtype=np.float32)
    image[wall_pixels] = np.asarray((0.55, 0.35, 0.18), dtype=np.float32)
    return image


class PixelAntMazeWrapper(envs.Wrapper):
    """Adds a semantic top-down RGB rendering to a JaxGCRL AntMaze state.

    The frame contains the maze walls, the desired goal, and every dynamic
    Ant body position.  Rendering body positions (rather than only the torso)
    preserves locomotion/pose novelty that a single agent dot would discard.
    """

    # ``ensure_pixel_ant_maze`` walks an environment chain looking for this
    # marker before adding another renderer.  ``brax.envs.Wrapper.__getattr__``
    # forwards unknown attributes to the wrapped environment, so outer wrappers
    # report it too and the idempotency check sees the whole chain.
    is_pixel_ant_maze_wrapper: bool = True

    def __init__(
        self,
        env,
        *,
        maze_layout_name: str,
        height: int = 84,
        width: int | None = None,
        maze_size_scaling: float = 4.0,
        observation_mode: str = "state",
        info_key: str = "pixels",
    ):
        super().__init__(env)
        if maze_layout_name not in ANT_MAZE_LAYOUTS:
            raise ValueError(
                f"Unknown AntMaze layout {maze_layout_name!r}; choose one of "
                f"{tuple(ANT_MAZE_LAYOUTS)}"
            )
        if height <= 0 or (width is not None and width <= 0):
            raise ValueError("pixel height and width must be positive")
        if maze_size_scaling <= 0:
            raise ValueError("maze_size_scaling must be positive")
        if observation_mode not in {"state", "pixels"}:
            raise ValueError("observation_mode must be 'state' or 'pixels'")

        self._layout_name = maze_layout_name
        self._layout = ANT_MAZE_LAYOUTS[maze_layout_name]
        self._height = int(height)
        self._width = int(width if width is not None else height)
        self._scale = float(maze_size_scaling)
        self._observation_mode = observation_mode
        self._info_key = info_key
        self._background = jnp.asarray(
            _static_maze_image(self._layout, self._height, self._width)
        )

        rows = len(self._layout)
        columns = len(self._layout[0])
        # World coordinates of pixel centres.  JaxGCRL puts cell (i, j) at
        # (i * scale, j * scale), with a half-cell border around the layout.
        x = (
            rows
            - (jnp.arange(self._height, dtype=jnp.float32) + 0.5)
            * rows
            / self._height
            - 0.5
        ) * self._scale
        y = (
            (jnp.arange(self._width, dtype=jnp.float32) + 0.5)
            * columns
            / self._width
            - 0.5
        ) * self._scale
        self._pixel_x, self._pixel_y = jnp.meshgrid(x, y, indexing="ij")

        # At least one visible pixel on the largest layout, while remaining
        # small enough for individual limb positions to be distinguishable.
        world_per_pixel = max(
            rows * self._scale / self._height,
            columns * self._scale / self._width,
        )
        self._body_radius = float(max(0.28, 0.80 * world_per_pixel))
        self._goal_radius = float(max(0.50, 1.20 * world_per_pixel))

    @property
    def pixel_shape(self) -> tuple[int, int, int]:
        return (self._height, self._width, 3)

    def _disk_mask(self, positions: jax.Array, radius: float) -> jax.Array:
        dx = self._pixel_x[..., None] - positions[..., 0]
        dy = self._pixel_y[..., None] - positions[..., 1]
        return jnp.any(dx * dx + dy * dy <= radius * radius, axis=-1)

    def render_pixels(self, state) -> jax.Array:
        """Renders one unbatched Brax state as float32 RGB in ``[0, 1]``."""

        positions = state.pipeline_state.x.pos
        if positions.shape[-2] < 2:
            raise ValueError("AntMaze pipeline state must include Ant and target links")

        # In JaxGCRL AntMaze the final link is the mocap-like target and all
        # preceding dynamic link transforms belong to the Ant.
        ant_xy = positions[:-1, :2]
        goal_xy = positions[-1:, :2]
        ant_mask = self._disk_mask(ant_xy, self._body_radius)
        goal_mask = self._disk_mask(goal_xy, self._goal_radius)

        image = self._background
        image = jnp.where(
            goal_mask[..., None],
            jnp.asarray((0.12, 0.82, 0.25), dtype=jnp.float32),
            image,
        )
        image = jnp.where(
            ant_mask[..., None],
            jnp.asarray((0.12, 0.42, 1.00), dtype=jnp.float32),
            image,
        )
        return image

    def _add_pixels(self, state):
        pixels = self.render_pixels(state)
        info = {**state.info, self._info_key: pixels}
        if self._observation_mode == "pixels":
            return state.replace(obs=pixels, info=info)
        return state.replace(info=info)

    def reset(self, rng: jax.Array):
        state = self._add_pixels(self.env.reset(rng))
        # Brax AutoResetWrapper resets pipeline_state and obs, but leaves
        # arbitrary info fields untouched on a terminal step.  Retaining the
        # reset frame lets the learner select the pixel observation matching
        # AutoResetWrapper's returned next_observation.
        return state.replace(
            info={**state.info, f"_{self._info_key}_reset": state.info[self._info_key]}
        )

    def step(self, state, action: jax.Array):
        return self._add_pixels(self.env.step(state, action))


__all__ = ["ANT_MAZE_LAYOUTS", "PixelAntMazeWrapper"]
