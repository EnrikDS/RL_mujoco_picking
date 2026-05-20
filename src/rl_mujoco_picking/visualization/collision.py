"""Viewer utilities for making MuJoCo collision geometry visible.

These helpers do not change collision behavior.  They only recolor geoms so we
can see which solids are actually participating in contacts while debugging.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class CollisionVisualizationSummary:
    """Counts reported after recoloring the model."""

    collidable_geoms: int
    noncollidable_geoms: int


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"


def _geom_body_name(model: mujoco.MjModel, geom_id: int) -> str:
    body_id = int(model.geom_bodyid[geom_id])
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"


def _is_collidable(model: mujoco.MjModel, geom_id: int) -> bool:
    """MuJoCo geoms collide only when both contype and conaffinity are non-zero."""

    return bool(model.geom_contype[geom_id] and model.geom_conaffinity[geom_id])


def _collision_rgba(body_name: str) -> np.ndarray:
    """Choose stable debug colors by body family."""

    if body_name.startswith("handd_"):
        return np.array((1.0, 0.24, 0.06, 0.58), dtype=float)
    if body_name.startswith("seal_"):
        return np.array((1.0, 0.85, 0.0, 0.70), dtype=float)
    if body_name.endswith("_proxy"):
        return np.array((0.0, 0.95, 1.0, 0.46), dtype=float)
    return np.array((0.0, 0.95, 0.28, 0.38), dtype=float)


def apply_collision_visualization(
    model: mujoco.MjModel,
    *,
    collision_only: bool = False,
    visual_alpha: float = 0.08,
) -> CollisionVisualizationSummary:
    """Tint collidable geoms so the physics solids can be inspected in the viewer."""

    collidable_count = 0
    noncollidable_count = 0
    for geom_id in range(model.ngeom):
        body_name = _geom_body_name(model, geom_id)
        if _is_collidable(model, geom_id):
            # Collision geoms become bright and semi-transparent.
            model.geom_rgba[geom_id] = _collision_rgba(body_name)
            collidable_count += 1
        else:
            # Visual-only geoms are hidden or faded so they do not mask collision solids.
            model.geom_rgba[geom_id][3] = 0.0 if collision_only else visual_alpha
            noncollidable_count += 1
    return CollisionVisualizationSummary(
        collidable_geoms=collidable_count,
        noncollidable_geoms=noncollidable_count,
    )


def describe_collidable_geoms(model: mujoco.MjModel) -> tuple[str, ...]:
    """Return text rows for every geom that can participate in contacts."""

    rows: list[str] = []
    for geom_id in range(model.ngeom):
        if not _is_collidable(model, geom_id):
            continue
        rows.append(f"{_geom_name(model, geom_id)} body={_geom_body_name(model, geom_id)}")
    return tuple(rows)
