from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import mujoco
import numpy as np


DEFAULT_CUP_TIP_SITE = "handd_tool_seal_tip_site"
DEFAULT_CUP_UNCOMPRESSED_SITE = "handd_tool_seal_axis_site"


@dataclass(frozen=True)
class SuctionCupParameters:
    tip_site_name: str = DEFAULT_CUP_TIP_SITE
    uncompressed_site_name: str = DEFAULT_CUP_UNCOMPRESSED_SITE
    seal_radius: float = 0.026
    lip_tolerance: float = 0.002
    compliance: float = 0.003
    max_gap: float = 0.003
    min_alignment: float = 0.94
    radial_margin: float = 0.0015


@dataclass(frozen=True)
class SealEvaluation:
    geom_name: str
    geom_id: int
    geom_type: str
    sealable: bool
    gap: float
    axial_offset: float
    radial_offset: float
    normal_alignment: float
    nearest_point: tuple[float, float, float]
    surface_normal: tuple[float, float, float]
    cup_center: tuple[float, float, float]
    cup_axis: tuple[float, float, float]
    reason: str


def _site_id(model: mujoco.MjModel, site_name: str) -> int:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"Site not found: {site_name}")
    return site_id


def _geom_id(model: mujoco.MjModel, geom_name_or_id: str | int) -> int:
    if isinstance(geom_name_or_id, int):
        return geom_name_or_id
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name_or_id)
    if geom_id < 0:
        raise ValueError(f"Geom not found: {geom_name_or_id}")
    return geom_id


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def suction_axis(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    parameters: SuctionCupParameters,
) -> tuple[np.ndarray, np.ndarray]:
    tip_site_id = _site_id(model, parameters.tip_site_name)
    uncompressed_site_id = _site_id(model, parameters.uncompressed_site_name)
    tip_pos = np.array(data.site_xpos[tip_site_id], dtype=float)
    uncompressed_pos = np.array(data.site_xpos[uncompressed_site_id], dtype=float)
    axis = _normalize(uncompressed_pos - tip_pos)
    return tip_pos, axis


def _geom_type_name(model: mujoco.MjModel, geom_id: int) -> str:
    geom_type = model.geom_type[geom_id]
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return "sphere"
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return "box"
    return f"unsupported:{int(geom_type)}"


def _closest_point_on_sphere(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
    query_point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    center = np.array(data.geom_xpos[geom_id], dtype=float)
    radius = float(model.geom_size[geom_id][0])
    direction = query_point - center
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        direction = np.array((1.0, 0.0, 0.0), dtype=float)
    else:
        direction /= norm
    nearest_point = center + radius * direction
    return nearest_point, direction


def _closest_point_on_box(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
    query_point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    center = np.array(data.geom_xpos[geom_id], dtype=float)
    rotation = np.array(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    half_size = np.array(model.geom_size[geom_id][:3], dtype=float)
    local_point = rotation.T @ (query_point - center)
    clamped_local = np.clip(local_point, -half_size, half_size)
    nearest_point = center + rotation @ clamped_local

    outside_delta = local_point - clamped_local
    if np.linalg.norm(outside_delta) > 1e-8:
        normal_local = _normalize(outside_delta)
    else:
        face_clearance = half_size - np.abs(local_point)
        axis = int(np.argmin(face_clearance))
        normal_local = np.zeros(3, dtype=float)
        normal_local[axis] = 1.0 if local_point[axis] >= 0.0 else -1.0

    surface_normal = _normalize(rotation @ normal_local)
    return nearest_point, surface_normal


def closest_point_on_geom(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
    query_point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    geom_type = model.geom_type[geom_id]
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return _closest_point_on_sphere(model, data, geom_id, query_point)
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return _closest_point_on_box(model, data, geom_id, query_point)
    raise NotImplementedError(
        "Suction seal evaluation currently supports sphere and box geoms only."
    )


def evaluate_suction_seal(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_name_or_id: str | int,
    parameters: SuctionCupParameters | None = None,
) -> SealEvaluation:
    params = parameters or SuctionCupParameters()
    geom_id = _geom_id(model, geom_name_or_id)
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
    geom_type_name = _geom_type_name(model, geom_id)

    if geom_type_name.startswith("unsupported:"):
        return SealEvaluation(
            geom_name=geom_name,
            geom_id=geom_id,
            geom_type=geom_type_name,
            sealable=False,
            gap=float("inf"),
            axial_offset=float("inf"),
            radial_offset=float("inf"),
            normal_alignment=-1.0,
            nearest_point=(0.0, 0.0, 0.0),
            surface_normal=(0.0, 0.0, 0.0),
            cup_center=(0.0, 0.0, 0.0),
            cup_axis=(0.0, 0.0, 0.0),
            reason="unsupported_geom_type",
        )

    cup_center, cup_axis = suction_axis(model, data, params)
    nearest_point, surface_normal = closest_point_on_geom(model, data, geom_id, cup_center)

    offset = nearest_point - cup_center
    axial_offset = float(np.dot(offset, cup_axis))
    radial_vector = offset - axial_offset * cup_axis
    radial_offset = float(np.linalg.norm(radial_vector))
    normal_alignment = float(np.dot(surface_normal, cup_axis))
    gap = abs(axial_offset)

    if radial_offset > params.seal_radius - params.radial_margin:
        reason = "outside_cup_radius"
        sealable = False
    elif gap > params.max_gap + params.compliance + params.lip_tolerance:
        reason = "too_far_from_surface"
        sealable = False
    elif normal_alignment < params.min_alignment:
        reason = "surface_misaligned"
        sealable = False
    else:
        reason = "seal_candidate"
        sealable = True

    return SealEvaluation(
        geom_name=geom_name,
        geom_id=geom_id,
        geom_type=geom_type_name,
        sealable=sealable,
        gap=gap,
        axial_offset=axial_offset,
        radial_offset=radial_offset,
        normal_alignment=normal_alignment,
        nearest_point=tuple(float(value) for value in nearest_point),
        surface_normal=tuple(float(value) for value in surface_normal),
        cup_center=tuple(float(value) for value in cup_center),
        cup_axis=tuple(float(value) for value in cup_axis),
        reason=reason,
    )


def find_best_seal_candidate(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_names_or_ids: Iterable[str | int],
    parameters: SuctionCupParameters | None = None,
) -> SealEvaluation | None:
    best: SealEvaluation | None = None
    for geom_name_or_id in geom_names_or_ids:
        evaluation = evaluate_suction_seal(model, data, geom_name_or_id, parameters)
        if not evaluation.sealable:
            continue
        if best is None:
            best = evaluation
            continue
        if evaluation.gap < best.gap:
            best = evaluation
            continue
        if evaluation.gap == best.gap and evaluation.radial_offset < best.radial_offset:
            best = evaluation
    return best
