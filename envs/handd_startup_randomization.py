from __future__ import annotations

import math
import secrets
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np


STORAGE_TOTE_PROXY_BODY = "storage_tote_proxy"
STORAGE_TOTE_FLOOR_PROXY_GEOM = "storage_tote_floor_proxy"
STORAGE_TOTE_RIM_SITE = "storage_tote_rim_site"
DEFAULT_DROP_HEIGHT = 0.25
DEFAULT_LAYER_SPACING = 0.1
DEFAULT_SETTLE_STEPS = 2400
DEFAULT_DENSITY = 325.0
DEFAULT_SPAWN_JITTER_SCALE = 0.75
DEFAULT_MIN_QUIET_STEPS = 120
DEFAULT_MIN_SETTLE_STEPS = 240
DEFAULT_VELOCITY_EPS = 0.025
DEFAULT_PARKING_POS = np.array((3.5, -3.5, -4.0), dtype=float)
OBJECT_COLORS = (
    (0.89, 0.27, 0.22, 1.0),
    (0.20, 0.64, 0.92, 1.0),
    (0.25, 0.74, 0.47, 1.0),
    (0.97, 0.69, 0.19, 1.0),
    (0.90, 0.46, 0.68, 1.0),
)
OBJECT_SHAPES = ("sphere", "cube", "rect")


@dataclass(frozen=True)
class ToteBounds:
    center: np.ndarray
    inner_half_x: float
    inner_half_y: float
    floor_top_z: float
    rim_z: float

    @property
    def min_x(self) -> float:
        return float(self.center[0] - self.inner_half_x)

    @property
    def max_x(self) -> float:
        return float(self.center[0] + self.inner_half_x)

    @property
    def min_y(self) -> float:
        return float(self.center[1] - self.inner_half_y)

    @property
    def max_y(self) -> float:
        return float(self.center[1] + self.inner_half_y)


@dataclass(frozen=True)
class StartupDropConfig:
    num_objects: int = 10
    min_size: float = 0.03
    max_size: float = 0.08
    seed: int | None = None
    drop_height: float = DEFAULT_DROP_HEIGHT
    layer_spacing: float = DEFAULT_LAYER_SPACING
    settle_steps: int = DEFAULT_SETTLE_STEPS
    spawn_jitter_scale: float = DEFAULT_SPAWN_JITTER_SCALE
    density: float = DEFAULT_DENSITY
    min_quiet_steps: int = DEFAULT_MIN_QUIET_STEPS
    min_settle_steps: int = DEFAULT_MIN_SETTLE_STEPS
    velocity_epsilon: float = DEFAULT_VELOCITY_EPS


@dataclass(frozen=True)
class GroceryObjectSpec:
    name: str
    joint_name: str
    geom_name: str
    shape: str
    spawn_pos: tuple[float, float, float]
    spawn_quat: tuple[float, float, float, float]
    size: tuple[float, float, float]
    color: tuple[float, float, float, float]
    density: float


@dataclass
class RuntimeStartupScene:
    base_scene_path: Path
    temp_scene_path: Path
    specs: list[GroceryObjectSpec]
    tote_bounds: ToteBounds
    seed: int
    config: StartupDropConfig

    def cleanup(self) -> None:
        if self.temp_scene_path.exists():
            self.temp_scene_path.unlink()


@dataclass(frozen=True)
class ObjectPlacement:
    spec: GroceryObjectSpec
    valid: bool
    final_pos: tuple[float, float, float]
    final_quat: tuple[float, float, float, float]
    aabb_min: tuple[float, float, float]
    aabb_max: tuple[float, float, float]


@dataclass(frozen=True)
class StartupDropResult:
    seed: int
    settle_steps_run: int
    kept_objects: list[ObjectPlacement]
    ignored_objects: list[ObjectPlacement]


def validate_config(config: StartupDropConfig) -> None:
    if config.num_objects <= 0:
        raise ValueError("num_objects must be positive.")
    if config.min_size <= 0 or config.max_size <= 0:
        raise ValueError("min_size and max_size must be positive.")
    if config.min_size > config.max_size:
        raise ValueError("min_size cannot be larger than max_size.")
    if config.drop_height <= 0:
        raise ValueError("drop_height must be positive.")
    if config.layer_spacing <= 0:
        raise ValueError("layer_spacing must be positive.")
    if config.settle_steps <= 0:
        raise ValueError("settle_steps must be positive.")


def _parse_vec(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split())


def default_tote_bounds(scene_path: Path) -> ToteBounds:
    tree, root = _clone_scene(scene_path)
    proxy_body = None
    for body in root.iter("body"):
        if body.attrib.get("name") == STORAGE_TOTE_PROXY_BODY:
            proxy_body = body
            break
    if proxy_body is None:
        raise ValueError(f"Scene is missing body '{STORAGE_TOTE_PROXY_BODY}'.")

    body_pos = np.array(_parse_vec(proxy_body.attrib.get("pos", "0 0 0")), dtype=float)
    floor_geom = None
    rim_site = None
    for child in proxy_body:
        name = child.attrib.get("name")
        if child.tag == "geom" and name == STORAGE_TOTE_FLOOR_PROXY_GEOM:
            floor_geom = child
        if child.tag == "site" and name == STORAGE_TOTE_RIM_SITE:
            rim_site = child

    if floor_geom is None:
        raise ValueError(f"Scene is missing geom '{STORAGE_TOTE_FLOOR_PROXY_GEOM}'.")
    if rim_site is None:
        raise ValueError(f"Scene is missing site '{STORAGE_TOTE_RIM_SITE}'.")

    floor_pos = np.array(_parse_vec(floor_geom.attrib.get("pos", "0 0 0")), dtype=float)
    floor_size = np.array(_parse_vec(floor_geom.attrib["size"]), dtype=float)
    rim_pos = np.array(_parse_vec(rim_site.attrib["pos"]), dtype=float)

    return ToteBounds(
        center=body_pos.copy(),
        inner_half_x=float(floor_size[0]),
        inner_half_y=float(floor_size[1]),
        floor_top_z=float(body_pos[2] + floor_pos[2] + floor_size[2]),
        rim_z=float(body_pos[2] + rim_pos[2]),
    )


def _resolve_seed(seed: int | None) -> int:
    if seed is not None:
        return seed
    return secrets.randbelow(2**31 - 1)


def _random_unit_quaternion(rng: np.random.Generator) -> tuple[float, float, float, float]:
    u1, u2, u3 = rng.random(3)
    qx = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2)
    qy = math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
    qz = math.sqrt(u1) * math.sin(2.0 * math.pi * u3)
    qw = math.sqrt(u1) * math.cos(2.0 * math.pi * u3)
    return (qw, qx, qy, qz)


def _sample_size(
    shape: str,
    rng: np.random.Generator,
    min_size: float,
    max_size: float,
) -> tuple[float, float, float]:
    if shape == "sphere":
        radius = rng.uniform(min_size, max_size) / 2.0
        return (radius, radius, radius)
    if shape == "cube":
        half = rng.uniform(min_size, max_size) / 2.0
        return (half, half, half)
    if shape == "rect":
        return (
            rng.uniform(min_size, max_size) / 2.0,
            rng.uniform(min_size, max_size) / 2.0,
            rng.uniform(min_size, max_size) / 2.0,
        )
    raise ValueError(f"Unsupported grocery shape: {shape}")


def _build_object_specs(config: StartupDropConfig, tote_bounds: ToteBounds, seed: int) -> list[GroceryObjectSpec]:
    rng = np.random.default_rng(seed)
    spawn_half_x = tote_bounds.inner_half_x * config.spawn_jitter_scale
    spawn_half_y = tote_bounds.inner_half_y * config.spawn_jitter_scale
    specs: list[GroceryObjectSpec] = []

    for index in range(config.num_objects):
        shape = OBJECT_SHAPES[int(rng.integers(0, len(OBJECT_SHAPES)))]
        size = _sample_size(shape, rng, config.min_size, config.max_size)
        layer = index // 4
        spawn_pos = (
            float(tote_bounds.center[0] + rng.uniform(-spawn_half_x, spawn_half_x)),
            float(tote_bounds.center[1] + rng.uniform(-spawn_half_y, spawn_half_y)),
            float(tote_bounds.rim_z + config.drop_height + layer * config.layer_spacing),
        )
        specs.append(
            GroceryObjectSpec(
                name=f"startup_grocery_{index:03d}",
                joint_name=f"startup_grocery_{index:03d}_freejoint",
                geom_name=f"startup_grocery_{index:03d}_geom",
                shape=shape,
                spawn_pos=spawn_pos,
                spawn_quat=_random_unit_quaternion(rng),
                size=size,
                color=OBJECT_COLORS[index % len(OBJECT_COLORS)],
                density=config.density,
            )
        )

    return specs


def _format_vec(values: Iterable[float]) -> str:
    return " ".join(f"{value:.8f}" for value in values)


def _make_object_body(
    spec: GroceryObjectSpec,
    pos: tuple[float, float, float] | None = None,
    quat: tuple[float, float, float, float] | None = None,
) -> ET.Element:
    body = ET.Element(
        "body",
        attrib={
            "name": spec.name,
            "pos": _format_vec(spec.spawn_pos if pos is None else pos),
            "quat": _format_vec(spec.spawn_quat if quat is None else quat),
        },
    )
    ET.SubElement(body, "freejoint", attrib={"name": spec.joint_name})

    geom_attrs = {
        "name": spec.geom_name,
        "density": f"{spec.density:.2f}",
        "friction": "0.95 0.10 0.02",
        "condim": "4",
        "rgba": _format_vec(spec.color),
    }

    if spec.shape == "sphere":
        geom_attrs["type"] = "sphere"
        geom_attrs["size"] = f"{spec.size[0]:.8f}"
    else:
        geom_attrs["type"] = "box"
        geom_attrs["size"] = _format_vec(spec.size)

    ET.SubElement(body, "geom", attrib=geom_attrs)
    return body


def _clone_scene(scene_path: Path) -> tuple[ET.ElementTree, ET.Element]:
    tree = ET.parse(scene_path)
    return tree, tree.getroot()


def build_runtime_startup_scene(
    base_scene_path: Path,
    config: StartupDropConfig,
) -> RuntimeStartupScene:
    validate_config(config)
    seed = _resolve_seed(config.seed)
    tote_bounds = default_tote_bounds(base_scene_path)
    specs = _build_object_specs(config, tote_bounds, seed)
    tree, root = _clone_scene(base_scene_path)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Scene is missing a worldbody.")

    for spec in specs:
        worldbody.append(_make_object_body(spec))

    temp_dir = base_scene_path.parent
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".xml",
        prefix="startup_drop_",
        dir=temp_dir,
        delete=False,
        encoding="utf-8",
    )
    temp_path = Path(temp_file.name)
    temp_file.close()

    ET.indent(tree, space="  ")
    tree.write(temp_path, encoding="utf-8", xml_declaration=False)

    return RuntimeStartupScene(
        base_scene_path=base_scene_path,
        temp_scene_path=temp_path,
        specs=specs,
        tote_bounds=tote_bounds,
        seed=seed,
        config=config,
    )


def restore_spawn_state(model: mujoco.MjModel, data: mujoco.MjData, runtime_scene: RuntimeStartupScene) -> None:
    for spec in runtime_scene.specs:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, spec.joint_name)
        if joint_id < 0:
            raise ValueError(f"Startup joint not found: {spec.joint_name}")
        qpos_adr = model.jnt_qposadr[joint_id]
        qvel_adr = model.jnt_dofadr[joint_id]
        data.qpos[qpos_adr : qpos_adr + 3] = spec.spawn_pos
        data.qpos[qpos_adr + 3 : qpos_adr + 7] = spec.spawn_quat
        data.qvel[qvel_adr : qvel_adr + 6] = 0.0
    mujoco.mj_forward(model, data)


def _max_object_velocity(model: mujoco.MjModel, data: mujoco.MjData, specs: Iterable[GroceryObjectSpec]) -> float:
    peak = 0.0
    for spec in specs:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, spec.joint_name)
        qvel_adr = model.jnt_dofadr[joint_id]
        peak = max(peak, float(np.max(np.abs(data.qvel[qvel_adr : qvel_adr + 6]))))
    return peak


def _sleep_to_realtime(start_time: float, timestep: float) -> None:
    elapsed = time.perf_counter() - start_time
    remaining = timestep - elapsed
    if remaining > 0.0:
        time.sleep(remaining)


def _geom_aabb(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    geom_type = model.geom_type[geom_id]
    center = np.array(data.geom_xpos[geom_id], dtype=float)

    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        extents = np.full(3, float(model.geom_size[geom_id][0]), dtype=float)
    elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        rotation = np.array(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        half_size = np.array(model.geom_size[geom_id][:3], dtype=float)
        extents = np.abs(rotation) @ half_size
    else:
        raise ValueError(f"Unsupported startup geom type: {int(geom_type)}")

    return center - extents, center + extents


def _is_inside_tote(
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    tote_bounds: ToteBounds,
    margin: float = 0.002,
) -> bool:
    return (
        aabb_min[0] >= tote_bounds.min_x + margin
        and aabb_max[0] <= tote_bounds.max_x - margin
        and aabb_min[1] >= tote_bounds.min_y + margin
        and aabb_max[1] <= tote_bounds.max_y - margin
        and aabb_min[2] >= tote_bounds.floor_top_z - margin
        and aabb_max[2] <= tote_bounds.rim_z - margin
    )


def _body_pose(
    data: mujoco.MjData,
    body_id: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    return (
        tuple(float(value) for value in data.xpos[body_id]),
        tuple(float(value) for value in data.xquat[body_id]),
    )


def _hide_and_park_object(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spec: GroceryObjectSpec,
    index: int,
) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, spec.joint_name)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, spec.geom_name)
    qpos_adr = model.jnt_qposadr[joint_id]
    qvel_adr = model.jnt_dofadr[joint_id]

    parking_pos = DEFAULT_PARKING_POS + np.array((0.0, 0.0, -0.25 * index))
    data.qpos[qpos_adr : qpos_adr + 3] = parking_pos
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[qvel_adr : qvel_adr + 6] = 0.0
    model.geom_contype[geom_id] = 0
    model.geom_conaffinity[geom_id] = 0
    model.geom_rgba[geom_id][3] = 0.0


def settle_objects(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    runtime_scene: RuntimeStartupScene,
    viewer: object | None = None,
    realtime: bool = True,
) -> StartupDropResult:
    config = runtime_scene.config
    quiet_steps = 0
    steps_run = 0

    for step in range(config.settle_steps):
        if viewer is not None and hasattr(viewer, "is_running") and not viewer.is_running():
            break

        frame_start = time.perf_counter()
        mujoco.mj_step(model, data)
        steps_run = step + 1

        if viewer is not None:
            viewer.sync()

        if step >= config.min_settle_steps:
            if _max_object_velocity(model, data, runtime_scene.specs) < config.velocity_epsilon:
                quiet_steps += 1
                if quiet_steps >= config.min_quiet_steps:
                    break
            else:
                quiet_steps = 0

        if realtime:
            _sleep_to_realtime(frame_start, model.opt.timestep)

    mujoco.mj_forward(model, data)

    kept_objects: list[ObjectPlacement] = []
    ignored_objects: list[ObjectPlacement] = []
    for index, spec in enumerate(runtime_scene.specs):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.name)
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, spec.geom_name)
        aabb_min, aabb_max = _geom_aabb(model, data, geom_id)
        final_pos, final_quat = _body_pose(data, body_id)
        placement = ObjectPlacement(
            spec=spec,
            valid=_is_inside_tote(aabb_min, aabb_max, runtime_scene.tote_bounds),
            final_pos=final_pos,
            final_quat=final_quat,
            aabb_min=tuple(float(value) for value in aabb_min),
            aabb_max=tuple(float(value) for value in aabb_max),
        )
        if placement.valid:
            kept_objects.append(placement)
        else:
            ignored_objects.append(placement)
            _hide_and_park_object(model, data, spec, index)

    mujoco.mj_forward(model, data)
    if viewer is not None:
        viewer.sync()

    return StartupDropResult(
        seed=runtime_scene.seed,
        settle_steps_run=steps_run,
        kept_objects=kept_objects,
        ignored_objects=ignored_objects,
    )


def write_settled_scene(
    base_scene_path: Path,
    output_path: Path,
    placements: Iterable[ObjectPlacement],
) -> None:
    tree, root = _clone_scene(base_scene_path)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Scene is missing a worldbody.")

    for placement in placements:
        worldbody.append(
            _make_object_body(
                placement.spec,
                pos=placement.final_pos,
                quat=placement.final_quat,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=False)
