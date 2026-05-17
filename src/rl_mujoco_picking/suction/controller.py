from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from rl_mujoco_picking.control import IKSolver, IKTarget
from rl_mujoco_picking.suction.seal import (
    SealEvaluation,
    SuctionCupParameters,
    evaluate_suction_seal,
    suction_axis,
)


DEFAULT_TARGET_ORDER = (
    "seal_sphere_small",
    "seal_cube_medium",
    "seal_rect_long",
    "seal_cube_tall",
)
TEST_SEAL_NEUTRAL_QPOS = (
    1.57,
    -1.57,
    1.57,
    -1.57,
    -1.57,
    0,
)
# User-tuned start pose for test_seal. Do not change this while tuning trajectories.


@dataclass(frozen=True)
class SealTarget:
    body_name: str
    geom_name: str
    pick_site_name: str
    weld_name: str


@dataclass(frozen=True)
class TestSealConfig:
    target_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    approach_height: float = 0.14
    lift_height: float = 0.26
    transit_clearance_height: float = 0.70
    descend_timeout_steps: int = 900
    max_linear_speed: float = 0.2


@dataclass(frozen=True)
class ControllerStatus:
    current_target: str | None
    phase: str
    attached_target: str | None
    completed_targets: tuple[str, ...]
    failed_targets: tuple[str, ...]
    last_seal_reason: str | None


def _geom_type_name(model: mujoco.MjModel, geom_id: int) -> str:
    geom_type = model.geom_type[geom_id]
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return "sphere"
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return "box"
    return f"unsupported:{int(geom_type)}"


def initialize_test_seal_robot_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: tuple[float, ...] = TEST_SEAL_NEUTRAL_QPOS,
) -> None:
    joint_names = (
        "handd_shoulder_pan_joint",
        "handd_shoulder_lift_joint",
        "handd_elbow_joint",
        "handd_wrist_1_joint",
        "handd_wrist_2_joint",
        "handd_wrist_3_joint",
    )
    for joint_name, joint_qpos in zip(joint_names, qpos, strict=True):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint not found: {joint_name}")
        qpos_adr = model.jnt_qposadr[joint_id]
        qvel_adr = model.jnt_dofadr[joint_id]
        data.qpos[qpos_adr] = joint_qpos
        data.qvel[qvel_adr] = 0.0
    data.ctrl[: len(joint_names)] = np.asarray(qpos, dtype=float)
    mujoco.mj_forward(model, data)


def _body_pose(data: mujoco.MjData, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    return np.array(data.xpos[body_id], dtype=float), np.array(data.xquat[body_id], dtype=float)


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    return np.array((quat[0], -quat[1], -quat[2], -quat[3]), dtype=float)


def _quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = lhs
    w2, x2, y2, z2 = rhs
    return np.array(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dtype=float,
    )


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    mat = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(mat, quat)
    return mat.reshape(3, 3)


def _relative_pose(
    parent_pos: np.ndarray,
    parent_quat: np.ndarray,
    child_pos: np.ndarray,
    child_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    parent_rot = _quat_to_mat(parent_quat)
    rel_pos = parent_rot.T @ (child_pos - parent_pos)
    rel_quat = _quat_multiply(_quat_conjugate(parent_quat), child_quat)
    rel_quat /= np.linalg.norm(rel_quat)
    return rel_pos, rel_quat


class SuctionGraspController:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        targets: list[SealTarget],
        config: TestSealConfig | None = None,
        suction_parameters: SuctionCupParameters | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.targets = targets
        self.config = config or TestSealConfig()
        self.suction_parameters = suction_parameters or SuctionCupParameters()
        self.ik_solver = IKSolver(model, data, suction_parameters=self.suction_parameters)
        self.planning_data = mujoco.MjData(model)
        self.position_planning_ik_solver = IKSolver(
            model,
            self.planning_data,
            suction_parameters=self.suction_parameters,
            position_gain=3.5,
            axis_gain=0.0,
            max_joint_step=0.05,
        )
        self.aligned_planning_ik_solver = IKSolver(
            model,
            self.planning_data,
            suction_parameters=self.suction_parameters,
            position_gain=3.5,
            axis_gain=0.8,
            max_joint_step=0.04,
        )
        self.target_axis = np.array(self.config.target_axis, dtype=float)
        self.target_axis /= np.linalg.norm(self.target_axis)
        self.completed_targets: list[str] = []
        self.failed_targets: list[str] = []
        self.current_index = 0
        self.phase = "approach"
        self.attached_target: SealTarget | None = None
        self.attached_rel_pos: np.ndarray | None = None
        self.attached_rel_quat: np.ndarray | None = None
        self.last_seal_reason: str | None = None
        self.phase_step_count = 0
        self.storage_tote_center_site_id = self._site_id("storage_tote_center_site")
        self.storage_tote_rim_site_id = self._site_id("storage_tote_rim_site")
        self.command_q = self.ik_solver.current_joint_positions()
        self.data.ctrl[: len(self.ik_solver.qpos_indices)] = self.command_q
        _, initial_axis = suction_axis(self.model, self.data, self.suction_parameters)
        self.tote_hover_q = self._solve_pose_ik(
            np.array(
                (
                    self._storage_tote_center()[0],
                    self._storage_tote_center()[1],
                    self._safe_transit_height(),
                ),
                dtype=float,
            ),
            seed_q=self.command_q,
            cup_axis=initial_axis,
            axis_tolerance=1.0,
        )
        self.target_hover_q: dict[str, np.ndarray] = {
            target.body_name: self._solve_pose_ik(
                self._safe_hover_target(target),
                seed_q=self.tote_hover_q,
                cup_axis=initial_axis,
                axis_tolerance=1.0,
            )
            for target in self.targets
        }

    def _bounded_target_position(self, requested_position: np.ndarray) -> np.ndarray:
        cup_center, _ = suction_axis(self.model, self.data, self.suction_parameters)
        delta = np.asarray(requested_position, dtype=float) - cup_center
        distance = float(np.linalg.norm(delta))
        if distance < 1e-9:
            return np.asarray(requested_position, dtype=float)
        max_step = self.config.max_linear_speed * self.model.opt.timestep
        if distance <= max_step:
            return np.asarray(requested_position, dtype=float)
        return cup_center + delta * (max_step / distance)

    @classmethod
    def from_test_seal_scene(
        cls,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: TestSealConfig | None = None,
        suction_parameters: SuctionCupParameters | None = None,
    ) -> "SuctionGraspController":
        tuned_parameters = suction_parameters or SuctionCupParameters(
            seal_radius=0.03,
            max_gap=0.004,
            compliance=0.004,
            min_alignment=0.75,
        )
        targets = [
            SealTarget(
                body_name=name,
                geom_name=f"{name}_geom",
                pick_site_name=f"{name}_pick_site",
                weld_name=f"seal_weld_{name.removeprefix('seal_')}",
            )
            for name in DEFAULT_TARGET_ORDER
        ]
        return cls(model, data, targets, config=config, suction_parameters=tuned_parameters)

    def status(self) -> ControllerStatus:
        current_target = None if self.current_index >= len(self.targets) else self.targets[self.current_index].body_name
        return ControllerStatus(
            current_target=current_target,
            phase=self.phase,
            attached_target=None if self.attached_target is None else self.attached_target.body_name,
            completed_targets=tuple(self.completed_targets),
            failed_targets=tuple(self.failed_targets),
            last_seal_reason=self.last_seal_reason,
        )

    def _site_id(self, site_name: str) -> int:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise ValueError(f"Site not found: {site_name}")
        return site_id

    def _target_ids(self, target: SealTarget) -> tuple[int, int, int, int]:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, target.body_name)
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, target.geom_name)
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, target.pick_site_name)
        eq_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, target.weld_name)
        if min(body_id, geom_id, site_id, eq_id) < 0:
            raise ValueError(f"Missing ids for target {target.body_name}")
        return body_id, geom_id, site_id, eq_id

    def _set_robot_ctrl(self, q_target: np.ndarray) -> None:
        dq = np.asarray(q_target, dtype=float) - self.command_q
        dq_norm = float(np.linalg.norm(dq))
        max_joint_step = self.ik_solver.max_joint_step
        if dq_norm > max_joint_step and dq_norm > 1e-9:
            dq *= max_joint_step / dq_norm
        bounded_q = self.command_q + dq
        self.data.ctrl[: len(bounded_q)] = bounded_q
        self.command_q = bounded_q.copy()

    def _joint_target_reached(self, q_target: np.ndarray, tolerance: float = 0.12) -> bool:
        current_q = self.ik_solver.current_joint_positions()
        return float(np.max(np.abs(current_q - np.asarray(q_target, dtype=float)))) <= tolerance

    def _plan_joint_target(
        self,
        position: np.ndarray,
        pos_tolerance: float = 0.006,
        axis_tolerance: float = 0.04,
        max_iterations: int = 80,
    ) -> np.ndarray:
        self.planning_data.qpos[:] = self.data.qpos
        self.planning_data.qvel[:] = 0.0
        self.planning_data.ctrl[:] = self.data.ctrl
        self.planning_data.qpos[self.aligned_planning_ik_solver.qpos_indices] = self.command_q
        mujoco.mj_forward(self.model, self.planning_data)

        ik_target = IKTarget(
            position=np.asarray(position, dtype=float),
            cup_axis=self.target_axis,
            pos_tolerance=pos_tolerance,
            axis_tolerance=axis_tolerance,
        )
        for _ in range(max_iterations):
            if self.aligned_planning_ik_solver.reached(ik_target):
                break
            q_target, _, _ = self.aligned_planning_ik_solver.solve_step(ik_target)
            self.planning_data.qpos[self.aligned_planning_ik_solver.qpos_indices] = q_target
            self.planning_data.qvel[self.aligned_planning_ik_solver.dof_indices] = 0.0
            mujoco.mj_forward(self.model, self.planning_data)
        return self.aligned_planning_ik_solver.current_joint_positions()

    def _solve_pose_ik(
        self,
        position: np.ndarray,
        seed_q: np.ndarray,
        cup_axis: np.ndarray | None = None,
        pos_tolerance: float = 0.006,
        axis_tolerance: float = 0.04,
        max_iterations: int = 120,
    ) -> np.ndarray:
        self.planning_data.qpos[:] = self.data.qpos
        self.planning_data.qvel[:] = 0.0
        self.planning_data.qpos[self.position_planning_ik_solver.qpos_indices] = np.asarray(seed_q, dtype=float)
        mujoco.mj_forward(self.model, self.planning_data)

        ik_target = IKTarget(
            position=np.asarray(position, dtype=float),
            cup_axis=self.target_axis if cup_axis is None else np.asarray(cup_axis, dtype=float),
            pos_tolerance=pos_tolerance,
            axis_tolerance=axis_tolerance,
        )
        for _ in range(max_iterations):
            if self.position_planning_ik_solver.reached(ik_target):
                break
            q_target, _, _ = self.position_planning_ik_solver.solve_step(ik_target)
            self.planning_data.qpos[self.position_planning_ik_solver.qpos_indices] = q_target
            self.planning_data.qvel[self.position_planning_ik_solver.dof_indices] = 0.0
            mujoco.mj_forward(self.model, self.planning_data)
        return self.position_planning_ik_solver.current_joint_positions().copy()

    def _current_target(self) -> SealTarget | None:
        if self.current_index >= len(self.targets):
            return None
        return self.targets[self.current_index]

    def _pick_site_target(self, target: SealTarget, z_offset: float) -> np.ndarray:
        _, _, site_id, _ = self._target_ids(target)
        return np.array(self.data.site_xpos[site_id], dtype=float) + np.array((0.0, 0.0, z_offset), dtype=float)

    def _storage_tote_center(self) -> np.ndarray:
        return np.array(self.data.site_xpos[self.storage_tote_center_site_id], dtype=float)

    def _storage_tote_rim(self) -> np.ndarray:
        return np.array(self.data.site_xpos[self.storage_tote_rim_site_id], dtype=float)

    def _safe_transit_height(self) -> float:
        rim = self._storage_tote_rim()
        return float(rim[2] + self.config.transit_clearance_height)

    def _safe_hover_target(self, target: SealTarget) -> np.ndarray:
        pick_target = self._pick_site_target(target, self.config.approach_height)
        pick_target[2] = max(pick_target[2], self._safe_transit_height())
        return pick_target

    def _approach_waypoint(self, target: SealTarget) -> np.ndarray:
        cup_center, _ = suction_axis(self.model, self.data, self.suction_parameters)
        safe_z = max(cup_center[2], self._safe_transit_height())
        tote_center = self._storage_tote_center()
        hover_target = self._safe_hover_target(target)

        if cup_center[2] < safe_z - 0.02:
            return np.array((cup_center[0], cup_center[1], safe_z), dtype=float)

        tote_xy_error = float(np.linalg.norm(cup_center[:2] - tote_center[:2]))
        if tote_xy_error > 0.08:
            return np.array((tote_center[0], tote_center[1], safe_z), dtype=float)

        target_xy_error = float(np.linalg.norm(cup_center[:2] - hover_target[:2]))
        if target_xy_error > 0.02:
            return np.array((hover_target[0], hover_target[1], safe_z), dtype=float)

        return hover_target

    def _activate_weld(self, target: SealTarget) -> None:
        tool_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "handd_tool_cup_tip_link")
        body_id, _, _, eq_id = self._target_ids(target)
        tool_pos, tool_quat = _body_pose(self.data, tool_body_id)
        body_pos, body_quat = _body_pose(self.data, body_id)
        rel_pos, rel_quat = _relative_pose(tool_pos, tool_quat, body_pos, body_quat)
        self.model.eq_data[eq_id, 3:6] = rel_pos
        self.model.eq_data[eq_id, 6:10] = rel_quat
        self.data.eq_active[eq_id] = 1
        self.attached_rel_pos = rel_pos
        self.attached_rel_quat = rel_quat
        mujoco.mj_forward(self.model, self.data)

    def _deactivate_weld(self, target: SealTarget) -> None:
        _, _, _, eq_id = self._target_ids(target)
        self.data.eq_active[eq_id] = 0
        self.attached_rel_pos = None
        self.attached_rel_quat = None

    def _hide_body(self, target: SealTarget) -> None:
        body_id, geom_id, _, _ = self._target_ids(target)
        joint_name = f"{target.body_name}_freejoint"
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Freejoint not found: {joint_name}")
        qpos_adr = self.model.jnt_qposadr[joint_id]
        qvel_adr = self.model.jnt_dofadr[joint_id]
        self.data.qpos[qpos_adr : qpos_adr + 3] = (3.5, -3.5, -4.0 - 0.2 * len(self.completed_targets))
        self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[qvel_adr : qvel_adr + 6] = 0.0
        self.model.geom_rgba[geom_id][3] = 0.0
        self.model.geom_contype[geom_id] = 0
        self.model.geom_conaffinity[geom_id] = 0
        _ = body_id

    def _advance_target(self, failed: bool) -> None:
        current = self._current_target()
        if current is not None:
            if failed:
                self.failed_targets.append(current.body_name)
            else:
                self.completed_targets.append(current.body_name)
        self.current_index += 1
        self.phase = "approach"
        self.phase_step_count = 0
        self.last_seal_reason = None

    def _lift_clear(self, target: SealTarget) -> bool:
        cup_center, _ = suction_axis(self.model, self.data, self.suction_parameters)
        lift_target = self._pick_site_target(target, self.config.lift_height)
        return cup_center[2] >= lift_target[2] - 0.01

    def step(self) -> ControllerStatus:
        target = self._current_target()
        if target is None:
            self.data.ctrl[:6] = self.ik_solver.current_joint_positions()
            self.phase = "done"
            return self.status()

        self.phase_step_count += 1

        if self.phase == "approach":
            if not self._joint_target_reached(self.tote_hover_q):
                self._set_robot_ctrl(self.tote_hover_q)
                return self.status()
            hover_q = self.target_hover_q[target.body_name]
            if not self._joint_target_reached(hover_q):
                self._set_robot_ctrl(hover_q)
                return self.status()
            self.phase = "descend"
            self.phase_step_count = 0
            return self.status()

        if self.phase == "descend":
            requested_position = self._pick_site_target(target, 0.0)
            target_position = self._bounded_target_position(requested_position)
            q_target = self._plan_joint_target(
                target_position,
                pos_tolerance=0.004,
                axis_tolerance=0.03,
            )
            self._set_robot_ctrl(q_target)
            seal_eval = evaluate_suction_seal(self.model, self.data, target.geom_name, self.suction_parameters)
            self.last_seal_reason = seal_eval.reason
            if seal_eval.sealable:
                self._activate_weld(target)
                self.attached_target = target
                self.phase = "lift"
                self.phase_step_count = 0
            elif self.ik_solver.reached(
                IKTarget(
                    position=requested_position,
                    cup_axis=self.target_axis,
                    pos_tolerance=0.004,
                    axis_tolerance=0.03,
                )
            ) or self.phase_step_count >= self.config.descend_timeout_steps:
                self._advance_target(failed=True)
            return self.status()

        if self.phase == "lift":
            target_position = self._bounded_target_position(
                self._pick_site_target(target, self.config.lift_height)
            )
            q_target = self._plan_joint_target(target_position)
            self._set_robot_ctrl(q_target)
            if self.attached_target is not None and self._lift_clear(self.attached_target):
                self._deactivate_weld(self.attached_target)
                self._hide_body(self.attached_target)
                self.attached_target = None
                mujoco.mj_forward(self.model, self.data)
                self._advance_target(failed=False)
            return self.status()

        raise ValueError(f"Unsupported controller phase: {self.phase}")
