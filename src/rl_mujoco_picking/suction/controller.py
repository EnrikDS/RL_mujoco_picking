"""Scriptable suction-picking controller used by the ``test_seal`` scene.

This controller is deliberately not a full motion planner.  It creates simple
Cartesian waypoint chains, solves IK for each waypoint, and then commands the
UR arm while continuously checking whether the suction cup has sealed.
The goal is to have a transparent debugging harness before moving the same
ideas into an RL environment reset/step loop.
"""

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
    "seal_rect_long",
    "seal_cube_tall",
    "seal_cube_medium",
    "seal_sphere_small",
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
    """Names that connect one pickable object to its geom, pick site and weld."""

    body_name: str
    geom_name: str
    pick_site_name: str
    weld_name: str


@dataclass(frozen=True)
class TestSealConfig:
    """Tunable parameters for the deterministic test_seal pick-and-drop routine."""

    # Desired cup axis in world frame.  In this scene +Z means "cup normal points down to the tote surface normal".
    target_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    # One frame is used for control and debug markers so visual and controlled positions agree.
    trajectory_site_name: str = "handd_tool_seal_tip_site"
    delivery_drop_site_name: str = "delivery_tote_1_drop_site"
    # Heights are intentionally conservative while we debug contacts and tool compliance.
    approach_height: float = 0.14
    lift_height: float = 0.26
    seal_probe_depth: float = 0.10
    transit_clearance_height: float = 0.55
    # Cartesian waypoints are close enough to debug the path but not so dense that execution crawls.
    waypoint_spacing: float = 0.03
    waypoint_tolerance: float = 0.01
    waypoint_stall_tolerance: float = 0.015
    waypoint_orientation_tolerance: float = 0.06
    descend_timeout_steps: int = 7000
    release_settle_steps: int = 250
    max_linear_speed: float = 0.2
    gravity_compensation: bool = True
    stop_on_environment_contact: bool = False


@dataclass(frozen=True)
class ControllerStatus:
    """Small immutable snapshot returned after every controller step."""

    current_target: str | None
    phase: str
    attached_target: str | None
    completed_targets: tuple[str, ...]
    failed_targets: tuple[str, ...]
    last_seal_reason: str | None
    last_contact: str | None


@dataclass(frozen=True)
class JacobianDiagnostics:
    """Numerical health metrics for the controlled site Jacobian."""

    determinant: float
    condition: float
    min_singular_value: float
    max_singular_value: float


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
    """Set the user-tuned neutral arm pose and synchronize actuators to it."""

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
    # Position actuators must start at the same values or gravity immediately pulls the arm away.
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
    """Express a child body pose in the local frame of a parent body."""

    parent_rot = _quat_to_mat(parent_quat)
    rel_pos = parent_rot.T @ (child_pos - parent_pos)
    rel_quat = _quat_multiply(_quat_conjugate(parent_quat), child_quat)
    rel_quat /= np.linalg.norm(rel_quat)
    return rel_pos, rel_quat


def _frame_with_x_axis(x_axis: np.ndarray, reference_frame: np.ndarray) -> np.ndarray:
    """Build an orthonormal frame that keeps X aligned while preserving roll as much as possible."""

    x_axis = np.asarray(x_axis, dtype=float)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.asarray(reference_frame[:, 1], dtype=float)
    y_axis = y_axis - float(np.dot(y_axis, x_axis)) * x_axis
    if np.linalg.norm(y_axis) < 1e-6:
        y_axis = np.array((0.0, 1.0, 0.0), dtype=float)
        y_axis = y_axis - float(np.dot(y_axis, x_axis)) * x_axis
    if np.linalg.norm(y_axis) < 1e-6:
        y_axis = np.array((1.0, 0.0, 0.0), dtype=float)
        y_axis = y_axis - float(np.dot(y_axis, x_axis)) * x_axis
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


class SuctionGraspController:
    """Finite-state controller for testing suction picking before training RL."""

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
        # IK should track the same site that is drawn in debug mode; seal parameters are copied separately.
        self.trajectory_parameters = SuctionCupParameters(
            tip_site_name=self.config.trajectory_site_name,
            uncompressed_site_name=self.suction_parameters.uncompressed_site_name,
            seal_radius=self.suction_parameters.seal_radius,
            contact_offset=self.suction_parameters.contact_offset,
            lip_tolerance=self.suction_parameters.lip_tolerance,
            compliance=self.suction_parameters.compliance,
            max_gap=self.suction_parameters.max_gap,
            min_alignment=self.suction_parameters.min_alignment,
            radial_margin=self.suction_parameters.radial_margin,
        )
        self.ik_solver = IKSolver(model, data, suction_parameters=self.trajectory_parameters)
        # Planning uses a separate MjData so offline IK does not disturb the live simulation state.
        self.planning_data = mujoco.MjData(model)
        self.planning_ik_solver = IKSolver(
            model,
            self.planning_data,
            suction_parameters=self.trajectory_parameters,
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
        self.last_contact: str | None = None
        self.phase_step_count = 0
        self.storage_tote_center_site_id = self._site_id("storage_tote_center_site")
        self.storage_tote_rim_site_id = self._site_id("storage_tote_rim_site")
        self.delivery_drop_site_id = self._site_id(self.config.delivery_drop_site_name)
        self.trajectory_site_id = self._site_id(self.config.trajectory_site_name)
        self.command_q = self.ik_solver.current_joint_positions()
        self.data.ctrl[: len(self.ik_solver.qpos_indices)] = self.command_q
        # Freeze the tool orientation from the initial pose; approach/descent should not roll the cup.
        _, self.fixed_cup_axis = suction_axis(self.model, self.data, self.trajectory_parameters)
        self.fixed_tip_xmat = self._tip_xmat(self.data)
        self.plan_target_name: str | None = None
        self.plan_phase: str | None = None
        self.plan_qpos: list[np.ndarray] = []
        self.plan_positions: list[np.ndarray] = []
        self.plan_index = 0
        self.plan_wait_steps = 0

    def _apply_gravity_compensation(self) -> None:
        """Apply feed-forward bias forces so the position actuators do not fight gravity alone."""

        if not self.config.gravity_compensation:
            return
        self.data.qfrc_applied[self.ik_solver.dof_indices] = self.data.qfrc_bias[self.ik_solver.dof_indices]

    def trajectory_position(self) -> np.ndarray:
        """Current world position of the site controlled by the trajectory."""

        return np.array(self.data.site_xpos[self.trajectory_site_id], dtype=float)

    def _bounded_target_position(self, requested_position: np.ndarray) -> np.ndarray:
        """Limit a Cartesian request to the configured max tool speed.

        This helper is kept for experiments with continuous tracking; the
        current waypoint follower mainly limits motion through joint step size.
        """

        cup_center = self.trajectory_position()
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
        """Factory for the fixed objects and tolerances used in models/scenes/test_seal."""

        tuned_parameters = suction_parameters or SuctionCupParameters(
            seal_radius=0.03,
            contact_offset=0.026,
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
        """Expose controller state without giving callers mutable internals."""

        current_target = None if self.current_index >= len(self.targets) else self.targets[self.current_index].body_name
        return ControllerStatus(
            current_target=current_target,
            phase=self.phase,
            attached_target=None if self.attached_target is None else self.attached_target.body_name,
            completed_targets=tuple(self.completed_targets),
            failed_targets=tuple(self.failed_targets),
            last_seal_reason=self.last_seal_reason,
            last_contact=self.last_contact,
        )

    def _site_id(self, site_name: str) -> int:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise ValueError(f"Site not found: {site_name}")
        return site_id

    def _target_ids(self, target: SealTarget) -> tuple[int, int, int, int]:
        """Resolve all MuJoCo ids needed to operate on one target object."""

        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, target.body_name)
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, target.geom_name)
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, target.pick_site_name)
        eq_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, target.weld_name)
        if min(body_id, geom_id, site_id, eq_id) < 0:
            raise ValueError(f"Missing ids for target {target.body_name}")
        return body_id, geom_id, site_id, eq_id

    def _set_robot_ctrl(self, q_target: np.ndarray) -> None:
        """Send a bounded joint target to the position actuators."""

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

    def _tip_xmat(self, data: mujoco.MjData) -> np.ndarray:
        return np.array(data.site_xmat[self.ik_solver.tip_site_id], dtype=float).reshape(3, 3)

    def debug_waypoint_positions(self) -> tuple[np.ndarray, ...]:
        return tuple(position.copy() for position in self.plan_positions)

    def active_waypoint_position(self) -> np.ndarray | None:
        if self.plan_index >= len(self.plan_positions):
            return None
        return self.plan_positions[self.plan_index].copy()

    def active_waypoint_error(self) -> float | None:
        """Distance from controlled site to the active Cartesian waypoint."""

        waypoint = self.active_waypoint_position()
        if waypoint is None:
            return None
        return float(np.linalg.norm(self.trajectory_position() - waypoint))

    def active_waypoint_orientation_error(self) -> float | None:
        """Orientation error relative to the initial tool frame used for constant-orientation motion."""

        if self.active_waypoint_position() is None:
            return None
        current_frame = self._tip_xmat(self.data)
        orientation_error = 0.5 * (
            np.cross(current_frame[:, 0], self.fixed_tip_xmat[:, 0])
            + np.cross(current_frame[:, 1], self.fixed_tip_xmat[:, 1])
            + np.cross(current_frame[:, 2], self.fixed_tip_xmat[:, 2])
        )
        return float(np.linalg.norm(orientation_error))

    def compliance_displacement(self) -> float:
        """Current slide-joint displacement of the compliant cup mount."""

        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "handd_tool_cup_compliance_joint")
        if joint_id < 0:
            return 0.0
        return float(self.data.qpos[self.model.jnt_qposadr[joint_id]])

    def compliance_velocity(self) -> float:
        """Current slide-joint velocity of the compliant cup mount."""

        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "handd_tool_cup_compliance_joint")
        if joint_id < 0:
            return 0.0
        return float(self.data.qvel[self.model.jnt_dofadr[joint_id]])

    def jacobian_diagnostics(self) -> JacobianDiagnostics:
        """Compute Jacobian conditioning for debugging sudden tracking failures."""

        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ik_solver.tip_site_id)
        jacobian = np.vstack((jacp[:, self.ik_solver.dof_indices], jacr[:, self.ik_solver.dof_indices]))
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        min_singular = float(np.min(singular_values))
        max_singular = float(np.max(singular_values))
        condition = float("inf") if min_singular < 1e-12 else max_singular / min_singular
        return JacobianDiagnostics(
            determinant=float(np.linalg.det(jacobian)),
            condition=condition,
            min_singular_value=min_singular,
            max_singular_value=max_singular,
        )

    def current_seal_evaluation(self) -> SealEvaluation | None:
        """Evaluate the current target with the current live simulation pose."""

        target = self._current_target()
        if target is None:
            return None
        return evaluate_suction_seal(self.model, self.data, target.geom_name, self.suction_parameters)

    def _robot_environment_contact(self) -> str | None:
        """Return a readable contact string for robot-environment collisions."""

        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            body1 = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                self.model.geom_bodyid[contact.geom1],
            ) or ""
            body2 = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                self.model.geom_bodyid[contact.geom2],
            ) or ""
            robot1 = body1.startswith("handd_")
            robot2 = body2.startswith("handd_")
            if robot1 == robot2:
                continue
            if body1.startswith("seal_") or body2.startswith("seal_"):
                # Object contacts are expected during picking and should not stop the debug run.
                continue
            geom1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or f"geom_{contact.geom1}"
            geom2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or f"geom_{contact.geom2}"
            return f"{body1}/{geom1} vs {body2}/{geom2} dist={contact.dist:.5f}"
        return None

    def _reset_plan(self) -> None:
        """Forget the current waypoint plan when changing phase or target."""

        self.plan_target_name = None
        self.plan_phase = None
        self.plan_qpos = []
        self.plan_positions = []
        self.plan_index = 0
        self.plan_wait_steps = 0

    def _cartesian_waypoints(self, anchors: list[np.ndarray]) -> list[np.ndarray]:
        """Interpolate straight Cartesian segments at approximately waypoint_spacing."""

        waypoints: list[np.ndarray] = [np.asarray(anchors[0], dtype=float)]
        spacing = max(self.config.waypoint_spacing, 1e-6)
        for start, end in zip(anchors, anchors[1:], strict=False):
            delta = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
            distance = float(np.linalg.norm(delta))
            if distance < 1e-9:
                continue
            steps = max(1, int(np.ceil(distance / spacing)))
            for step in range(1, steps + 1):
                waypoints.append(np.asarray(start, dtype=float) + delta * (step / steps))
        return waypoints

    def _solve_waypoint_ik(
        self,
        position: np.ndarray,
        seed_q: np.ndarray,
        pos_tolerance: float = 0.008,
        axis_tolerance: float = 0.06,
        max_iterations: int = 180,
    ) -> tuple[np.ndarray, float, float]:
        """Solve IK for one waypoint using the previous waypoint as the seed."""

        self.planning_data.qpos[:] = self.data.qpos
        self.planning_data.qvel[:] = 0.0
        self.planning_data.qpos[self.planning_ik_solver.qpos_indices] = np.asarray(seed_q, dtype=float)
        mujoco.mj_forward(self.model, self.planning_data)

        ik_target = IKTarget(
            position=np.asarray(position, dtype=float),
            cup_axis=self.fixed_cup_axis,
            pos_tolerance=pos_tolerance,
            axis_tolerance=axis_tolerance,
            frame_xmat=self.fixed_tip_xmat,
        )
        pos_error = float("inf")
        orient_error = float("inf")
        for _ in range(max_iterations):
            if self.planning_ik_solver.reached(ik_target):
                break
            q_target, pos_error, orient_error = self.planning_ik_solver.solve_step(ik_target)
            # Advance only the planning copy; the real robot is moved later in _follow_phase_plan.
            self.planning_data.qpos[self.planning_ik_solver.qpos_indices] = q_target
            self.planning_data.qvel[self.planning_ik_solver.dof_indices] = 0.0
            mujoco.mj_forward(self.model, self.planning_data)
        _, pos_error, orient_error = self.planning_ik_solver.solve_step(ik_target)
        return self.planning_ik_solver.current_joint_positions().copy(), pos_error, orient_error

    def _build_phase_plan(self, target: SealTarget, phase: str) -> None:
        """Build all Cartesian waypoints and IK joint targets for one controller phase."""

        cup_center = self.trajectory_position()
        pick_position = self._pick_site_target(target, 0.0)
        transit_z = max(cup_center[2], self._safe_transit_height())
        approach_z = max(transit_z, pick_position[2] + self.config.approach_height)
        delivery_drop = self._delivery_drop_position()

        if phase == "approach":
            # Go up first, then translate over the object; this avoids dragging the tool through the tote.
            anchors = [
                cup_center,
                np.array((cup_center[0], cup_center[1], approach_z), dtype=float),
                np.array((pick_position[0], pick_position[1], approach_z), dtype=float),
            ]
        elif phase == "descend":
            # Descend vertically beyond the nominal pick site until seal or timeout.
            probe_position = pick_position - np.array((0.0, 0.0, self.config.seal_probe_depth), dtype=float)
            anchors = [
                cup_center,
                np.array((pick_position[0], pick_position[1], cup_center[2]), dtype=float),
                np.array((pick_position[0], pick_position[1], probe_position[2]), dtype=float),
            ]
        elif phase == "retract":
            # Once welded, move straight up before going sideways to the delivery tote.
            anchors = [
                cup_center,
                np.array((cup_center[0], cup_center[1], transit_z), dtype=float),
            ]
        elif phase == "to_delivery":
            # Move over the delivery tote and lower to a drop site above it.
            delivery_safe_z = max(transit_z, delivery_drop[2] + 0.05)
            anchors = [
                cup_center,
                np.array((cup_center[0], cup_center[1], delivery_safe_z), dtype=float),
                np.array((delivery_drop[0], delivery_drop[1], delivery_safe_z), dtype=float),
                delivery_drop,
            ]
        else:
            raise ValueError(f"Unsupported plan phase: {phase}")

        positions = self._cartesian_waypoints(anchors)
        qpos: list[np.ndarray] = []
        seed_q = self.command_q.copy()
        for position in positions:
            # Seeding with the previous solution avoids large IK jumps between nearby waypoints.
            seed_q, _, _ = self._solve_waypoint_ik(position, seed_q=seed_q)
            qpos.append(seed_q.copy())

        self.plan_target_name = target.body_name
        self.plan_phase = phase
        self.plan_positions = positions
        self.plan_qpos = qpos
        self.plan_index = 0

    def _ensure_phase_plan(self, target: SealTarget, phase: str) -> None:
        """Reuse the current plan unless target or phase changed."""

        if self.plan_target_name == target.body_name and self.plan_phase == phase:
            return
        self._build_phase_plan(target, phase)

    def _advance_plan_waypoint(self) -> bool:
        """Move to the next waypoint and report whether the phase plan is finished."""

        self.plan_index += 1
        self.plan_wait_steps = 0
        return self.plan_index >= len(self.plan_qpos)

    def _follow_phase_plan(self) -> bool:
        """Track the active waypoint and advance when the live robot is close enough."""

        if self.plan_index >= len(self.plan_qpos):
            return True
        self.plan_wait_steps += 1
        waypoint_error = self.active_waypoint_error()
        orientation_error = self.active_waypoint_orientation_error()
        if (
            waypoint_error is not None
            and waypoint_error <= self.config.waypoint_tolerance
            and self.plan_index == 0
        ):
            # The first waypoint is usually the current pose; do not require orientation settling there.
            return self._advance_plan_waypoint()
        if (
            waypoint_error is not None
            and orientation_error is not None
            and waypoint_error <= self.config.waypoint_tolerance
            and orientation_error <= self.config.waypoint_orientation_tolerance
        ):
            return self._advance_plan_waypoint()
        if (
            waypoint_error is not None
            and waypoint_error <= self.config.waypoint_stall_tolerance
            and self.plan_wait_steps >= 200
        ):
            # Avoid freezing forever near a waypoint when contact/compliance prevents perfect convergence.
            return self._advance_plan_waypoint()

        waypoint = self.plan_positions[self.plan_index]
        q_target, _, _ = self.ik_solver.solve_step(
            IKTarget(
                position=waypoint,
                cup_axis=self.fixed_cup_axis,
                pos_tolerance=self.config.waypoint_tolerance,
                axis_tolerance=0.08,
                frame_xmat=self.fixed_tip_xmat,
            )
        )
        self._set_robot_ctrl(q_target)
        return self.plan_index >= len(self.plan_qpos)

    def _current_target(self) -> SealTarget | None:
        if self.current_index >= len(self.targets):
            return None
        return self.targets[self.current_index]

    def _pick_site_target(self, target: SealTarget, z_offset: float) -> np.ndarray:
        """World position of the target's authored pick site plus optional vertical offset."""

        _, _, site_id, _ = self._target_ids(target)
        return np.array(self.data.site_xpos[site_id], dtype=float) + np.array((0.0, 0.0, z_offset), dtype=float)

    def _storage_tote_center(self) -> np.ndarray:
        return np.array(self.data.site_xpos[self.storage_tote_center_site_id], dtype=float)

    def _storage_tote_rim(self) -> np.ndarray:
        return np.array(self.data.site_xpos[self.storage_tote_rim_site_id], dtype=float)

    def _delivery_drop_position(self) -> np.ndarray:
        return np.array(self.data.site_xpos[self.delivery_drop_site_id], dtype=float)

    def _safe_transit_height(self) -> float:
        """Height used when moving laterally between totes or objects."""

        rim = self._storage_tote_rim()
        return float(rim[2] + self.config.transit_clearance_height)

    def _safe_hover_target(self, target: SealTarget) -> np.ndarray:
        pick_target = self._pick_site_target(target, self.config.approach_height)
        pick_target[2] = max(pick_target[2], self._safe_transit_height())
        return pick_target

    def _approach_waypoint(self, target: SealTarget) -> np.ndarray:
        cup_center = self.trajectory_position()
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
        """Attach the target body to the suction cup using its predeclared weld equality."""

        tool_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "handd_tool_cup_tip_link")
        body_id, _, _, eq_id = self._target_ids(target)
        tool_pos, tool_quat = _body_pose(self.data, tool_body_id)
        body_pos, body_quat = _body_pose(self.data, body_id)
        rel_pos, rel_quat = _relative_pose(tool_pos, tool_quat, body_pos, body_quat)
        # Store the current relative pose so enabling the weld does not snap the object.
        self.model.eq_data[eq_id, 3:6] = rel_pos
        self.model.eq_data[eq_id, 6:10] = rel_quat
        self.data.eq_active[eq_id] = 1
        self.attached_rel_pos = rel_pos
        self.attached_rel_quat = rel_quat
        mujoco.mj_forward(self.model, self.data)

    def _deactivate_weld(self, target: SealTarget) -> None:
        """Release the object by disabling its weld equality."""

        _, _, _, eq_id = self._target_ids(target)
        self.data.eq_active[eq_id] = 0
        self.attached_rel_pos = None
        self.attached_rel_quat = None

    def _hide_body(self, target: SealTarget) -> None:
        """Park a completed/ignored object far below the scene and disable its collisions."""

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
        """Record target outcome and reset the state machine for the next object."""

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
        cup_center = self.trajectory_position()
        lift_target = self._pick_site_target(target, self.config.lift_height)
        return cup_center[2] >= lift_target[2] - 0.01

    def step(self) -> ControllerStatus:
        """Advance the finite-state controller by one simulation control step."""

        self._apply_gravity_compensation()
        contact = self._robot_environment_contact()
        if contact is not None:
            self.last_contact = contact
            if self.config.stop_on_environment_contact:
                self.phase = "collision"
                self.data.ctrl[: len(self.command_q)] = self.command_q
                return self.status()
        else:
            self.last_contact = None
        target = self._current_target()
        if target is None:
            self.data.ctrl[:6] = self.ik_solver.current_joint_positions()
            self.phase = "done"
            return self.status()

        self.phase_step_count += 1

        if self.phase == "approach":
            # Move from the current pose to a safe hover pose above the active object.
            self._ensure_phase_plan(target, "approach")
            if self._follow_phase_plan():
                self.phase = "descend"
                self.phase_step_count = 0
                self._reset_plan()
            return self.status()

        if self.phase == "descend":
            # Move vertically down while continuously evaluating the suction seal.
            self._ensure_phase_plan(target, "descend")
            seal_eval = self.current_seal_evaluation()
            if seal_eval is None:
                self.phase = "done"
                return self.status()
            self.last_seal_reason = seal_eval.reason
            if seal_eval.sealable:
                # A valid seal immediately turns into a weld so the object follows the cup.
                self._activate_weld(target)
                self.attached_target = target
                self.phase = "retract"
                self.phase_step_count = 0
                self._reset_plan()
            elif self._follow_phase_plan() or self.phase_step_count >= self.config.descend_timeout_steps:
                self._advance_target(failed=True)
                self._reset_plan()
            return self.status()

        if self.phase == "retract":
            # Pull the sealed object vertically away from the storage tote.
            self._ensure_phase_plan(target, "retract")
            if self._follow_phase_plan():
                self.phase = "to_delivery"
                self.phase_step_count = 0
                self._reset_plan()
            return self.status()

        if self.phase == "to_delivery":
            # Carry the object to the delivery tote and release it at the drop site.
            self._ensure_phase_plan(target, "to_delivery")
            if self._follow_phase_plan():
                if self.attached_target is not None:
                    self._deactivate_weld(self.attached_target)
                    self.attached_target = None
                    mujoco.mj_forward(self.model, self.data)
                self.phase = "release"
                self.phase_step_count = 0
                self._reset_plan()
            return self.status()

        if self.phase == "release":
            # Let the dropped object settle before moving on to the next target.
            self.data.ctrl[: len(self.command_q)] = self.command_q
            if self.phase_step_count >= self.config.release_settle_steps:
                self._advance_target(failed=False)
                self._reset_plan()
            return self.status()

        if self.phase == "collision":
            self.data.ctrl[: len(self.command_q)] = self.command_q
            return self.status()

        raise ValueError(f"Unsupported controller phase: {self.phase}")
