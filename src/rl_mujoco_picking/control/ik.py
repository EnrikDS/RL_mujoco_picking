"""Small damped-least-squares IK helper for the UR arm.

The controller asks this solver for one bounded joint update at a time.  This
keeps the logic easy to inspect: high-level code owns waypoints and phases,
while this module only converts a Cartesian tool target into joint commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np


DEFAULT_ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


@dataclass(frozen=True)
class IKTarget:
    """Cartesian target for the controlled tool site."""

    position: np.ndarray
    cup_axis: np.ndarray
    pos_tolerance: float = 0.006
    axis_tolerance: float = 0.04
    frame_xmat: np.ndarray | None = None


@dataclass(frozen=True)
class RobotJointSet:
    """Names and MuJoCo indices for the six arm joints used by IK."""

    joint_names: tuple[str, ...] = DEFAULT_ARM_JOINTS

    def joint_ids(self, model: mujoco.MjModel) -> list[int]:
        """Resolve joint ids, accepting both plain UR names and handd-prefixed names."""

        joint_ids: list[int] = []
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"handd_{joint_name}")
            if joint_id < 0:
                raise ValueError(f"Joint not found: {joint_name}")
            joint_ids.append(joint_id)
        return joint_ids

    def qpos_indices(self, model: mujoco.MjModel) -> np.ndarray:
        return np.array([model.jnt_qposadr[joint_id] for joint_id in self.joint_ids(model)], dtype=int)

    def dof_indices(self, model: mujoco.MjModel) -> np.ndarray:
        return np.array([model.jnt_dofadr[joint_id] for joint_id in self.joint_ids(model)], dtype=int)


class IKSolver:
    """Damped least-squares inverse kinematics around one tool site."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_set: RobotJointSet | None = None,
        suction_parameters: Any | None = None,
        damping: float = 1e-3,
        position_gain: float = 2.5,
        axis_gain: float = 0.8,
        max_joint_step: float = 0.08,
    ) -> None:
        self.model = model
        self.data = data
        self.joint_set = joint_set or RobotJointSet()
        self.suction_parameters = suction_parameters
        self.damping = damping
        self.position_gain = position_gain
        self.axis_gain = axis_gain
        self.max_joint_step = max_joint_step
        self.qpos_indices = self.joint_set.qpos_indices(model)
        self.dof_indices = self.joint_set.dof_indices(model)
        # The tip/axis sites come from suction parameters so control and seal can share frames.
        self.tip_site_name = (
            "handd_tool_seal_tip_site"
            if self.suction_parameters is None
            else self.suction_parameters.tip_site_name
        )
        self.uncompressed_site_name = (
            "handd_tool_seal_axis_site"
            if self.suction_parameters is None
            else self.suction_parameters.uncompressed_site_name
        )
        self.tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.tip_site_name)
        if self.tip_site_id < 0:
            raise ValueError(f"Tip site not found: {self.tip_site_name}")
        self.uncompressed_site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            self.uncompressed_site_name,
        )
        if self.uncompressed_site_id < 0:
            raise ValueError(f"Uncompressed site not found: {self.uncompressed_site_name}")

    def suction_axis(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the current tip position and cup longitudinal axis."""

        tip_pos = np.array(self.data.site_xpos[self.tip_site_id], dtype=float)
        uncompressed_pos = np.array(self.data.site_xpos[self.uncompressed_site_id], dtype=float)
        axis = uncompressed_pos - tip_pos
        axis /= np.linalg.norm(axis)
        return tip_pos, axis

    def tip_position(self) -> np.ndarray:
        return np.array(self.data.site_xpos[self.tip_site_id], dtype=float)

    def current_joint_positions(self) -> np.ndarray:
        """Current qpos values for the controlled six joints."""

        return np.array(self.data.qpos[self.qpos_indices], dtype=float)

    def solve_step(self, target: IKTarget) -> tuple[np.ndarray, float, float]:
        """Compute one bounded IK update toward a target.

        Position error uses the site translational Jacobian.  Orientation error
        either aligns only the suction axis or, when ``frame_xmat`` is provided,
        keeps the full tool frame close to the chosen reference orientation.
        """

        cup_center = self.tip_position()
        position_error = np.asarray(target.position, dtype=float) - cup_center
        if target.frame_xmat is None:
            _, current_axis = self.suction_axis()
            axis_target = np.asarray(target.cup_axis, dtype=float)
            axis_target /= np.linalg.norm(axis_target)
            orientation_error = np.cross(current_axis, axis_target)
        else:
            current_frame = np.array(self.data.site_xmat[self.tip_site_id], dtype=float).reshape(3, 3)
            target_frame = np.asarray(target.frame_xmat, dtype=float).reshape(3, 3)
            orientation_error = 0.5 * (
                np.cross(current_frame[:, 0], target_frame[:, 0])
                + np.cross(current_frame[:, 1], target_frame[:, 1])
                + np.cross(current_frame[:, 2], target_frame[:, 2])
            )

        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.tip_site_id)
        # Build a 6x6 task Jacobian: top rows are translation, bottom rows are rotation.
        jacobian = np.vstack(
            (
                self.position_gain * jacp[:, self.dof_indices],
                self.axis_gain * jacr[:, self.dof_indices],
            )
        )
        error = np.concatenate(
            (
                self.position_gain * position_error,
                self.axis_gain * orientation_error,
            )
        )
        # Damped least squares is more stable than a direct inverse near singular poses.
        damping_matrix = (self.damping**2) * np.eye(jacobian.shape[0], dtype=float)
        dq = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping_matrix, error)
        # Per-step clipping prevents huge actuator jumps when the target is far or ill-conditioned.
        dq = np.clip(dq, -self.max_joint_step, self.max_joint_step)
        q_target = self.current_joint_positions() + dq
        return q_target, float(np.linalg.norm(position_error)), float(np.linalg.norm(orientation_error))

    def reached(self, target: IKTarget) -> bool:
        """Check whether the current site pose is inside target tolerances."""

        cup_center = self.tip_position()
        position_error = float(np.linalg.norm(np.asarray(target.position, dtype=float) - cup_center))
        if target.frame_xmat is None:
            _, current_axis = self.suction_axis()
            axis_target = np.asarray(target.cup_axis, dtype=float)
            axis_target /= np.linalg.norm(axis_target)
            orientation_error = float(np.linalg.norm(np.cross(current_axis, axis_target)))
        else:
            current_frame = np.array(self.data.site_xmat[self.tip_site_id], dtype=float).reshape(3, 3)
            target_frame = np.asarray(target.frame_xmat, dtype=float).reshape(3, 3)
            orientation_error_vector = 0.5 * (
                np.cross(current_frame[:, 0], target_frame[:, 0])
                + np.cross(current_frame[:, 1], target_frame[:, 1])
                + np.cross(current_frame[:, 2], target_frame[:, 2])
            )
            orientation_error = float(np.linalg.norm(orientation_error_vector))
        return position_error <= target.pos_tolerance and orientation_error <= target.axis_tolerance
