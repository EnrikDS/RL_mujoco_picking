from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rl_mujoco_picking.suction import (
    SuctionGraspController,
    initialize_test_seal_robot_pose,
)
from rl_mujoco_picking.visualization import apply_collision_visualization


DEFAULT_SCENE = REPO_ROOT / "models" / "scenes" / "test_seal" / "scene.xml"
ARM_JOINT_NAMES = (
    "handd_shoulder_pan_joint",
    "handd_shoulder_lift_joint",
    "handd_elbow_joint",
    "handd_wrist_1_joint",
    "handd_wrist_2_joint",
    "handd_wrist_3_joint",
)
DEBUG_AXIS_SITE_NAMES = (
    "handd_wrist_3_frame_site",
    "handd_tool_tool_tip_frame_site",
    "handd_tool_gripper_body_frame_site",
    "handd_tool_cup_nominal_site",
    "handd_tool_cup_tip_site",
    "handd_tool_cup_uncompressed_site",
)
SEAL_DEBUG_AXIS_SITE_NAME = "handd_tool_seal_tip_site"
DEBUG_AXIS_BODY_NAMES = (
    "robot_mount",
    "handd_base",
    "handd_shoulder_link",
    "handd_upper_arm_link",
    "handd_forearm_link",
    "handd_wrist_1_link",
    "handd_wrist_2_link",
    "handd_wrist_3_link",
    "handd_tool_mount_frame",
    "handd_tool_roll_frame",
    "handd_tool_tool_tip_link",
    "handd_tool_gripper_body_link",
    "handd_tool_cup_tip_link",
    "handd_tool_cup_uncompressed_link",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed test_seal suction-pick sequence.")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="Path to the test_seal MJCF scene.")
    parser.add_argument("--keyframe", type=str, default="handd_home", help="Robot keyframe name to load before starting.")
    parser.add_argument("--camera", type=str, default=None, help="Optional fixed camera name.")
    parser.add_argument("--headless", action="store_true", help="Run without opening the MuJoCo viewer.")
    parser.add_argument("--debug-waypoints", action="store_true", help="Draw controller Cartesian waypoints as red points.")
    parser.add_argument(
        "--debug-axes",
        action="store_true",
        help="Draw robot link and key tool/site frames. X=red, Y=green, Z=blue.",
    )
    parser.add_argument(
        "--show-collision-geoms",
        action="store_true",
        help="Tint all collidable geoms and reveal invisible collision proxies in the viewer.",
    )
    parser.add_argument(
        "--collision-only",
        action="store_true",
        help="Hide non-collidable visual geoms so only collision solids remain visible.",
    )
    parser.add_argument("--close-when-done", action="store_true", help="Close the viewer when the controller finishes.")
    parser.add_argument("--max-steps", type=int, default=20000, help="Maximum simulation steps before stopping.")
    parser.add_argument(
        "--initial-hold-seconds",
        type=float,
        default=5.0,
        help="Seconds to hold the initial robot pose before starting the picking controller.",
    )
    parser.add_argument(
        "--debug-seal",
        action="store_true",
        help="Print suction seal geometry metrics during descent.",
    )
    parser.add_argument(
        "--debug-seal-interval",
        type=int,
        default=100,
        help="Simulation steps between --debug-seal metric prints.",
    )
    return parser.parse_args()


def reset_scene(model: mujoco.MjModel, data: mujoco.MjData, keyframe_name: str) -> None:
    if keyframe_name:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
        if key_id < 0:
            raise ValueError(f"Keyframe not found in scene: {keyframe_name}")
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    initialize_test_seal_robot_pose(model, data)
    mujoco.mj_forward(model, data)


def current_arm_qpos(model: mujoco.MjModel, data: mujoco.MjData) -> list[float]:
    qpos: list[float] = []
    for joint_name in ARM_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint not found: {joint_name}")
        qpos.append(float(data.qpos[model.jnt_qposadr[joint_id]]))
    return qpos


def hold_initial_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hold_seconds: float,
    viewer: mujoco.viewer.Handle | None = None,
) -> None:
    if hold_seconds <= 0.0:
        return

    hold_qpos = current_arm_qpos(model, data)
    hold_steps = max(1, int(round(hold_seconds / model.opt.timestep)))
    print(f"Holding initial pose for {hold_seconds:.2f}s ({hold_steps} simulation steps).")

    for _ in range(hold_steps):
        if viewer is not None and not viewer.is_running():
            break
        frame_start = time.perf_counter()
        data.ctrl[: len(hold_qpos)] = hold_qpos
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
            remaining = model.opt.timestep - (time.perf_counter() - frame_start)
            if remaining > 0.0:
                time.sleep(remaining)


def maybe_set_camera(viewer: mujoco.viewer.Handle, model: mujoco.MjModel, camera_name: str | None) -> None:
    if camera_name is None:
        return
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise ValueError(f"Camera not found in scene: {camera_name}")
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = camera_id


def draw_debug_waypoints(
    viewer: mujoco.viewer.Handle,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    waypoints: tuple[np.ndarray, ...],
    active_waypoint: np.ndarray | None,
    tcp_trace: list[np.ndarray],
    draw_axes: bool,
) -> None:
    if not hasattr(viewer, "user_scn"):
        return
    scene = viewer.user_scn
    scene.ngeom = 0
    marker_size = np.array((0.018, 0.0, 0.0), dtype=float)
    marker_mat = np.eye(3, dtype=float).reshape(9)
    marker_rgba = np.array((1.0, 0.0, 0.0, 0.95), dtype=float)
    for waypoint in waypoints:
        if scene.ngeom >= scene.maxgeom:
            break
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            marker_size,
            np.asarray(waypoint, dtype=float),
            marker_mat,
            marker_rgba,
        )
        scene.ngeom += 1
    trace_rgba = np.array((0.0, 0.35, 1.0, 0.75), dtype=float)
    for trace_point in tcp_trace[-250:]:
        if scene.ngeom >= scene.maxgeom:
            break
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array((0.010, 0.0, 0.0), dtype=float),
            np.asarray(trace_point, dtype=float),
            marker_mat,
            trace_rgba,
        )
        scene.ngeom += 1
    if active_waypoint is not None and scene.ngeom < scene.maxgeom:
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array((0.026, 0.0, 0.0), dtype=float),
            np.asarray(active_waypoint, dtype=float),
            marker_mat,
            np.array((1.0, 0.85, 0.0, 1.0), dtype=float),
        )
        scene.ngeom += 1
    if draw_axes:
        draw_debug_axes(scene, model, data)


def draw_debug_axes(scene: mujoco.MjvScene, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    axis_colors = (
        np.array((1.0, 0.0, 0.0, 0.95), dtype=float),
        np.array((0.0, 0.9, 0.1, 0.95), dtype=float),
        np.array((0.0, 0.25, 1.0, 0.95), dtype=float),
    )
    if not _draw_seal_tip_axes(scene, model, data, axis_colors):
        return

    body_axis_length = 0.11
    site_axis_length = 0.075
    axis_width = 0.006
    for body_name in DEBUG_AXIS_BODY_NAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            continue
        origin = np.array(data.xpos[body_id], dtype=float)
        frame = np.array(data.xmat[body_id], dtype=float).reshape(3, 3)
        if not _draw_frame_axes(scene, origin, frame, body_axis_length, axis_width, axis_colors):
            return

    for site_name in DEBUG_AXIS_SITE_NAMES:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            continue
        origin = np.array(data.site_xpos[site_id], dtype=float)
        frame = np.array(data.site_xmat[site_id], dtype=float).reshape(3, 3)
        if not _draw_frame_axes(scene, origin, frame, site_axis_length, axis_width * 0.75, axis_colors):
            return



def _draw_seal_tip_axes(
    scene: mujoco.MjvScene,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    axis_colors: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> bool:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SEAL_DEBUG_AXIS_SITE_NAME)
    if site_id < 0:
        return True
    origin = np.array(data.site_xpos[site_id], dtype=float)
    frame = np.array(data.site_xmat[site_id], dtype=float).reshape(3, 3)
    if scene.ngeom >= scene.maxgeom:
        return False
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array((0.026, 0.0, 0.0), dtype=float),
        origin,
        np.eye(3, dtype=float).reshape(9),
        np.array((1.0, 1.0, 1.0, 1.0), dtype=float),
    )
    scene.ngeom += 1
    return _draw_frame_axes(scene, origin, frame, 0.22, 0.016, axis_colors)


def _draw_frame_axes(
    scene: mujoco.MjvScene,
    origin: np.ndarray,
    frame: np.ndarray,
    axis_length: float,
    axis_width: float,
    axis_colors: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> bool:
    for axis_index, rgba in enumerate(axis_colors):
        if scene.ngeom >= scene.maxgeom:
            return False
        endpoint = origin + axis_length * frame[:, axis_index]
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3, dtype=float),
            np.zeros(3, dtype=float),
            np.eye(3, dtype=float).reshape(9),
            rgba,
        )
        mujoco.mjv_connector(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_ARROW,
            axis_width,
            origin,
            endpoint,
        )
        scene.geoms[scene.ngeom].rgba[:] = rgba
        scene.ngeom += 1
    return True


def _format_seal_debug(controller: SuctionGraspController) -> str:
    evaluation = controller.current_seal_evaluation()
    if evaluation is None:
        return "seal=none"
    return (
        f"seal={evaluation.reason} "
        f"gap={evaluation.gap:.4f} "
        f"radial={evaluation.radial_offset:.4f} "
        f"align={evaluation.normal_alignment:.3f} "
        f"axial={evaluation.axial_offset:.4f}"
    )


def run_headless(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    max_steps: int,
    initial_hold_seconds: float,
    debug_seal: bool,
    debug_seal_interval: int,
) -> None:
    hold_initial_pose(model, data, initial_hold_seconds)
    controller = SuctionGraspController.from_test_seal_scene(model, data)
    previous_signature = None
    seal_interval = max(1, debug_seal_interval)
    for step in range(max_steps):
        status = controller.step()
        signature = (status.current_target, status.phase, status.attached_target)
        if signature != previous_signature:
            contact_text = "" if status.last_contact is None else f" contact={status.last_contact}"
            print(f"step={step} target={status.current_target} phase={status.phase} attached={status.attached_target} reason={status.last_seal_reason}{contact_text}")
            previous_signature = signature
        if debug_seal and status.phase == "descend" and step % seal_interval == 0:
            print(f"step={step} {_format_seal_debug(controller)}")
        mujoco.mj_step(model, data)
        if status.phase in ("done", "collision"):
            break
    final_status = controller.status()
    print(
        "Completed:",
        final_status.completed_targets,
        "Failed:",
        final_status.failed_targets,
        "Phase:",
        final_status.phase,
    )


def run_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str | None,
    max_steps: int,
    initial_hold_seconds: float,
    debug_waypoints: bool,
    debug_axes: bool,
    show_collision_geoms: bool,
    close_when_done: bool,
    debug_seal: bool,
    debug_seal_interval: int,
) -> None:
    previous_signature = None
    tcp_trace: list[np.ndarray] = []
    seal_interval = max(1, debug_seal_interval)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        if show_collision_geoms:
            viewer.opt.geomgroup[:] = 1
        maybe_set_camera(viewer, model, camera_name)
        hold_initial_pose(model, data, initial_hold_seconds, viewer)
        controller = SuctionGraspController.from_test_seal_scene(model, data)
        for step in range(max_steps):
            if not viewer.is_running():
                break
            frame_start = time.perf_counter()
            status = controller.step()
            signature = (status.current_target, status.phase, status.attached_target)
            if signature != previous_signature:
                waypoint_error = controller.active_waypoint_error()
                orientation_error = controller.active_waypoint_orientation_error()
                waypoint_text = "" if waypoint_error is None else f" waypoint_error={waypoint_error:.4f}"
                orientation_text = "" if orientation_error is None else f" orientation_error={orientation_error:.4f}"
                contact_text = "" if status.last_contact is None else f" contact={status.last_contact}"
                print(f"step={step} target={status.current_target} phase={status.phase} attached={status.attached_target} reason={status.last_seal_reason}{waypoint_text}{orientation_text}{contact_text}")
                previous_signature = signature
            if debug_seal and status.phase == "descend" and step % seal_interval == 0:
                print(f"step={step} {_format_seal_debug(controller)}")
            mujoco.mj_step(model, data)
            if debug_waypoints or debug_axes:
                tcp_trace.append(controller.trajectory_position())
                draw_debug_waypoints(
                    viewer,
                    model,
                    data,
                    controller.debug_waypoint_positions(),
                    controller.active_waypoint_position(),
                    tcp_trace,
                    debug_axes,
                )
            viewer.sync()
            if status.phase == "done" and close_when_done:
                break
            remaining = model.opt.timestep - (time.perf_counter() - frame_start)
            if remaining > 0.0:
                time.sleep(remaining)
        final_status = controller.status()
        print(
            "Completed:",
            final_status.completed_targets,
            "Failed:",
            final_status.failed_targets,
            "Phase:",
            final_status.phase,
        )
        if not close_when_done:
            print("Viewer kept open for inspection. Close the window to exit.")
            while viewer.is_running():
                if debug_waypoints or debug_axes:
                    tcp_trace.append(controller.trajectory_position())
                    draw_debug_waypoints(
                        viewer,
                        model,
                        data,
                        controller.debug_waypoint_positions(),
                        controller.active_waypoint_position(),
                        tcp_trace,
                        debug_axes,
                    )
                viewer.sync()
                time.sleep(model.opt.timestep)


def main() -> None:
    args = parse_args()
    scene_path = args.scene.resolve()
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_path}")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    reset_scene(model, data, args.keyframe)
    if args.show_collision_geoms or args.collision_only:
        summary = apply_collision_visualization(model, collision_only=args.collision_only)
        print(
            "Collision visualization: "
            f"collidable={summary.collidable_geoms}, noncollidable={summary.noncollidable_geoms}, "
            f"collision_only={args.collision_only}"
        )

    if args.headless:
        run_headless(
            model,
            data,
            args.max_steps,
            args.initial_hold_seconds,
            args.debug_seal,
            args.debug_seal_interval,
        )
    else:
        run_viewer(
            model,
            data,
            args.camera,
            args.max_steps,
            args.initial_hold_seconds,
            args.debug_waypoints,
            args.debug_axes,
            args.show_collision_geoms or args.collision_only,
            args.close_when_done,
            args.debug_seal,
            args.debug_seal_interval,
        )


if __name__ == "__main__":
    main()
