from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rl_mujoco_picking.suction import (
    SuctionGraspController,
    initialize_test_seal_robot_pose,
)


DEFAULT_SCENE = REPO_ROOT / "models" / "scenes" / "test_seal" / "scene.xml"
ARM_JOINT_NAMES = (
    "handd_shoulder_pan_joint",
    "handd_shoulder_lift_joint",
    "handd_elbow_joint",
    "handd_wrist_1_joint",
    "handd_wrist_2_joint",
    "handd_wrist_3_joint",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed test_seal suction-pick sequence.")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="Path to the test_seal MJCF scene.")
    parser.add_argument("--keyframe", type=str, default="handd_home", help="Robot keyframe name to load before starting.")
    parser.add_argument("--camera", type=str, default=None, help="Optional fixed camera name.")
    parser.add_argument("--headless", action="store_true", help="Run without opening the MuJoCo viewer.")
    parser.add_argument("--max-steps", type=int, default=20000, help="Maximum simulation steps before stopping.")
    parser.add_argument(
        "--initial-hold-seconds",
        type=float,
        default=5.0,
        help="Seconds to hold the initial robot pose before starting the picking controller.",
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


def run_headless(model: mujoco.MjModel, data: mujoco.MjData, max_steps: int, initial_hold_seconds: float) -> None:
    hold_initial_pose(model, data, initial_hold_seconds)
    controller = SuctionGraspController.from_test_seal_scene(model, data)
    previous_signature = None
    for step in range(max_steps):
        status = controller.step()
        signature = (status.current_target, status.phase, status.attached_target)
        if signature != previous_signature:
            print(f"step={step} target={status.current_target} phase={status.phase} attached={status.attached_target} reason={status.last_seal_reason}")
            previous_signature = signature
        mujoco.mj_step(model, data)
        if status.phase == "done":
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
) -> None:
    previous_signature = None
    with mujoco.viewer.launch_passive(model, data) as viewer:
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
                print(f"step={step} target={status.current_target} phase={status.phase} attached={status.attached_target} reason={status.last_seal_reason}")
                previous_signature = signature
            mujoco.mj_step(model, data)
            viewer.sync()
            if status.phase == "done":
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


def main() -> None:
    args = parse_args()
    scene_path = args.scene.resolve()
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_path}")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    reset_scene(model, data, args.keyframe)

    if args.headless:
        run_headless(model, data, args.max_steps, args.initial_hold_seconds)
    else:
        run_viewer(model, data, args.camera, args.max_steps, args.initial_hold_seconds)


if __name__ == "__main__":
    main()
