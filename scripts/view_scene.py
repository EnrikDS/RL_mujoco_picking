from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from envs.handd_startup_randomization import (
    StartupDropConfig,
    build_runtime_startup_scene,
    restore_spawn_state,
    settle_objects,
)
from rl_mujoco_picking.visualization import apply_collision_visualization

DEFAULT_SCENE = REPO_ROOT / "models" / "scenes" / "handd_simple_ur_test_scene" / "scene.xml"
FRAME_MODE_MAP = {
    "none": mujoco.mjtFrame.mjFRAME_NONE,
    "body": mujoco.mjtFrame.mjFRAME_BODY,
    "site": mujoco.mjtFrame.mjFRAME_SITE,
    "geom": mujoco.mjtFrame.mjFRAME_GEOM,
    "world": mujoco.mjtFrame.mjFRAME_WORLD,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open a MuJoCo scene in the native viewer.")
    parser.add_argument(
        "--scene",
        type=Path,
        default=DEFAULT_SCENE,
        help="Path to an MJCF scene file.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Optional fixed camera name. Leave unset to use MuJoCo's free camera.",
    )
    parser.add_argument(
        "--keyframe",
        type=str,
        default="handd_home",
        help="Optional keyframe name to load before launching the viewer. Use an empty string to skip.",
    )
    parser.add_argument(
        "--frames",
        choices=tuple(FRAME_MODE_MAP),
        default="site",
        help="Reference-frame overlay mode. 'site' is useful for the main robot/tool frames.",
    )
    parser.add_argument(
        "--startup-drop",
        choices=("on", "off"),
        default="on",
        help="Drop random grocery objects into storage_tote at startup before normal interaction.",
    )
    parser.add_argument(
        "--num-objects",
        type=int,
        default=16,
        help="Number of random grocery objects to launch during startup drop.",
    )
    parser.add_argument(
        "--min-size",
        type=float,
        default=0.08,
        help="Minimum linear grocery size in meters.",
    )
    parser.add_argument(
        "--max-size",
        type=float,
        default=0.15,
        help="Maximum linear grocery size in meters.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible startup drops. Omit for a fresh random scene each run.",
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
    return parser.parse_args()


def step_realtime(model: mujoco.MjModel, data: mujoco.MjData, viewer: mujoco.viewer.Handle) -> None:
    frame_start = time.perf_counter()
    mujoco.mj_step(model, data)
    viewer.sync()
    remaining = model.opt.timestep - (time.perf_counter() - frame_start)
    if remaining > 0.0:
        time.sleep(remaining)


def main() -> None:
    args = parse_args()
    scene_path = args.scene.resolve()

    if not scene_path.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_path}")

    runtime_scene = None
    if args.startup_drop == "on":
        runtime_scene = build_runtime_startup_scene(
            scene_path,
            StartupDropConfig(
                num_objects=args.num_objects,
                min_size=args.min_size,
                max_size=args.max_size,
                seed=args.seed,
            ),
        )
        model = mujoco.MjModel.from_xml_path(str(runtime_scene.temp_scene_path))
        runtime_scene.cleanup()
    else:
        model = mujoco.MjModel.from_xml_path(str(scene_path))

    data = mujoco.MjData(model)
    if args.keyframe:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe)
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
        else:
            raise ValueError(f"Keyframe not found in scene: {args.keyframe}")
    if runtime_scene is not None:
        restore_spawn_state(model, data, runtime_scene)
    if args.show_collision_geoms or args.collision_only:
        summary = apply_collision_visualization(model, collision_only=args.collision_only)
        print(
            "Collision visualization: "
            f"collidable={summary.collidable_geoms}, noncollidable={summary.noncollidable_geoms}, "
            f"collision_only={args.collision_only}"
        )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        if args.show_collision_geoms or args.collision_only:
            viewer.opt.geomgroup[:] = 1
        viewer.opt.frame = FRAME_MODE_MAP[args.frames]

        if args.camera is not None:
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
            if camera_id < 0:
                raise ValueError(f"Camera not found in scene: {args.camera}")
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = camera_id

        if runtime_scene is not None:
            result = settle_objects(model, data, runtime_scene, viewer=viewer, realtime=True)
            print(
                "Startup drop complete: "
                f"seed={result.seed}, kept={len(result.kept_objects)}, "
                f"ignored={len(result.ignored_objects)}, steps={result.settle_steps_run}"
            )

        while viewer.is_running():
            step_realtime(model, data, viewer)


if __name__ == "__main__":
    main()
