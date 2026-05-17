from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import mujoco


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.handd_startup_randomization import (
    StartupDropConfig,
    build_runtime_startup_scene,
    restore_spawn_state,
    settle_objects,
    write_settled_scene,
)


BASE_SCENE = REPO_ROOT / "models" / "scenes" / "handd_simple_ur_test_scene" / "scene.xml"
GENERATED_DIR = BASE_SCENE.parent / "generated"
DEFAULT_OUTPUT = GENERATED_DIR / "initialized_random_grocery_scene.xml"
DEFAULT_METADATA = GENERATED_DIR / "initialized_random_grocery_scene.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug wrapper that settles the HandD startup-drop objects and exports a static scene."
    )
    parser.add_argument("--scene", type=Path, default=BASE_SCENE, help="Base MJCF scene to augment.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output path for the settled MJCF scene.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA, help="Optional JSON metadata output.")
    parser.add_argument("--num-objects", type=int, default=12, help="Number of random grocery objects to launch.")
    parser.add_argument("--min-size", type=float, default=0.03, help="Minimum linear grocery size in meters.")
    parser.add_argument("--max-size", type=float, default=0.08, help="Maximum linear grocery size in meters.")
    parser.add_argument("--seed", type=int, default=None, help="Optional fixed RNG seed.")
    parser.add_argument(
        "--keyframe",
        type=str,
        default="handd_home",
        help="Optional robot keyframe to load before the drop. Use an empty string to skip.",
    )
    parser.add_argument("--view", action="store_true", help="Open the exported static scene in the viewer afterwards.")
    return parser.parse_args()


def write_metadata(metadata_path: Path, args: argparse.Namespace, result) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scene": str(args.scene),
        "seed": result.seed,
        "num_launched": args.num_objects,
        "min_size": args.min_size,
        "max_size": args.max_size,
        "settle_steps_run": result.settle_steps_run,
        "kept_objects": [
            {
                "name": placement.spec.name,
                "shape": placement.spec.shape,
                "half_size": list(placement.spec.size),
                "final_pos": list(placement.final_pos),
                "final_quat": list(placement.final_quat),
                "aabb_min": list(placement.aabb_min),
                "aabb_max": list(placement.aabb_max),
            }
            for placement in result.kept_objects
        ],
        "ignored_objects": [
            {
                "name": placement.spec.name,
                "shape": placement.spec.shape,
                "half_size": list(placement.spec.size),
                "final_pos": list(placement.final_pos),
                "final_quat": list(placement.final_quat),
                "aabb_min": list(placement.aabb_min),
                "aabb_max": list(placement.aabb_max),
            }
            for placement in result.ignored_objects
        ],
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def open_viewer(scene_path: Path, keyframe: str) -> None:
    command = [sys.executable, str(REPO_ROOT / "scripts" / "view_scene.py"), "--scene", str(scene_path), "--startup-drop", "off"]
    if keyframe != "":
        command.extend(["--keyframe", keyframe])
    else:
        command.extend(["--keyframe", ""])
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def main() -> None:
    args = parse_args()
    runtime_scene = build_runtime_startup_scene(
        args.scene.resolve(),
        StartupDropConfig(
            num_objects=args.num_objects,
            min_size=args.min_size,
            max_size=args.max_size,
            seed=args.seed,
        ),
    )

    model = mujoco.MjModel.from_xml_path(str(runtime_scene.temp_scene_path))
    runtime_scene.cleanup()
    data = mujoco.MjData(model)

    if args.keyframe:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe)
        if key_id < 0:
            raise ValueError(f"Keyframe not found in scene: {args.keyframe}")
        mujoco.mj_resetDataKeyframe(model, data, key_id)

    restore_spawn_state(model, data, runtime_scene)
    result = settle_objects(model, data, runtime_scene, viewer=None, realtime=False)

    write_settled_scene(args.scene.resolve(), args.output.resolve(), result.kept_objects)
    write_metadata(args.metadata.resolve(), args, result)

    print(f"Generated settled scene: {args.output}")
    print(
        f"Startup drop seed={result.seed}, kept={len(result.kept_objects)}, "
        f"ignored={len(result.ignored_objects)}, steps={result.settle_steps_run}"
    )
    print(f"Metadata written to: {args.metadata}")

    if args.view:
        open_viewer(args.output.resolve(), args.keyframe)


if __name__ == "__main__":
    main()
