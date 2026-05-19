# RL_mujoco_picking

RL algorithms for robotic picking using MuJoCo.

## Current baseline

The project now contains a **HandD-derived UR10e scene** imported from your Toshiba drive.

It is split into separate layers so you can edit them independently:

- robot: `models/robots/handd_ur10e/ur10e.xml`
- suction tool: `models/tools/picking_station_tool/picking_station_tool.xml`
- robot + tool assembly: `models/assemblies/handd_ur10e_picking_station.xml`
- environment scene: `models/scenes/handd_simple_ur_test_scene/scene.xml`

The imported environment uses the geometry and placements from:

- `simple_ur_test_scene/collision_scene/scene.yaml`
- `simple_ur_test_scene/meshes/*.STL`

The suction cup geometry and TCP offsets come from:

- the picking-station tool assets imported into `models/tools/picking_station_tool/`

## Project layout

```text
RL_mujoco_picking/
├── envs/
├── models/
│   ├── assemblies/
│   │   └── handd_ur10e_picking_station.xml
│   ├── basic_pick_scene.xml
│   ├── robots/
│   │   ├── handd_ur10e/
│   │   └── ur10e/
│   ├── scenes/
│   │   ├── handd_simple_ur_test_scene/
│   │   └── ur10e_workcell.xml
│   └── tools/
│       └── picking_station_tool/
├── scripts/
│   ├── initialize_handd_random_grocery_scene.py
│   └── view_scene.py
└── third_party/
```

## What To Edit

If you want to tweak the imported UR10e:

- `models/robots/handd_ur10e/ur10e.xml`

If you want to tweak the suction tool:

- `models/tools/picking_station_tool/picking_station_tool.xml`

If you want to tweak how the tool is mounted to the robot:

- `models/assemblies/handd_ur10e_picking_station.xml`

If you want to tweak the HandD workcell:

- `models/scenes/handd_simple_ur_test_scene/scene.xml`

If you want to tweak the grocery-object initialization routine:

- `scripts/initialize_handd_random_grocery_scene.py`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Open the imported HandD scene

```bash
python scripts/view_scene.py
```

By default, the viewer now:

- opens `models/scenes/handd_simple_ur_test_scene/scene.xml`
- resets the robot to the attached keyframe `handd_home`
- uses MuJoCo's free camera

If you want a fixed camera:

```bash
python scripts/view_scene.py --camera overview
```

If you want to skip keyframe reset:

```bash
python scripts/view_scene.py --keyframe ""
```

If you want the main reference frames overlaid:

```bash
python scripts/view_scene.py --frames site
```

## Random Grocery Initialization

You can generate a settled randomized scene with grocery-sized objects dropped into the two equal delivery totes.

Current assumption:

- the two target containers are `delivery_tote_1` and `delivery_tote_2`
- objects outside those two totes are discarded from the final generated scene

Example:

```bash
python scripts/initialize_handd_random_grocery_scene.py \
  --num-objects 12 \
  --min-size 0.03 \
  --max-size 0.09 \
  --seed 7
```

This writes:

- `models/scenes/handd_simple_ur_test_scene/generated/initialized_random_grocery_scene.xml`
- `models/scenes/handd_simple_ur_test_scene/generated/initialized_random_grocery_scene.json`

You can open the generated scene with:

```bash
python scripts/view_scene.py \
  --scene models/scenes/handd_simple_ur_test_scene/generated/initialized_random_grocery_scene.xml \
  --keyframe ""
```

Or generate and open in one go:

```bash
python scripts/initialize_handd_random_grocery_scene.py --num-objects 12 --view
```

## Notes

The scene attaches the full robot assembly with the prefix `handd_`, so names imported from the robot are prefixed in the compiled model. For example:

- keyframe: `handd_home`
- wrist body: `handd_wrist_3_link`
- cup tip body: `handd_tool_cup_tip_link`
- cup uncompressed body: `handd_tool_cup_uncompressed_link`

That gives us a clean base for adding suction logic and later wrapping the task as a Gymnasium environment.
