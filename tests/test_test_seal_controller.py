from __future__ import annotations

import unittest
from pathlib import Path

import mujoco
import numpy as np

from rl_mujoco_picking.suction import (
    SuctionGraspController,
    initialize_test_seal_robot_pose,
)
from rl_mujoco_picking.suction.seal import SuctionCupParameters, suction_axis


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_SCENE = REPO_ROOT / "models" / "scenes" / "test_seal" / "scene.xml"


class TestSealControllerTests(unittest.TestCase):
    def test_scene_compiles_and_targets_are_discoverable(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(TEST_SCENE))
        data = mujoco.MjData(model)
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "handd_home")
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        initialize_test_seal_robot_pose(model, data)
        mujoco.mj_forward(model, data)

        controller = SuctionGraspController.from_test_seal_scene(model, data)
        status = controller.status()
        self.assertEqual(status.current_target, "seal_rect_long")
        self.assertEqual(len(controller.targets), 4)
        self.assertEqual(model.neq, 4)
        compliance_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "handd_tool_cup_compliance_joint")
        self.assertGreaterEqual(compliance_joint, 0)
        np.testing.assert_allclose(model.jnt_range[compliance_joint], (0.0, 0.10), atol=1e-9)

    def test_robot_pose_and_object_sizes_match_spec(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(TEST_SCENE))
        data = mujoco.MjData(model)
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "handd_home")
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        initialize_test_seal_robot_pose(model, data)
        mujoco.mj_forward(model, data)

        expected_dims = {
            "seal_sphere_small_geom": (0.10, 0.10, 0.10),
            "seal_cube_medium_geom": (0.11, 0.11, 0.11),
            "seal_rect_long_geom": (0.15, 0.10, 0.12),
            "seal_cube_tall_geom": (0.10, 0.10, 0.15),
        }
        for geom_name, expected in expected_dims.items():
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE:
                dims = (2 * model.geom_size[geom_id][0],) * 3
            else:
                dims = tuple(2 * float(model.geom_size[geom_id][axis]) for axis in range(3))
            self.assertEqual(tuple(round(value, 3) for value in dims), expected)

        self.assertGreater(np.linalg.norm(model.opt.gravity), 0.0)

        tool_tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "handd_tool_seal_tip_site")
        tote_rim_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "storage_tote_rim_site")
        tool_tip = np.array(data.site_xpos[tool_tip_site_id], dtype=float)
        tote_rim = np.array(data.site_xpos[tote_rim_site_id], dtype=float)
        self.assertGreater(tool_tip[2], tote_rim[2] + 0.2)
        np.testing.assert_allclose(data.ctrl[:6], data.qpos[:6], atol=1e-6)

    def test_controller_updates_actuator_targets_and_moves_under_gravity(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(TEST_SCENE))
        data = mujoco.MjData(model)
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "handd_home")
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        initialize_test_seal_robot_pose(model, data)
        mujoco.mj_forward(model, data)

        controller = SuctionGraspController.from_test_seal_scene(model, data)
        params = SuctionCupParameters()
        initial_ctrl = np.array(data.ctrl[:6], dtype=float)
        initial_q = np.array(data.qpos[:6], dtype=float)
        previous_tip = np.array(suction_axis(model, data, params)[0], dtype=float)
        for _ in range(200):
            controller.step()
            mujoco.mj_step(model, data)
        current_tip = np.array(suction_axis(model, data, params)[0], dtype=float)
        self.assertFalse(np.allclose(data.ctrl[:6], initial_ctrl))
        self.assertFalse(np.allclose(data.qpos[:6], initial_q))
        self.assertGreater(np.linalg.norm(current_tip - previous_tip), 1e-3)
        self.assertEqual(controller.status().phase, "approach")

    def test_robot_avoids_environment_contacts_during_initial_approach(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(TEST_SCENE))
        data = mujoco.MjData(model)
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "handd_home")
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        initialize_test_seal_robot_pose(model, data)
        mujoco.mj_forward(model, data)

        controller = SuctionGraspController.from_test_seal_scene(model, data)
        for _ in range(1500):
            controller.step()
            mujoco.mj_step(model, data)
            for contact_id in range(data.ncon):
                contact = data.contact[contact_id]
                body1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom1]) or ""
                body2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom2]) or ""
                robot_involved = body1.startswith("handd_") or body2.startswith("handd_")
                self_collision = body1.startswith("handd_") and body2.startswith("handd_")
                target_contact = body1.startswith("seal_") or body2.startswith("seal_")
                tote_proxy_contact = body1 == "storage_tote_proxy" or body2 == "storage_tote_proxy"
                if robot_involved and not self_collision and not target_contact and not tote_proxy_contact:
                    self.fail(f"Unexpected robot contact during approach: {body1} vs {body2}")

    def test_debug_waypoints_are_spaced_at_fifty_millimeters_or_less(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(TEST_SCENE))
        data = mujoco.MjData(model)
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "handd_home")
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        initialize_test_seal_robot_pose(model, data)
        mujoco.mj_forward(model, data)

        controller = SuctionGraspController.from_test_seal_scene(model, data)
        target = controller._current_target()
        self.assertIsNotNone(target)
        controller._ensure_phase_plan(target, "approach")
        waypoints = np.array(controller.debug_waypoint_positions())
        self.assertGreater(len(waypoints), 1)
        spacing = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        self.assertLessEqual(float(np.max(spacing)), controller.config.waypoint_spacing + 1e-9)


if __name__ == "__main__":
    unittest.main()
