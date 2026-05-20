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
        self.assertEqual(controller.config.trajectory_site_name, "handd_tool_seal_tip_site")
        self.assertEqual(controller.ik_solver.tip_site_name, controller.config.trajectory_site_name)
        self.assertEqual(model.neq, 4)
        compliance_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "handd_tool_cup_compliance_joint")
        self.assertGreaterEqual(compliance_joint, 0)
        np.testing.assert_allclose(model.jnt_range[compliance_joint], (0.0, 0.04), atol=1e-9)
        self.assertAlmostEqual(float(model.jnt_stiffness[compliance_joint]), 100.0)
        self.assertAlmostEqual(float(model.dof_damping[model.jnt_dofadr[compliance_joint]]), 1.8)
        delivery_drop_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "delivery_tote_1_drop_site")
        self.assertGreaterEqual(delivery_drop_site, 0)

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

        cup_tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "handd_tool_cup_tip_site")
        cup_lip_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "handd_tool_cup_lip_collision")
        np.testing.assert_allclose(data.site_xpos[cup_tip_site_id], tool_tip, atol=1e-9)
        np.testing.assert_allclose(data.geom_xpos[cup_lip_geom_id], tool_tip, atol=1e-9)

    def test_tool_contact_exclusions_are_limited_to_final_wrist_links(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(TEST_SCENE))
        excluded_pairs = set()
        for exclude_id in range(model.nexclude):
            signature = int(model.exclude_signature[exclude_id])
            body1_id = signature >> 16
            body2_id = signature & 0xFFFF
            body1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
            body2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
            excluded_pairs.add(frozenset((body1, body2)))

        expected_tool_bodies = (
            "handd_tool_tool_tip_link",
            "handd_tool_gripper_body_link",
            "handd_tool_cup_tip_link",
            "handd_tool_cup_uncompressed_link",
        )
        for wrist_body in ("handd_wrist_1_link", "handd_wrist_2_link", "handd_wrist_3_link"):
            for tool_body in expected_tool_bodies:
                self.assertIn(frozenset((wrist_body, tool_body)), excluded_pairs)

        proximal_robot_bodies = (
            "handd_shoulder_link",
            "handd_upper_arm_link",
            "handd_forearm_link",
        )
        for robot_body in proximal_robot_bodies:
            for tool_body in expected_tool_bodies:
                self.assertNotIn(frozenset((robot_body, tool_body)), excluded_pairs)

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

    def test_debug_waypoints_are_spaced_at_thirty_millimeters_or_less(self) -> None:
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

    def test_approach_waypoints_target_the_physical_seal_tip(self) -> None:
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

        _, _, pick_site_id, _ = controller._target_ids(target)
        pick_position = np.array(data.site_xpos[pick_site_id], dtype=float)
        final_waypoint = controller.debug_waypoint_positions()[-1]
        np.testing.assert_allclose(final_waypoint[:2], pick_position[:2], atol=1e-6)

        seal_tip_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "handd_tool_seal_tip_site")
        planning_data = controller.planning_data
        planning_data.qpos[controller.ik_solver.qpos_indices] = controller.plan_qpos[-1]
        mujoco.mj_forward(model, planning_data)
        planned_seal_tip = np.array(planning_data.site_xpos[seal_tip_site], dtype=float)
        np.testing.assert_allclose(planned_seal_tip, final_waypoint, atol=0.002)


if __name__ == "__main__":
    unittest.main()
