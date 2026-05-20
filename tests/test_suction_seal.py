from __future__ import annotations

import unittest

import mujoco

from rl_mujoco_picking.suction import SuctionCupParameters, evaluate_suction_seal


def _build_model(xml: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


class SuctionSealTests(unittest.TestCase):
    def test_seals_sphere_when_tip_touches_top(self) -> None:
        model, data = _build_model(
            """
            <mujoco model="sphere_test">
              <worldbody>
                <body name="tool">
                  <site name="handd_tool_seal_tip_site" pos="0 0 0.06"/>
                  <site name="handd_tool_seal_axis_site" pos="0 0 0.072"/>
                </body>
                <body name="obj">
                  <geom name="sphere_geom" type="sphere" pos="0 0 0.03" size="0.03"/>
                </body>
              </worldbody>
            </mujoco>
            """
        )
        result = evaluate_suction_seal(model, data, "sphere_geom")
        self.assertTrue(result.sealable)
        self.assertEqual(result.reason, "seal_candidate")

    def test_rejects_box_when_radially_offset(self) -> None:
        model, data = _build_model(
            """
            <mujoco model="box_offset_test">
              <worldbody>
                <body name="tool">
                  <site name="handd_tool_seal_tip_site" pos="0.05 0 0.05"/>
                  <site name="handd_tool_seal_axis_site" pos="0.05 0 0.062"/>
                </body>
                <body name="obj">
                  <geom name="box_geom" type="box" pos="0 0 0.025" size="0.025 0.025 0.025"/>
                </body>
              </worldbody>
            </mujoco>
            """
        )
        result = evaluate_suction_seal(model, data, "box_geom")
        self.assertFalse(result.sealable)
        self.assertEqual(result.reason, "outside_cup_radius")

    def test_contact_offset_evaluates_the_lip_instead_of_sphere_center(self) -> None:
        model, data = _build_model(
            """
            <mujoco model="contact_offset_test">
              <worldbody>
                <body name="tool">
                  <site name="handd_tool_seal_tip_site" pos="0 0 0.076"/>
                  <site name="handd_tool_seal_axis_site" pos="0 0 0.088"/>
                </body>
                <body name="obj">
                  <geom name="box_geom" type="box" pos="0 0 0.025" size="0.04 0.04 0.025"/>
                </body>
              </worldbody>
            </mujoco>
            """
        )
        without_offset = evaluate_suction_seal(model, data, "box_geom")
        with_offset = evaluate_suction_seal(
            model,
            data,
            "box_geom",
            SuctionCupParameters(contact_offset=0.026),
        )
        self.assertFalse(without_offset.sealable)
        self.assertEqual(without_offset.reason, "too_far_from_surface")
        self.assertTrue(with_offset.sealable)
        self.assertEqual(with_offset.reason, "seal_candidate")
        self.assertLess(with_offset.gap, without_offset.gap)


if __name__ == "__main__":
    unittest.main()
