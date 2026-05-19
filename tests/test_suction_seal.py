from __future__ import annotations

import unittest

import mujoco

from rl_mujoco_picking.suction import evaluate_suction_seal


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


if __name__ == "__main__":
    unittest.main()
