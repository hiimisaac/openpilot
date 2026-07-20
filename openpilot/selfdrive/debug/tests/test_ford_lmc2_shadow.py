import math

from opendbc.can import CANPacker

from openpilot.cereal import log
from openpilot.selfdrive.debug.ford_lmc2_shadow import (
  decode_lmc2_command,
  Lmc2Command,
  polynomial_state,
  SimpleLmc2Controller,
)
from openpilot.tools.ford_lmc2_shadow_report import summarize_pairs


def test_shadow_uses_c2_alone_for_a_delivered_gentle_curve():
  controller = SimpleLmc2Controller()

  command = controller.update(0.003, 0.003, 15.0, active=True, driver_override=False)

  assert math.isclose(command.curvature, 0.003)
  assert command.path_angle == 0.0
  assert command.path_offset == 0.0
  assert command.curvature_rate == 0.0


def test_shadow_continuously_moves_maneuver_authority_into_c0_c1():
  controller = SimpleLmc2Controller()

  midpoint = controller.update(0.009, 0.009, 7.0, active=True, driver_override=False)
  tight_turn = controller.update(0.02, 0.01, 7.0, active=True, driver_override=False)

  assert math.isclose(midpoint.curvature, 0.0045)
  assert math.isclose(midpoint.path_angle, 0.0045 * 7.0)
  assert math.isclose(midpoint.path_offset, 0.5 * 0.0045 * 7.0 ** 2)
  assert tight_turn.curvature == 0.0
  assert math.isclose(tight_turn.path_angle, 0.02 * 7.0)
  assert math.isclose(tight_turn.path_offset, 0.5 * 0.02 * 7.0 ** 2)


def test_decode_lmc2_command_recovers_every_polynomial_coefficient():
  packer = CANPacker("ford_lincoln_base_pt")
  values = {
    "LatCtlPathOffst_L_Actl": 0.31,
    "LatCtlPath_An_Actl": -0.123,
    "LatCtlCurv_No_Actl": 0.008,
    "LatCtlCrv_NoRate2_Actl": -0.0004,
  }
  _, dat, _ = packer.make_can_msg("LateralMotionControl2", 0, values)

  command = decode_lmc2_command(dat)

  assert math.isclose(command.path_offset, values["LatCtlPathOffst_L_Actl"], abs_tol=1e-9)
  assert math.isclose(command.path_angle, values["LatCtlPath_An_Actl"], abs_tol=1e-9)
  assert math.isclose(command.curvature, values["LatCtlCurv_No_Actl"], abs_tol=1e-9)
  assert math.isclose(command.curvature_rate, values["LatCtlCrv_NoRate2_Actl"], abs_tol=1e-9)


def test_shadow_log_preserves_live_and_shadow_polynomials():
  msg = log.Event.new_message(valid=True)
  msg.init("fordLmc2Shadow")
  msg.fordLmc2Shadow.liveCommand = {
    "pathOffset": 0.1,
    "pathAngle": 0.2,
    "curvature": 0.003,
    "curvatureRate": 0.0004,
  }
  msg.fordLmc2Shadow.shadowCommand = {
    "pathOffset": -0.1,
    "pathAngle": -0.2,
    "curvature": -0.003,
    "curvatureRate": -0.0004,
  }

  assert msg.which() == "fordLmc2Shadow"
  assert math.isclose(msg.fordLmc2Shadow.liveCommand.pathOffset, 0.1, rel_tol=1e-6)
  assert math.isclose(msg.fordLmc2Shadow.shadowCommand.curvatureRate, -0.0004, rel_tol=1e-6)


def test_polynomial_state_compares_equivalent_commands_at_a_distance():
  command = Lmc2Command(path_offset=0.1, path_angle=0.2, curvature=0.03, curvature_rate=0.004)

  path_offset, path_angle, curvature = polynomial_state(command, 5.0)

  assert math.isclose(path_offset, 0.1 + 0.2 * 5.0 + 0.5 * 0.03 * 5.0 ** 2 + 0.004 * 5.0 ** 3 / 6.0)
  assert math.isclose(path_angle, 0.2 + 0.03 * 5.0 + 0.5 * 0.004 * 5.0 ** 2)
  assert math.isclose(curvature, 0.03 + 0.004 * 5.0)


def test_shadow_report_summarizes_coefficients_and_polynomial_shape():
  live = Lmc2Command(path_offset=0.1, path_angle=0.02, curvature=0.003, curvature_rate=0.0001)
  shadow = Lmc2Command(path_offset=0.2, path_angle=0.01, curvature=0.004, curvature_rate=0.0002)

  summary = summarize_pairs([(live, shadow, 0.00005)])

  assert summary["sampleCount"] == 1
  assert math.isclose(summary["coefficients"]["pathOffset"]["mae"], 0.1)
  assert math.isclose(summary["coefficients"]["curvature"]["mae"], 0.001)
  assert summary["polynomial"]["7m"]["pathOffset"]["mae"] > 0.0
  assert math.isclose(summary["computationTimeS"]["mean"], 0.00005)
