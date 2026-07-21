import math

from opendbc.can import CANPacker
from openpilot.cereal import log
from openpilot.selfdrive.debug.ford_lmc2_shadow import (
  decode_lmc2_command,
  Lmc2Command,
  polynomial_state,
)
from openpilot.tools.ford_lmc2_shadow_report import summarize_pairs, timing_is_coherent


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
  msg.fordLmc2Shadow.shadowValid = True
  msg.fordLmc2Shadow.modelMonoTime = 123456789
  msg.fordLmc2Shadow.carControlMonoTime = 123456999
  msg.fordLmc2Shadow.sendcanMonoTime = 123457999
  msg.fordLmc2Shadow.projectedCurvature = 0.002
  msg.fordLmc2Shadow.projectedSteeringAngleDeg = -12.0
  msg.fordLmc2Shadow.desiredAngleCurvature = 0.004
  msg.fordLmc2Shadow.previewConflictShare = 0.75
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
  assert msg.fordLmc2Shadow.shadowValid
  assert msg.fordLmc2Shadow.modelMonoTime == 123456789
  assert msg.fordLmc2Shadow.carControlMonoTime == 123456999
  assert msg.fordLmc2Shadow.sendcanMonoTime == 123457999
  assert math.isclose(msg.fordLmc2Shadow.projectedCurvature, 0.002, rel_tol=1e-6)
  assert math.isclose(msg.fordLmc2Shadow.projectedSteeringAngleDeg, -12.0, rel_tol=1e-6)
  assert math.isclose(msg.fordLmc2Shadow.desiredAngleCurvature, 0.004, rel_tol=1e-6)
  assert math.isclose(msg.fordLmc2Shadow.previewConflictShare, 0.75, rel_tol=1e-6)
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


def test_shadow_report_keeps_signed_bias_separate_from_absolute_error():
  live = Lmc2Command(path_offset=0.2)
  shadow = Lmc2Command(path_offset=0.1)

  summary = summarize_pairs([(live, shadow, 0.0)])

  assert math.isclose(summary["coefficients"]["pathOffset"]["mean"], -0.1)
  assert math.isclose(summary["coefficients"]["pathOffset"]["mae"], 0.1)


def test_shadow_report_includes_model_and_transport_age():
  command = Lmc2Command()

  summary = summarize_pairs([(command, command, 0.00005, 0.02, 0.01)])

  assert math.isclose(summary["modelToControlAgeS"]["mean"], 0.02)
  assert math.isclose(summary["controlToCanAgeS"]["mean"], 0.01)


def test_shadow_report_rejects_future_or_stale_inputs():
  assert timing_is_coherent(0.02, 0.01)
  assert not timing_is_coherent(-0.001, 0.01)
  assert not timing_is_coherent(0.101, 0.01)
  assert not timing_is_coherent(0.02, -0.001)
  assert not timing_is_coherent(0.02, 0.031)
