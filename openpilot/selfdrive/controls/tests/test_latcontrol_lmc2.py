import math
from types import SimpleNamespace

from openpilot.selfdrive.controls.lib.latcontrol_lmc2 import (
  LMC2Params, lmc2_gains, lmc2_step, path_command,
)
from openpilot.selfdrive.controls.lib.lmc2_path import LMCStates


def test_gains_match_059_16_19():
  p = LMC2Params()
  v = 20.0
  k_y, k_psi = lmc2_gains(v, p)
  omega_n = math.sqrt(18.5) * p.zeta / p.t_r  # 059 (18)+(19); 4.3 is the rounded form
  assert math.isclose(k_y, 18.5 * p.zeta ** 2 / (p.t_r ** 2 * v ** 2), rel_tol=1e-12)
  assert math.isclose(k_y, omega_n ** 2 / v ** 2, rel_tol=1e-12)
  assert math.isclose(k_psi, 2 * p.zeta * omega_n / v, rel_tol=1e-3)  # 8.6 vs 2*sqrt(18.5)
  assert math.isclose(k_psi, 8.6 * p.zeta ** 2 / (p.t_r * v), rel_tol=1e-12)
  # ζ² — not the dropped-square 8.6 ζ /(T_r V)
  assert not math.isclose(k_psi, 8.6 * p.zeta / (p.t_r * v), rel_tol=1e-3)


def test_gains_scale_with_speed():
  p = LMC2Params()
  k_y_lo, k_psi_lo = lmc2_gains(10.0, p)
  k_y_hi, k_psi_hi = lmc2_gains(20.0, p)
  assert math.isclose(k_y_lo / k_y_hi, 4.0, rel_tol=1e-9)
  assert math.isclose(k_psi_lo / k_psi_hi, 2.0, rel_tol=1e-9)


def test_feedback_limiter_does_not_clip_ff_only_high_kappa():
  p = LMC2Params()
  meas = LMCStates(True, 0.0, 0.0, 0.018, 0.0)
  kappa_cmd, kappa_fb, kappa_ff, _ = lmc2_step(meas, 15.0, p, 0.0, 0.01, True)
  assert math.isclose(kappa_fb, 0.0, abs_tol=1e-12)
  assert math.isclose(kappa_cmd, 0.018, rel_tol=1e-9)
  assert math.isclose(kappa_ff, 0.018, rel_tol=1e-9)


def test_inactive_zeros():
  p = LMC2Params()
  meas = LMCStates(True, 0.3, 0.05, 0.01, 0.0002)
  kappa_cmd, kappa_fb, _, _ = lmc2_step(meas, 15.0, p, 0.1, 0.01, False)
  assert kappa_cmd == 0.0
  assert kappa_fb == 0.0
  pc = path_command(meas, 0.012, False)
  assert not pc.valid
  assert pc.pathOffset == 0.0
  assert pc.curvature == 0.0


def test_path_command_is_commanded_cubic():
  meas = LMCStates(True, 0.3, 0.05, 0.004, 0.0002)
  kappa_cmd = 0.012
  pc = path_command(meas, kappa_cmd, True)
  assert pc.valid
  assert pc.pathOffset == 0.3
  assert pc.pathAngle == 0.05
  assert pc.curvature == kappa_cmd
  assert pc.curvatureRate == 0.0002


def test_meas_none_is_invalid():
  pc = path_command(None, 0.01, True)
  assert not pc.valid
  assert pc.curvature == 0.0


def test_reset_zeros_fb_state():
  from openpilot.selfdrive.controls.lib.latcontrol_lmc2 import LatControlLMC2

  CP = SimpleNamespace(steerActuatorDelay=0.2, steerLimitTimer=1.0)
  lac = LatControlLMC2(CP, None, 0.01)
  lac.kappa_fb = 0.4
  lac.reset()
  assert lac.kappa_fb == 0.0
