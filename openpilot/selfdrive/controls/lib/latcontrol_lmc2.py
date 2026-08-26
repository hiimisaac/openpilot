from dataclasses import dataclass

import numpy as np

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.selfdrive.controls.lib.lmc2_path import LMCStates

# PSCM DBC ranges (hardware)
DBC_OFFSET = (5.12, 5.11)          # m, -/+
DBC_ANGLE = (0.5, 0.5235)          # rad, -/+
DBC_CURVATURE = 0.02               # 1/m
DBC_CURVATURE_RATE = 0.001024      # 1/m^2


@dataclass
class LMC2Params:
  zeta: float = 0.707
  t_r: float = 5.0
  k_ff: float = 1.0
  t_prev: float = 0.2
  a_y_fb_max: float = 3.0
  j_fb_max: float = 5.0
  v_min: float = 1.0


@dataclass(frozen=True)
class PathCommand:
  valid: bool
  pathOffset: float
  pathAngle: float
  curvature: float
  curvatureRate: float


def lmc2_gains(v_x: float, p: LMC2Params) -> tuple[float, float]:
  v = max(v_x, p.v_min)
  k_y = 18.5 * p.zeta ** 2 / (p.t_r ** 2 * v ** 2)
  k_psi = 8.6 * p.zeta ** 2 / (p.t_r * v)
  return k_y, k_psi


def lmc2_step(meas: LMCStates, v_x: float, p: LMC2Params, kappa_fb_prev: float, dt: float, active: bool) -> tuple[float, float, float, float]:
  """Returns kappa_cmd, kappa_fb, kappa_ff, (k_y unused by caller via gains)."""
  if (not active) or (not meas.valid):
    return 0.0, 0.0, 0.0, 0.0
  v = max(v_x, p.v_min)
  k_y, k_psi = lmc2_gains(v_x, p)
  kappa_ff = p.k_ff * (meas.kappa_road + v * p.t_prev * meas.kappa_road_dot)
  kappa_fb = k_y * meas.e_y + k_psi * meas.e_psi
  ay = float(np.clip(kappa_fb * v ** 2, -p.a_y_fb_max, p.a_y_fb_max))
  kappa_fb = ay / v ** 2
  max_d = (p.j_fb_max / v ** 2) * dt
  kappa_fb = float(np.clip(kappa_fb, kappa_fb_prev - max_d, kappa_fb_prev + max_d))
  return kappa_ff + kappa_fb, kappa_fb, kappa_ff, k_psi


def path_command(meas: LMCStates | None, kappa_cmd: float, active: bool) -> PathCommand:
  """Pure function of this-frame meas, this-frame κ_cmd, and active. No stored meas."""
  if (not active) or meas is None or (not meas.valid):
    return PathCommand(valid=False, pathOffset=0.0, pathAngle=0.0,
                       curvature=0.0, curvatureRate=0.0)
  return PathCommand(valid=True, pathOffset=meas.e_y, pathAngle=meas.e_psi,
                     curvature=kappa_cmd, curvatureRate=meas.kappa_road_dot)


def path_at_dbc_limit(pc: PathCommand) -> bool:
  if not pc.valid:
    return False
  return (
    pc.pathOffset <= -DBC_OFFSET[0] or pc.pathOffset >= DBC_OFFSET[1] or
    pc.pathAngle <= -DBC_ANGLE[0] or pc.pathAngle >= DBC_ANGLE[1] or
    abs(pc.curvature) >= DBC_CURVATURE or
    abs(pc.curvatureRate) >= DBC_CURVATURE_RATE
  )


class LatControlLMC2(LatControl):
  def __init__(self, CP, CI, dt):
    super().__init__(CP, CI, dt)
    self.p = LMC2Params(t_prev=float(CP.steerActuatorDelay))
    self.kappa_fb = 0.0

  def reset(self) -> None:
    super().reset()
    self.kappa_fb = 0.0

  def update(self, active, CS, VM, params, steer_limited_by_safety,
             desired_curvature, curvature_limited, lat_delay,
             meas: LMCStates | None = None):
    lmc2_log = log.ControlsState.LateralLMC2State.new_message()
    if meas is None or (not active) or (not meas.valid):
      self.kappa_fb = 0.0
      kappa_cmd = 0.0
      kappa_ff = 0.0
      k_y = k_psi = 0.0
      valid = False
    else:
      kappa_cmd, self.kappa_fb, kappa_ff, k_psi = lmc2_step(
        meas, float(CS.vEgo), self.p, self.kappa_fb, self.dt, True)
      k_y, k_psi = lmc2_gains(float(CS.vEgo), self.p)
      valid = True

    pc = path_command(meas, kappa_cmd, bool(active))
    lmc2_log.active = bool(active)
    lmc2_log.validMeas = bool(valid)
    if meas is not None:
      lmc2_log.eY = float(meas.e_y)
      lmc2_log.ePsi = float(meas.e_psi)
      lmc2_log.kappaRoad = float(meas.kappa_road)
      lmc2_log.kappaRoadDot = float(meas.kappa_road_dot)
    lmc2_log.kappaFf = float(kappa_ff)
    lmc2_log.kappaFb = float(self.kappa_fb)
    lmc2_log.kappaCmd = float(kappa_cmd)
    lmc2_log.kY = float(k_y)
    lmc2_log.kPsi = float(k_psi)
    dbc_sat = path_at_dbc_limit(pc)
    lmc2_log.saturated = bool(self._check_saturation(dbc_sat, CS, steer_limited_by_safety, False))
    return 0.0, 0.0, lmc2_log
