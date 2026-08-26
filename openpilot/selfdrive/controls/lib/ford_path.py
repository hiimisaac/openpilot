from dataclasses import dataclass
import math

import numpy as np


DBC_OFFSET = (-5.12, 5.11)
DBC_ANGLE = (-0.5, 0.5235)
DBC_CURVATURE = (-0.02, 0.02)
DBC_CURVATURE_RATE = (-0.001024, 0.001023)

_PREVIEW_M = 7.0

# Hands-off channel identification from 863k archived 20 Hz samples. This is
# used only to cancel curvature still retained by the PSCM's C2 low-pass filter.
_C1_SPEEDS = (4.5, 8.0, 12.5, 18.5, 28.5)
_C1_GAINS = (0.1273, 0.0876, 0.0724, 0.0504, 0.0380)
_C2_SPEEDS = (1.5, 4.5, 8.0, 12.5, 18.5, 28.5)
_C2_TAUS = (0.750, 0.800, 0.791, 0.779, 0.598, 1.330)
_C2_UNWIND_DEADBAND = 0.001
_C2_UNWIND_MIN = 0.002


@dataclass(frozen=True)
class FordPath:
  valid: bool = False
  path_offset: float = 0.0
  path_angle: float = 0.0
  curvature: float = 0.0
  curvature_rate: float = 0.0


def _clip(value: float, limits: tuple[float, float]) -> float:
  return float(np.clip(value, limits[0], limits[1]))


def _finite(value: float) -> float:
  return float(value) if math.isfinite(value) else 0.0


def _fit_path(model) -> FordPath | None:
  try:
    x = np.asarray(model.position.x, dtype=float)
    y = np.asarray(model.position.y, dtype=float)
    heading = np.unwrap(np.asarray(model.orientation.z, dtype=float))
  except (AttributeError, TypeError, ValueError):
    return None
  if len(x) < 2 or len(y) != len(x) or len(heading) != len(x) or not np.isfinite(np.concatenate((x, y, heading))).all():
    return None

  x, unique = np.unique(x, return_index=True)
  y = y[unique]
  heading = heading[unique]
  if len(x) < 2:
    return None

  # Build the one cubic y(x) fixed by the model path's offset and heading at
  # the vehicle and at one preview point. Rebase every Ford field at that same
  # preview point; unlike the old dual-lookahead encoder, no term comes from a
  # different path sample or from desiredCurvature.
  preview = float(np.clip(x[0] + _PREVIEW_M, x[0], x[-1]))
  length = preview - x[0]
  if length < 1e-3:
    return None
  y0 = float(y[0])
  offset = float(np.interp(preview, x, y))
  slope0 = math.tan(float(heading[0]))
  slope = math.tan(float(np.interp(preview, x, heading)))
  a2 = 3.0 * (offset - y0) / length ** 2 - (2.0 * slope0 + slope) / length
  a3 = -2.0 * (offset - y0) / length ** 3 + (slope0 + slope) / length ** 2
  second = 2.0 * a2 + 6.0 * a3 * length
  third = 6.0 * a3

  slope_scale = 1.0 + slope ** 2
  curvature = second / slope_scale ** 1.5
  curvature_rate = third / slope_scale ** 2 - 3.0 * slope * second ** 2 / slope_scale ** 3
  return FordPath(
    valid=True,
    path_offset=_clip(offset, DBC_OFFSET),
    path_angle=_clip(math.atan(slope), DBC_ANGLE),
    curvature=_clip(curvature, DBC_CURVATURE),
    curvature_rate=_clip(curvature_rate, DBC_CURVATURE_RATE),
  )


class FordPathController:
  """Send one Ford path polynomial, with fast compensation for retained C2."""

  def __init__(self, dt: float = 0.01):
    self.dt = dt
    self.c2_response = 0.0

  def reset(self) -> None:
    self.c2_response = 0.0

  def update(self, model, desired_curvature: float = 0.0, *, v_ego: float = 0.0, active: bool = True,
             applied_curvature: float | None = None) -> FordPath:
    del desired_curvature
    path = _fit_path(model) if model is not None else None
    if not active or path is None:
      self.reset()
      return FordPath()

    speed = max(_finite(v_ego), 0.0)
    c2_tau = float(np.interp(speed, _C2_SPEEDS, _C2_TAUS))
    applied_c2 = path.curvature if applied_curvature is None else _finite(applied_curvature)
    alpha = 1.0 - math.exp(-self.dt / max(c2_tau, self.dt))
    self.c2_response += alpha * (applied_c2 - self.c2_response)

    # C0-C3 above remain the direct polynomial. Only override C1 while C2 is
    # unwinding or reversing, when the PSCM's retained curvature would otherwise
    # keep steering through a turn that the requested polynomial has exited.
    retained_c2 = self.c2_response
    same_direction = path.curvature * retained_c2 > 0.0
    c2_error = path.curvature - retained_c2
    c2_overhang = abs(retained_c2) > _C2_UNWIND_MIN and abs(c2_error) > _C2_UNWIND_DEADBAND and (
      (same_direction and abs(retained_c2) > abs(path.curvature)) or (retained_c2 != 0.0 and not same_direction)
    )
    path_angle = path.path_angle
    if c2_overhang:
      c1_gain = float(np.interp(speed, _C1_SPEEDS, _C1_GAINS))
      path_angle += c2_error / c1_gain

    return FordPath(
      valid=True,
      path_offset=path.path_offset,
      path_angle=_clip(path_angle, DBC_ANGLE),
      curvature=path.curvature,
      curvature_rate=path.curvature_rate,
    )


def encode_ford_path(model, t_prev: float, desired_curvature: float = 0.0, *, v_ego: float = 0.0) -> FordPath:
  """Stateless compatibility helper; live control uses FordPathController."""
  del t_prev
  return FordPathController().update(model, desired_curvature, v_ego=v_ego)
