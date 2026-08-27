from dataclasses import dataclass
import math

import numpy as np


DBC_OFFSET = (-5.12, 5.11)
DBC_ANGLE = (-0.5, 0.5235)
DBC_CURVATURE = (-0.02, 0.02)
DBC_CURVATURE_RATE = (-0.001024, 0.001023)

_PREVIEW_M = 7.0
_PATH_FIT_M = 20.0

# Hands-off channel identification from 863k archived 20 Hz samples. C2 is
# the only slow path field; use its time constant to estimate retained state.
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

  # Rebase all four Ford fields at one point on the model path.
  preview = float(np.clip(x[0] + _PREVIEW_M, x[0], x[-1]))
  offset = float(np.interp(preview, x, y))
  path_angle = float(np.interp(preview, x, heading))

  # Fit the derivative form of Ford's cubic over the local model path. This
  # rejects point noise while yielding C2/C3 at the same reference as C0/C1.
  distance = np.cumsum(np.hypot(np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0])))
  moving = np.concatenate(([True], np.diff(distance) > 1e-3))
  local = moving & (distance <= _PATH_FIT_M)
  if np.count_nonzero(local) < 3:
    return None
  coefficients = np.polynomial.polynomial.polyfit(distance[local], heading[local], 2)
  preview_distance = float(np.interp(preview, x, distance))
  curvature = float(coefficients[1] + 2.0 * coefficients[2] * preview_distance)
  curvature_rate = float(2.0 * coefficients[2])
  return FordPath(
    valid=True,
    path_offset=_clip(offset, DBC_OFFSET),
    path_angle=_clip(path_angle, DBC_ANGLE),
    curvature=_clip(curvature, DBC_CURVATURE),
    curvature_rate=_clip(curvature_rate, DBC_CURVATURE_RATE),
  )


class FordPathController:
  """Send one Ford path polynomial and actively drain retained C2."""

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
    tau = float(np.interp(speed, _C2_SPEEDS, _C2_TAUS))
    applied_c2 = path.curvature if applied_curvature is None else _finite(applied_curvature)
    alpha = 1.0 - math.exp(-self.dt / max(tau, self.dt))
    self.c2_response += alpha * (applied_c2 - self.c2_response)

    # C0/C1 remain the model polynomial and therefore never close another loop
    # around the PSCM. When the polynomial unwinds or reverses, use C2 itself to
    # cancel the estimated slow state instead of contaminating the fast fields.
    curvature = path.curvature
    c2_error = path.curvature - self.c2_response
    unwinding = path.curvature * self.c2_response <= 0.0 or abs(path.curvature) < abs(self.c2_response)
    if unwinding and abs(self.c2_response) > _C2_UNWIND_MIN and abs(c2_error) > _C2_UNWIND_DEADBAND:
      curvature += c2_error

    return FordPath(
      valid=True,
      path_offset=path.path_offset,
      path_angle=path.path_angle,
      curvature=_clip(curvature, DBC_CURVATURE),
      curvature_rate=path.curvature_rate,
    )


def encode_ford_path(model, t_prev: float, desired_curvature: float = 0.0, *, v_ego: float = 0.0) -> FordPath:
  """Stateless compatibility helper; live control uses FordPathController."""
  del t_prev
  return FordPathController().update(model, desired_curvature, v_ego=v_ego)
