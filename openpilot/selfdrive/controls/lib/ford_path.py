from dataclasses import dataclass
import math

import numpy as np


DBC_OFFSET = (-5.12, 5.11)
DBC_ANGLE = (-0.5, 0.5235)
DBC_CURVATURE = (-0.02, 0.02)
DBC_CURVATURE_RATE = (-0.001024, 0.001023)

# Hands-off channel identification from 863k archived 20 Hz samples. C1 is
# the fast path-error channel; C2 is the slower road-curvature channel.
_C1_SPEEDS = (4.5, 8.0, 12.5, 18.5, 28.5)
_C1_GAINS = (0.1273, 0.0876, 0.0724, 0.0504, 0.0380)
_C2_SPEEDS = (1.5, 4.5, 8.0, 12.5, 18.5, 28.5)
_C2_GAINS = (0.8800, 0.8312, 0.9109, 1.0235, 1.0616, 1.0476)
_C2_TAUS = (0.750, 0.800, 0.791, 0.779, 0.598, 1.330)
_C0_SPEEDS = (3.0, 6.0, 10.0)
_C0_GAINS = (0.00903, 0.00914, 0.00134)
_C3_PREVIEW_M = 1.25
_PATH_FIT_M = 20.0


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


def _curvature_rate(model) -> float | None:
  try:
    x = np.asarray(model.position.x, dtype=float)
    y = np.asarray(model.position.y, dtype=float)
    heading = np.unwrap(np.asarray(model.orientation.z, dtype=float))
  except (AttributeError, TypeError, ValueError):
    return None
  if len(x) < 3 or len(y) != len(x) or len(heading) != len(x) or not np.isfinite(np.concatenate((x, y, heading))).all():
    return None

  distance = np.cumsum(np.hypot(np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0])))
  moving = np.concatenate(([True], np.diff(distance) > 1e-3))
  local = moving & (distance <= _PATH_FIT_M)
  if np.count_nonzero(local) < 3:
    return 0.0
  coefficients = np.polynomial.polynomial.polyfit(distance[local], heading[local], 2)
  return _finite(2.0 * coefficients[2])


class FordPathController:
  """Allocate one curvature request between Ford's slow C2 and fast C0/C1 channels."""

  def __init__(self, dt: float = 0.01):
    self.dt = dt
    self.c2_response = 0.0

  def reset(self) -> None:
    self.c2_response = 0.0

  def update(self, model, desired_curvature: float = 0.0, *, v_ego: float = 0.0, active: bool = True,
             applied_curvature: float | None = None) -> FordPath:
    curvature_rate = _curvature_rate(model) if model is not None else None
    if not active or curvature_rate is None:
      self.reset()
      return FordPath()

    speed = max(_finite(v_ego), 0.0)
    desired = _finite(desired_curvature)
    c1_gain = float(np.interp(speed, _C1_SPEEDS, _C1_GAINS))
    c2_gain = float(np.interp(speed, _C2_SPEEDS, _C2_GAINS))
    c2_tau = float(np.interp(speed, _C2_SPEEDS, _C2_TAUS))

    # C2 carries the complete steady road curvature. Track the PSCM's measured
    # first-order response so C1 supplies only what C2 has not delivered yet.
    c2 = _clip(desired / c2_gain, DBC_CURVATURE)
    applied_c2 = c2 if applied_curvature is None else _finite(applied_curvature)
    c2_target = c2_gain * (applied_c2 + _C3_PREVIEW_M * curvature_rate)
    alpha = 1.0 - math.exp(-self.dt / max(c2_tau, self.dt))
    self.c2_response += alpha * (c2_target - self.c2_response)
    residual = desired - self.c2_response

    c1_unclipped = residual / c1_gain
    c1 = _clip(c1_unclipped, DBC_ANGLE)
    residual -= c1 * c1_gain

    # C0 is the remaining low-speed overflow only. Its identified gain becomes
    # too small and uncertain above 10 m/s; C1 has ample authority there.
    c0 = 0.0
    if abs(residual) > 1e-6 and speed < _C0_SPEEDS[-1]:
      c0_gain = float(np.interp(speed, _C0_SPEEDS, _C0_GAINS))
      c0 = _clip(residual / c0_gain, DBC_OFFSET)

    return FordPath(
      valid=True,
      path_offset=c0,
      path_angle=c1,
      curvature=c2,
      curvature_rate=_clip(curvature_rate, DBC_CURVATURE_RATE),
    )


def encode_ford_path(model, t_prev: float, desired_curvature: float = 0.0, *, v_ego: float = 0.0) -> FordPath:
  """Stateless compatibility helper; live control uses FordPathController."""
  del t_prev
  return FordPathController().update(model, desired_curvature, v_ego=v_ego)
