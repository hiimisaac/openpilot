from dataclasses import dataclass
import math

import numpy as np


DBC_OFFSET = (-5.12, 5.11)
DBC_ANGLE = (-0.5, 0.5235)
DBC_CURVATURE = (-0.02, 0.02)
DBC_CURVATURE_RATE = (-0.001024, 0.001023)

_MIN_PREDICTION_S = 0.05
_MAX_PREDICTION_S = 1.0

# Hands-off channel identification from 863k archived 20 Hz samples. C2 is
# the only slow path field; its time constant provides continuous lead
# compensation for the geometric curvature derivative carried by C3.
_C2_SPEEDS = (1.5, 4.5, 8.0, 12.5, 18.5, 28.5)
_C2_TAUS = (0.750, 0.800, 0.791, 0.779, 0.598, 1.330)


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


def _predicted_pose(curvature: float, distance: float) -> tuple[float, float, float]:
  """Propagate the measured vehicle pose over the short actuator delay."""
  curvature = _finite(curvature)
  distance = max(_finite(distance), 0.0)
  heading = curvature * distance
  if abs(curvature) < 1e-9:
    return distance, 0.0, 0.0
  return math.sin(heading) / curvature, (1.0 - math.cos(heading)) / curvature, heading


def _model_arrays(model) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
  try:
    x = np.asarray(model.position.x, dtype=float)
    y = np.asarray(model.position.y, dtype=float)
    heading = np.unwrap(np.asarray(model.orientation.z, dtype=float))
    t = np.asarray(model.position.t, dtype=float)
  except (AttributeError, TypeError, ValueError):
    return None
  if len(x) < 3 or len(y) != len(x) or len(heading) != len(x) or len(t) != len(x):
    return None
  if not np.isfinite(np.concatenate((t, x, y, heading))).all():
    return None

  t, unique = np.unique(t, return_index=True)
  x = x[unique]
  y = y[unique]
  heading = heading[unique]
  if len(t) < 3 or t[-1] <= t[0]:
    return None
  return t, x, y, heading


def _local_geometry(t: np.ndarray, x: np.ndarray, y: np.ndarray, heading: np.ndarray,
                    target_time: float) -> tuple[float, float] | None:
  """Evaluate C2/C3 at the same predicted-time origin used by C0/C1."""
  path_distance = np.cumsum(np.hypot(np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0])))
  target_distance = float(np.interp(target_time, t, path_distance))
  distance, unique = np.unique(path_distance, return_index=True)
  local_heading = heading[unique]
  if len(distance) < 3 or distance[-1] - distance[0] < 1e-3:
    return None

  curvature = np.gradient(local_heading, distance, edge_order=2)
  curvature_rate = np.gradient(curvature, distance, edge_order=2)
  return float(np.interp(target_distance, distance, curvature)), float(np.interp(target_distance, distance, curvature_rate))


def _fit_path(model, speed: float, current_curvature: float, actuator_delay: float) -> FordPath | None:
  arrays = _model_arrays(model)
  if arrays is None:
    return None
  t, x, y, heading = arrays

  prediction_time = float(np.clip(_finite(actuator_delay), _MIN_PREDICTION_S, _MAX_PREDICTION_S))
  target_time = float(np.clip(t[0] + prediction_time, t[0], t[-1]))
  target_x = float(np.interp(target_time, t, x))
  target_y = float(np.interp(target_time, t, y))
  target_heading = float(np.interp(target_time, t, heading))

  predicted_x, predicted_y, predicted_heading = _predicted_pose(current_curvature, speed * prediction_time)
  dx = target_x - predicted_x
  dy = target_y - predicted_y
  path_offset = -math.sin(predicted_heading) * dx + math.cos(predicted_heading) * dy
  path_angle = target_heading - predicted_heading

  # Evaluate the local derivatives at the predicted-time origin. A forward
  # window fit can pull a future curve into the present and make the four fields
  # disagree; point-local derivatives keep one coherent short polynomial.
  geometry = _local_geometry(t, x, y, heading, target_time)
  if geometry is None:
    return None
  geometric_curvature, geometric_curvature_rate = geometry
  curvature_rate = _clip(geometric_curvature_rate, DBC_CURVATURE_RATE)

  # Invert the identified first-order C2 lag continuously. For a spatial path,
  # d(curvature)/dt = speed * C3. This supplies lead on entry and an equal-sign
  # drain on exit without entry/unwind states or thresholds.
  tau = float(np.interp(speed, _C2_SPEEDS, _C2_TAUS))
  curvature_command = geometric_curvature + tau * speed * curvature_rate

  return FordPath(
    valid=True,
    path_offset=_clip(path_offset, DBC_OFFSET),
    path_angle=_clip(path_angle, DBC_ANGLE),
    curvature=_clip(curvature_command, DBC_CURVATURE),
    curvature_rate=curvature_rate,
  )


class FordPathController:
  """Send continuously refreshed, delay-aligned Ford path polynomials."""

  def __init__(self, dt: float = 0.01):
    del dt

  def reset(self) -> None:
    pass

  def update(self, model, desired_curvature: float = 0.0, *, v_ego: float = 0.0, active: bool = True,
             current_curvature: float = 0.0, actuator_delay: float = 0.0) -> FordPath:
    del desired_curvature
    path = _fit_path(model, max(_finite(v_ego), 0.0), current_curvature, actuator_delay) if model is not None else None
    if not active or path is None:
      return FordPath()
    return path


def encode_ford_path(model, t_prev: float, desired_curvature: float = 0.0, *, v_ego: float = 0.0,
                     current_curvature: float = 0.0, actuator_delay: float = 0.0) -> FordPath:
  """Stateless compatibility helper; live control uses FordPathController."""
  del t_prev
  return FordPathController().update(model, desired_curvature, v_ego=v_ego, current_curvature=current_curvature,
                                     actuator_delay=actuator_delay)
