from dataclasses import dataclass
import math

import numpy as np


DBC_OFFSET = (-5.12, 5.11)
DBC_ANGLE = (-0.5, 0.5235)
DBC_CURVATURE = (-0.02, 0.02)
DBC_CURVATURE_RATE = (-0.001024, 0.001023)

_PREVIEW_M = 7.0
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


def _predicted_pose(curvature: float, distance: float) -> tuple[float, float]:
  """Constant-curvature vehicle pose at a longitudinal preview distance."""
  curvature = _finite(curvature)
  if abs(curvature) < 1e-9:
    return 0.0, 0.0
  kx = float(np.clip(curvature * distance, -0.99, 0.99))
  heading = math.asin(kx)
  offset = (1.0 - math.sqrt(1.0 - kx ** 2)) / curvature
  return offset, heading


def _fit_path(model, current_curvature: float) -> FordPath | None:
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

  preview = float(np.clip(x[0] + _PREVIEW_M, x[0], x[-1]))
  length = preview - x[0]
  if length < 1e-3:
    return None

  # Ford wants path offset and heading error, not the absolute pose of a point
  # ahead. Express the model target in the vehicle pose predicted from measured
  # curvature at the same preview distance. An aligned curve therefore has zero
  # C0/C1; entry, unwind, and reversal naturally produce the correct error sign.
  target_offset = float(np.interp(preview, x, y))
  target_heading = float(np.interp(preview, x, heading))
  predicted_offset, predicted_heading = _predicted_pose(current_curvature, length)
  path_offset = (target_offset - predicted_offset) * math.cos(predicted_heading)
  path_angle = target_heading - predicted_heading

  # A quadratic heading fit is the derivative form of Ford's cubic path. It
  # rejects point-to-point model noise while providing curvature and spatial
  # curvature rate at the same reference used for C0/C1.
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
    path_offset=_clip(path_offset, DBC_OFFSET),
    path_angle=_clip(path_angle, DBC_ANGLE),
    curvature=_clip(curvature, DBC_CURVATURE),
    curvature_rate=_clip(curvature_rate, DBC_CURVATURE_RATE),
  )


class FordPathController:
  """Encode Ford path geometry relative to the predicted vehicle pose."""

  def __init__(self, dt: float = 0.01):
    del dt

  def reset(self) -> None:
    pass

  def update(self, model, desired_curvature: float = 0.0, *, v_ego: float = 0.0, active: bool = True,
             current_curvature: float = 0.0) -> FordPath:
    del desired_curvature, v_ego
    path = _fit_path(model, current_curvature) if model is not None else None
    if not active or path is None:
      return FordPath()
    return path


def encode_ford_path(model, t_prev: float, desired_curvature: float = 0.0, *, v_ego: float = 0.0,
                     current_curvature: float = 0.0) -> FordPath:
  """Stateless compatibility helper; live control uses FordPathController."""
  del t_prev
  return FordPathController().update(model, desired_curvature, v_ego=v_ego, current_curvature=current_curvature)
