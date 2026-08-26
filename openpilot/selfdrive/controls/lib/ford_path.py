from dataclasses import dataclass
import math

import numpy as np

from openpilot.selfdrive.modeld.constants import ModelConstants

DBC_OFFSET = (-5.12, 5.11)
DBC_ANGLE = (-0.5, 0.5235)
DBC_CURVATURE = (-0.02, 0.02)
DBC_CURVATURE_RATE = (-0.001024, 0.001023)

T_WALK_MAX = 2.0


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


def _score(y: float, psi: float) -> float:
  return abs(y) / DBC_OFFSET[1] + abs(psi) / DBC_ANGLE[1]


def _times(model, n: int) -> np.ndarray | None:
  t = np.asarray(getattr(getattr(model, "position", None), "t", []), dtype=float)
  if t.size == n and np.all(np.isfinite(t)):
    return t
  t = np.asarray(ModelConstants.T_IDXS[:n], dtype=float)
  return t if t.size == n else None


def encode_ford_path(model, t_prev: float, desired_curvature: float = 0.0) -> FordPath:
  inactive = FordPath()
  if model is None:
    return inactive

  try:
    x = np.asarray(model.position.x, dtype=float)
    y = np.asarray(model.position.y, dtype=float)
    psi = np.asarray(model.orientation.z, dtype=float)
  except (AttributeError, TypeError, ValueError):
    return inactive
  n = len(x)
  if n < 2 or len(y) != n or len(psi) != n or not np.isfinite(np.concatenate((x, y, psi))).all():
    return inactive

  times = _times(model, n)
  if times is None:
    return inactive

  heading = np.unwrap(psi)
  t_lo = float(np.clip(t_prev, times[0], times[-1]))
  t_hi = min(T_WALK_MAX, float(times[-1]))
  y_s = float(np.interp(t_lo, times, y))
  psi_s = float(np.interp(t_lo, times, heading))
  best = _score(y_s, psi_s)
  for t_i, y_i, psi_i in zip(times, y, heading, strict=True):
    if t_i < t_lo or t_i > t_hi:
      continue
    sc = _score(float(y_i), float(psi_i))
    if sc > best:
      best = sc
      y_s = float(y_i)
      psi_s = float(psi_i)
  y_h = float(np.interp(t_hi, times, y))
  psi_h = float(np.interp(t_hi, times, heading))
  if _score(y_h, psi_h) > best:
    y_s, psi_s = y_h, psi_h

  return FordPath(
    valid=True,
    path_offset=_clip(y_s, DBC_OFFSET),
    path_angle=_clip(psi_s, DBC_ANGLE),
    curvature=_clip(_finite(desired_curvature), DBC_CURVATURE),
    curvature_rate=_clip(0.0, DBC_CURVATURE_RATE),
  )
