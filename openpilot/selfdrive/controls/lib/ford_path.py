from dataclasses import dataclass
import math

import numpy as np

from openpilot.selfdrive.modeld.constants import ModelConstants

DBC_OFFSET = (-5.12, 5.11)
DBC_ANGLE = (-0.5, 0.5235)
DBC_CURVATURE = (-0.02, 0.02)
DBC_CURVATURE_RATE = (-0.001024, 0.001023)

T_WALK_MAX = 2.0
LOOKAHEAD_S = 7.0
# c2 is LPF'd in the PSCM and does not unwind with c0/c1. Only use it for
# shallow curves; intersection turns live on c0/c1.
_C2_OFFSET = 0.3
_C2_ANGLE = 0.05


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
  ds = np.hypot(np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0]))
  ds[0] = 0.0
  dist = np.cumsum(ds)

  t_lo = float(np.clip(t_prev, times[0], times[-1]))
  t_hi = min(T_WALK_MAX, float(times[-1]))
  t_star = float(np.interp(LOOKAHEAD_S, dist, times))
  t_star = float(np.clip(t_star, t_lo, t_hi))
  y_s = float(np.interp(t_star, times, y))
  psi_s = float(np.interp(t_star, times, heading))
  kappa = _finite(desired_curvature) if abs(y_s) < _C2_OFFSET and abs(psi_s) < _C2_ANGLE else 0.0

  return FordPath(
    valid=True,
    path_offset=_clip(y_s, DBC_OFFSET),
    path_angle=_clip(psi_s, DBC_ANGLE),
    curvature=_clip(kappa, DBC_CURVATURE),
    curvature_rate=_clip(0.0, DBC_CURVATURE_RATE),
  )
