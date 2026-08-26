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
# C2 is LPF'd in the PSCM, so keep it small enough to remain a centering trim.
# C0/C1 always describe the path from the same sample and handle fast motion.
_C2_OFFSET = 0.3
_C2_ANGLE = 0.05
_C2_MAX = 0.001


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


def encode_ford_path(model, t_prev: float) -> FordPath:
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
  s_star = float(np.interp(t_star, times, dist))
  y_s = float(np.interp(t_star, times, y))
  psi_s = float(np.interp(t_star, times, heading))
  moving = np.concatenate(([True], np.diff(dist) > 1e-3))
  path_dist = dist[moving]
  path_heading = heading[moving]
  kappa = 0.0
  kappa_rate = 0.0
  if path_dist.size >= 3:
    local = np.abs(path_dist - s_star) <= LOOKAHEAD_S
    if np.count_nonzero(local) >= 3:
      # A quadratic heading fit gives curvature and curvature rate at the same
      # reference point while rejecting point-to-point model noise.
      heading_poly = np.polynomial.polynomial.polyfit(path_dist[local] - s_star, path_heading[local], 2)
      kappa = float(heading_poly[1])
      kappa_rate = float(2.0 * heading_poly[2])
  turn = abs(y_s) >= _C2_OFFSET or abs(psi_s) >= _C2_ANGLE
  kappa = 0.0 if turn else _finite(kappa)

  return FordPath(
    valid=True,
    path_offset=_clip(y_s, DBC_OFFSET),
    path_angle=_clip(psi_s, DBC_ANGLE),
    curvature=_clip(kappa, (-_C2_MAX, _C2_MAX)),
    curvature_rate=_clip(_finite(kappa_rate), DBC_CURVATURE_RATE),
  )
