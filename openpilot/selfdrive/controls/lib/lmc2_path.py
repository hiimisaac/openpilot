from dataclasses import dataclass

import numpy as np
from numpy.polynomial.polynomial import polyfit

X_FIT_M = 20.0  # local spatial window for 059 eqs. (7)/(10)/(11)
MIN_PTS = 4


@dataclass(frozen=True)
class LMCStates:
  valid: bool
  e_y: float            # m, model y-left +
  e_psi: float          # rad
  kappa_road: float     # 1/m
  kappa_road_dot: float # 1/m^2, dκ/dx


def path_to_lmc_states(position_x, position_y) -> LMCStates:
  """Walk modelV2.position in model order. No argsort."""
  invalid = LMCStates(False, 0.0, 0.0, 0.0, 0.0)
  x = np.asarray(position_x, dtype=np.float64)
  y = np.asarray(position_y, dtype=np.float64)
  if x.shape != y.shape or x.size == 0:
    return invalid
  # Skip leading nonfinite / x<0 (behind bumper). Then longest strictly
  # increasing finite x>=0 prefix. Mid-path x decrease is a fold → break.
  px, py = [], []
  x_prev = None
  for i in range(x.size):
    xi, yi = float(x[i]), float(y[i])
    if x_prev is None:
      if not (np.isfinite(xi) and np.isfinite(yi) and xi >= 0.0):
        continue
      px.append(xi)
      py.append(yi)
      x_prev = xi
      continue
    if not (np.isfinite(xi) and np.isfinite(yi) and xi >= 0.0 and xi > x_prev):
      break
    px.append(xi)
    py.append(yi)
    x_prev = xi
  if len(px) < MIN_PTS:
    return invalid
  xw, yw = np.asarray(px), np.asarray(py)
  in_win = xw <= X_FIT_M
  xw, yw = xw[in_win], yw[in_win]
  if xw.size < MIN_PTS:
    return invalid  # do not take points beyond 20 m
  coef, (_resid, _rank, sv, _rcond) = polyfit(xw, yw, 3, full=True)
  if coef.size < 4 or not np.all(np.isfinite(coef)):
    return invalid
  sv = np.asarray(sv)
  if sv.size >= 2 and sv[-1] <= 1e-12 * sv[0]:
    return invalid
  a0, a1, a2, a3 = (float(coef[i]) for i in range(4))
  return LMCStates(True, a0, a1, 2.0 * a2, 6.0 * a3)
