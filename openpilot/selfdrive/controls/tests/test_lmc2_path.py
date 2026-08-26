import math

import numpy as np

from openpilot.selfdrive.controls.lib.lmc2_path import X_FIT_M, path_to_lmc_states


def cubic_y(x, e_y, e_psi, kappa, kappa_dot):
  x = np.asarray(x, dtype=np.float64)
  return e_y + e_psi * x + 0.5 * kappa * x ** 2 + kappa_dot * x ** 3 / 6.0


def test_identities_on_patent_cubic():
  e_y, e_psi, kappa, kappa_dot = 0.3, 0.05, 0.01, 0.0004
  x = np.linspace(0.0, 18.0, 20)
  meas = path_to_lmc_states(x, cubic_y(x, e_y, e_psi, kappa, kappa_dot))
  assert meas.valid
  assert math.isclose(meas.e_y, e_y, abs_tol=1e-6)
  assert math.isclose(meas.e_psi, e_psi, abs_tol=1e-6)
  assert math.isclose(meas.kappa_road, kappa, rel_tol=1e-4, abs_tol=1e-6)
  assert math.isclose(meas.kappa_road_dot, kappa_dot, rel_tol=1e-4, abs_tol=1e-6)


def test_kappa_is_two_a2():
  kappa, kappa_dot = 0.008, -0.0002
  x = np.linspace(0.0, 16.0, 17)
  y = cubic_y(x, 0.0, 0.0, kappa, kappa_dot)
  meas = path_to_lmc_states(x, y)
  assert meas.valid
  assert math.isclose(meas.kappa_road, kappa, rel_tol=1e-4, abs_tol=1e-6)
  assert math.isclose(meas.kappa_road_dot, kappa_dot, rel_tol=1e-4, abs_tol=1e-6)


def test_leading_negative_x_is_skipped():
  e_y, e_psi, kappa, kappa_dot = 0.3, 0.05, 0.01, 0.0004
  x = np.concatenate(([-0.05], np.linspace(0.0, 18.0, 20)))
  y = np.concatenate(([99.0], cubic_y(x[1:], e_y, e_psi, kappa, kappa_dot)))
  meas = path_to_lmc_states(x, y)
  assert meas.valid
  assert math.isclose(meas.e_y, e_y, abs_tol=1e-5)
  assert math.isclose(meas.e_psi, e_psi, abs_tol=1e-5)
  assert math.isclose(meas.kappa_road, kappa, rel_tol=1e-3, abs_tol=1e-6)


def test_invalid_short_and_nonfinite():
  assert not path_to_lmc_states([], []).valid
  assert not path_to_lmc_states([0.0, 1.0, 2.0], [0.0, 0.1, 0.2]).valid
  x = np.linspace(0.0, 18.0, 20)
  y = cubic_y(x, 0.1, 0.0, 0.002, 0.0)
  y[4] = np.nan
  # NaN mid-path ends the prefix at the first bad sample; if that prefix is
  # still long enough the fit stays valid. A NaN at index 2 is short.
  y_short = y.copy()
  y_short[2] = np.nan
  assert not path_to_lmc_states(x, y_short).valid


def test_time_ordered_fold_uses_rising_prefix_only():
  e_y, e_psi, kappa, kappa_dot = 0.2, 0.02, 0.006, 0.0001
  x_up = np.linspace(0.0, 18.0, 19)
  y_up = cubic_y(x_up, e_y, e_psi, kappa, kappa_dot)
  x_down = np.linspace(17.0, 1.0, 10)
  y_down = np.full(x_down.shape, 50.0)
  meas = path_to_lmc_states(np.concatenate((x_up, x_down)), np.concatenate((y_up, y_down)))
  assert meas.valid
  assert math.isclose(meas.e_y, e_y, abs_tol=1e-4)
  assert abs(meas.e_y - 50.0) > 10.0
  assert math.isclose(meas.kappa_road, kappa, rel_tol=1e-3, abs_tol=1e-6)


def test_s_curve_beyond_window_does_not_own_a3():
  kappa, kappa_dot = 0.004, 0.00015
  x_near = np.linspace(0.0, 18.0, 19)
  y_near = cubic_y(x_near, 0.0, 0.0, kappa, kappa_dot)
  x_far = np.linspace(22.0, 80.0, 20)
  y_far = 4.0 * np.sin((x_far - 22.0) / 3.0)
  meas = path_to_lmc_states(np.concatenate((x_near, x_far)), np.concatenate((y_near, y_far)))
  assert meas.valid
  assert math.isclose(meas.kappa_road, kappa, rel_tol=0.05, abs_tol=1e-5)
  assert math.isclose(meas.kappa_road_dot, kappa_dot, rel_tol=0.05, abs_tol=1e-5)


def test_sparse_near_field_is_invalid():
  x = np.array([0.0, 8.0, 16.0, 25.0, 30.0, 40.0])
  y = cubic_y(x, 0.0, 0.0, 0.01, 0.0)
  meas = path_to_lmc_states(x, y)
  in_win = x[x <= X_FIT_M]
  assert in_win.size < 4
  assert not meas.valid
