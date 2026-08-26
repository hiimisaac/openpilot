from types import SimpleNamespace

import numpy as np

from openpilot.selfdrive.controls.lib.ford_path import encode_ford_path


T_PREV = 0.2


def _model(t, x, y, psi, yaw_rate=None, desired_curvature=0.0):
  t = np.asarray(t, dtype=float)
  x = np.asarray(x, dtype=float)
  y = np.asarray(y, dtype=float)
  psi = np.asarray(psi, dtype=float)
  n = len(t)
  if yaw_rate is None:
    yaw_rate = np.gradient(psi, t) if n > 1 else np.zeros(n)
  return SimpleNamespace(
    position=SimpleNamespace(x=x.tolist(), y=y.tolist(), t=t.tolist()),
    orientation=SimpleNamespace(z=psi.tolist()),
    orientationRate=SimpleNamespace(z=np.asarray(yaw_rate, dtype=float).tolist()),
    action=SimpleNamespace(desiredCurvature=desired_curvature),
  )


def test_describes_intersection_turn_from_stop():
  # First in line: launch into a 15 m radius left. The 90 lives at 2–5 s, not 0.2 s.
  R, a, v_max = 15.0, 1.5, 6.0
  t = np.linspace(0.0, 10.0, 101)
  v = np.clip(a * t, 0.0, v_max)
  s = np.cumsum(np.pad(np.diff(t) * 0.5 * (v[1:] + v[:-1]), (1, 0)))
  th = np.minimum(s / R, 0.5 * np.pi)
  path = encode_ford_path(_model(t, R * np.sin(th), R * (1.0 - np.cos(th)), th), T_PREV)

  bumper_psi = np.interp(T_PREV, t, th)
  assert path.valid
  assert abs(path.path_angle) > 5.0 * abs(bumper_psi)
  assert abs(path.path_offset) > 0.1
  assert path.path_angle > 0.0


def test_moving_samples_along_track_not_two_seconds():
  t = np.linspace(0.0, 3.0, 31)
  x = 10.0 * t
  y = np.interp(t, [0.0, 0.2, 2.0], [0.0, 0.04, 3.0])
  psi = np.interp(t, [0.0, 0.2, 2.0], [0.0, 0.01, 0.4])
  path = encode_ford_path(_model(t, x, y, psi), T_PREV)

  y_7m = float(np.interp(0.7, t, y))
  assert path.valid
  assert abs(path.path_offset - y_7m) < 0.15
  assert abs(path.path_offset) < abs(np.interp(2.0, t, y)) - 0.5


def test_keeps_describing_a_turn_after_leaving_the_stop():
  R, v = 15.0, 4.0
  t = np.linspace(0.0, 5.0, 51)
  s = v * t
  th = np.minimum(s / R, 0.5 * np.pi)
  path = encode_ford_path(_model(t, R * np.sin(th), R * (1.0 - np.cos(th)), th), T_PREV)

  bumper = float(np.interp(T_PREV, t, R * (1.0 - np.cos(th))))
  assert path.path_offset > 1.0
  assert path.path_offset > 5.0 * bumper


def test_left_path_is_positive():
  t = np.array([0.0, 0.2, 1.0])
  path = encode_ford_path(_model(t, [0.0, 1.0, 5.0], [0.0, 0.4, 2.0], [0.0, 0.1, 0.4]), T_PREV)

  assert path.path_offset > 0
  assert path.path_angle > 0


def test_standstill_walks_time_until_launch_turn():
  t = np.linspace(0.0, 3.0, 31)
  x = np.zeros_like(t)
  y = np.zeros_like(t)
  psi = np.zeros_like(t)
  later = t >= 1.0
  y[later] = np.interp(t[later], [1.0, 2.0], [0.0, 2.0])
  psi[later] = np.interp(t[later], [1.0, 2.0], [0.0, 0.3])

  path = encode_ford_path(_model(t, x, y, psi), T_PREV)

  assert path.valid
  assert path.path_offset > 0.0
  assert path.path_angle > 0.0
  assert path.path_offset <= 2.0 + 1e-9


def test_does_not_sample_the_exit_of_a_turn():
  t = np.linspace(0.0, 3.0, 31)
  y = np.where(t <= 0.5, 0.4 + 0.2 * t, np.interp(t, [0.5, 2.0], [0.5, 0.0]))
  psi = np.where(t <= 0.5, 0.12, np.interp(t, [0.5, 2.0], [0.12, 0.0]))
  path = encode_ford_path(_model(t, 10.0 * t, y, psi), T_PREV)

  assert path.path_offset > 0.3
  assert path.path_angle > 0.05
  assert path.path_offset > np.interp(2.0, t, y) + 0.2


def test_does_not_invent_path_when_plan_stays_at_origin():
  t = np.linspace(0.0, 4.0, 21)
  path = encode_ford_path(_model(t, np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)), T_PREV)

  assert path.valid
  assert path.path_offset == 0.0
  assert path.path_angle == 0.0


def test_does_not_charge_c2_during_a_path_turn():
  R, v = 15.0, 4.0
  t = np.linspace(0.0, 5.0, 51)
  s = v * t
  th = np.minimum(s / R, 0.5 * np.pi)
  path = encode_ford_path(_model(t, R * np.sin(th), R * (1.0 - np.cos(th)), th), T_PREV)

  assert abs(path.path_offset) > 1.0
  assert path.curvature == 0.0


def test_keeps_c2_centering_with_continuous_shallow_path():
  t = np.linspace(0.0, 2.0, 21)
  x = 20.0 * t
  y = 0.5 * 0.003 * x * x
  psi = 0.003 * x
  path = encode_ford_path(_model(t, x, y, psi), T_PREV)

  assert path.path_offset > 0.0
  assert path.path_angle > 0.0
  assert path.curvature == 0.001


def test_c2_uses_same_model_path_as_c0_c1():
  t = np.linspace(0.0, 2.0, 21)
  radius = 1000.0
  theta = 10.0 * t / radius
  path = encode_ford_path(_model(t, radius * np.sin(theta), radius * (1.0 - np.cos(theta)), theta), T_PREV)

  assert path.path_offset > 0.0
  assert path.path_angle > 0.0
  assert path.curvature == 0.001


def test_c3_describes_curvature_change_at_same_path_sample():
  t = np.linspace(0.0, 2.0, 41)
  s = 10.0 * t
  curvature = 0.0005 + 0.00005 * s
  psi = 0.0005 * s + 0.5 * 0.00005 * s * s
  y = np.cumsum(np.pad(np.diff(s) * 0.5 * (np.tan(psi[1:]) + np.tan(psi[:-1])), (1, 0)))
  path = encode_ford_path(_model(t, s, y, psi), T_PREV)

  expected_curvature = float(np.interp(7.0, s, curvature))
  assert np.isclose(path.curvature, expected_curvature, atol=2e-6)
  assert np.isclose(path.curvature_rate, 0.00005, atol=2e-6)


def test_c3_previews_c2_unwind_from_same_path_segment():
  t = np.linspace(0.0, 2.0, 41)
  s = 10.0 * t
  psi = 0.001 * s - 0.5 * 0.0001 * s * s
  y = np.cumsum(np.pad(np.diff(s) * 0.5 * (np.tan(psi[1:]) + np.tan(psi[:-1])), (1, 0)))
  path = encode_ford_path(_model(t, s, y, psi), T_PREV)

  assert path.curvature > 0.0
  assert np.isclose(path.curvature_rate, -0.0001, atol=2e-6)


def test_c3_tracks_path_trend_without_amplifying_sample_noise():
  t = np.linspace(0.0, 2.0, 41)
  s = 10.0 * t
  psi = 0.0005 * s + 0.5 * 0.00005 * s * s
  psi[np.argmin(abs(s - 7.0))] += 0.0005
  y = np.cumsum(np.pad(np.diff(s) * 0.5 * (np.tan(psi[1:]) + np.tan(psi[:-1])), (1, 0)))
  path = encode_ford_path(_model(t, s, y, psi), T_PREV)

  assert np.isclose(path.curvature_rate, 0.00005, atol=1e-5)


def test_keeps_geometric_c2_for_lane_wander_with_continuous_path_feedback():
  t = np.linspace(0.0, 2.0, 21)
  x = 16.0 * t
  y = np.interp(t, [0.0, 0.2, 2.0], [0.0, 0.04, 0.18])
  psi = np.interp(t, [0.0, 0.2, 2.0], [0.0, 0.005, 0.02])
  path = encode_ford_path(_model(t, x, y, psi), T_PREV)

  assert 0.0 < path.path_offset < 0.3
  assert 0.0 < path.path_angle < 0.05
  assert 0.0 < path.curvature < 0.001


def test_clips_to_dbc_range():
  t = np.array([0.0, 0.2, 1.0])
  path = encode_ford_path(_model(t, [0.0, 1.0, 5.0], [10.0, 10.0, 10.0], [1.0, 1.0, 1.0]), T_PREV)

  assert path.path_offset == 5.11
  assert path.path_angle == 0.5235
  assert path.curvature == 0.0

  shallow = encode_ford_path(_model(t, [0.0, 4.0, 20.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]), T_PREV)
  assert shallow.curvature == 0.0


def test_invalid_model_is_inactive():
  path = encode_ford_path(None, T_PREV)

  assert not path.valid
  assert path.path_offset == 0.0
  assert path.path_angle == 0.0
  assert path.curvature == 0.0
  assert path.curvature_rate == 0.0
