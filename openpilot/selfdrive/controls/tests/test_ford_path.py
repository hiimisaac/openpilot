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
  path = encode_ford_path(_model(t, R * np.sin(th), R * (1.0 - np.cos(th)), th), T_PREV, 1.0 / R)

  bumper_psi = np.interp(T_PREV, t, th)
  assert path.valid
  assert abs(path.path_angle) > 5.0 * abs(bumper_psi)
  assert abs(path.path_offset) > 0.1
  assert path.path_angle > 0.0


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


def test_clips_to_dbc_range():
  t = np.array([0.0, 0.2, 1.0])
  path = encode_ford_path(_model(t, [0.0, 1.0, 5.0], [10.0, 10.0, 10.0], [1.0, 1.0, 1.0],
                                 desired_curvature=0.05), T_PREV, desired_curvature=0.05)

  assert path.path_offset == 5.11
  assert path.path_angle == 0.5235
  assert path.curvature == 0.02


def test_invalid_model_is_inactive():
  path = encode_ford_path(None, T_PREV)

  assert not path.valid
  assert path.path_offset == 0.0
  assert path.path_angle == 0.0
  assert path.curvature == 0.0
  assert path.curvature_rate == 0.0
