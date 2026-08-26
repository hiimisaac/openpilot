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


def test_samples_offset_and_heading_at_actuator_delay():
  t = np.linspace(0.0, 2.0, 21)
  y = 2.0 * t
  psi = np.full_like(t, 0.15)
  path = encode_ford_path(_model(t, t, y, psi), T_PREV)

  assert path.valid
  assert path.path_offset == np.interp(T_PREV, t, y)
  assert path.path_angle == 0.15


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
  assert path.path_offset < 2.0  # first content, not the far exit


def test_does_not_walk_past_a_turn_already_at_delay():
  t = np.linspace(0.0, 3.0, 31)
  y = np.where(t <= 0.5, 0.4 + 0.2 * t, np.interp(t, [0.5, 2.0], [0.5, 0.0]))
  psi = np.where(t <= 0.5, 0.12, np.interp(t, [0.5, 2.0], [0.12, 0.0]))
  path = encode_ford_path(_model(t, 10.0 * t, y, psi), T_PREV)

  assert np.isclose(path.path_offset, np.interp(T_PREV, t, y))
  assert np.isclose(path.path_angle, np.interp(T_PREV, t, psi))


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
