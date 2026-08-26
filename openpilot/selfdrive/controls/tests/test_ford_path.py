from types import SimpleNamespace

import numpy as np

from openpilot.selfdrive.controls.lib.ford_path import FordPathController, encode_ford_path


def _model(curvature_rate=0.0):
  distance = np.linspace(0.0, 20.0, 41)
  heading = 0.001 * distance + 0.5 * curvature_rate * distance ** 2
  y = np.cumsum(np.pad(np.diff(distance) * 0.5 * (np.tan(heading[1:]) + np.tan(heading[:-1])), (1, 0)))
  return SimpleNamespace(
    position=SimpleNamespace(x=distance.tolist(), y=y.tolist()),
    orientation=SimpleNamespace(z=heading.tolist()),
  )


def test_c2_carries_full_steady_curvature():
  path = encode_ford_path(_model(), 0.2, 0.01, v_ego=15.0)

  assert path.valid
  assert path.curvature > 0.009
  assert path.path_angle > 0.0


def test_c1_drains_as_c2_fills():
  controller = FordPathController(dt=0.05)
  first = controller.update(_model(), 0.006, v_ego=15.0)
  settled = first
  for _ in range(120):
    settled = controller.update(_model(), 0.006, v_ego=15.0)

  assert first.path_angle > 0.05
  assert abs(settled.path_angle) < 0.002


def test_c1_tracks_applied_not_prelimited_c2():
  controller = FordPathController(dt=0.05)
  path = controller.update(_model(), 0.01, v_ego=15.0, applied_curvature=0.0)

  assert path.curvature > 0.009
  assert path.path_angle > 0.1
  assert abs(controller.c2_response) < 1e-8


def test_c1_actively_cancels_sticky_c2_on_unwind():
  controller = FordPathController(dt=0.05)
  for _ in range(40):
    controller.update(_model(), 0.01, v_ego=15.0)

  unwind = controller.update(_model(), 0.0, v_ego=15.0)

  assert unwind.curvature == 0.0
  assert unwind.path_angle < -0.05


def test_s_turn_commands_opposite_fast_channel_before_c2_drains():
  controller = FordPathController(dt=0.05)
  for _ in range(20):
    controller.update(_model(), 0.01, v_ego=8.0)

  reverse = controller.update(_model(), -0.01, v_ego=8.0)

  assert reverse.curvature < 0.0
  assert reverse.path_angle < -0.1


def test_c0_is_only_low_speed_c1_overflow():
  controller = FordPathController(dt=0.05)
  low_speed = controller.update(_model(), 0.09, v_ego=4.0)
  high_speed = FordPathController(dt=0.05).update(_model(), 0.09, v_ego=15.0)

  assert low_speed.path_angle == 0.5235
  assert low_speed.path_offset > 0.0
  assert high_speed.path_offset == 0.0


def test_c3_is_spatial_curvature_slope():
  path = encode_ford_path(_model(0.00005), 0.2, 0.001, v_ego=12.0)

  assert np.isclose(path.curvature_rate, 0.00005, atol=2e-6)


def test_inactive_or_invalid_resets_slow_state():
  controller = FordPathController(dt=0.05)
  controller.update(_model(), 0.01, v_ego=12.0)

  assert not controller.update(_model(), 0.0, v_ego=12.0, active=False).valid
  assert controller.c2_response == 0.0
  assert not controller.update(None, 0.01, v_ego=12.0).valid
