import math
from types import SimpleNamespace

import numpy as np

from openpilot.selfdrive.controls.lib.ford_path import FordPathController, encode_ford_path


def _path(curvature: float, curvature_rate: float = 0.0, speed: float = 8.0):
  t = np.linspace(0.0, 3.0, 61)
  distance = speed * t
  heading = curvature * distance + 0.5 * curvature_rate * distance ** 2
  x = np.zeros_like(distance)
  y = np.zeros_like(distance)
  for i in range(1, len(distance)):
    ds = distance[i] - distance[i - 1]
    average_heading = 0.5 * (heading[i] + heading[i - 1])
    x[i] = x[i - 1] + ds * math.cos(average_heading)
    y[i] = y[i - 1] + ds * math.sin(average_heading)
  return SimpleNamespace(
    position=SimpleNamespace(t=t.tolist(), x=x.tolist(), y=y.tolist()),
    orientation=SimpleNamespace(z=heading.tolist()),
  )


def _delayed_turn(onset: float = 0.5, speed: float = 8.0, curvature_rate: float = 0.002):
  t = np.linspace(0.0, 3.0, 61)
  distance = speed * t
  turn_distance = np.maximum(distance - speed * onset, 0.0)
  heading = 0.5 * curvature_rate * turn_distance ** 2
  y = curvature_rate * turn_distance ** 3 / 6.0
  return SimpleNamespace(
    position=SimpleNamespace(t=t.tolist(), x=distance.tolist(), y=y.tolist()),
    orientation=SimpleNamespace(z=heading.tolist()),
  )


def test_fast_fields_are_errors_from_pose_at_actuation_time():
  controller = FordPathController()
  aligned = controller.update(_path(0.008), v_ego=8.0, current_curvature=0.008, actuator_delay=0.4)
  behind = controller.update(_path(0.008), v_ego=8.0, current_curvature=0.0, actuator_delay=0.4)

  assert aligned.valid and behind.valid
  assert abs(aligned.path_offset) < 2e-3
  assert abs(aligned.path_angle) < 2e-3
  assert behind.path_offset > aligned.path_offset
  assert behind.path_angle > aligned.path_angle


def test_lateral_delay_sets_when_future_turn_reaches_fast_fields():
  controller = FordPathController()
  before = controller.update(_delayed_turn(), v_ego=8.0, current_curvature=0.0, actuator_delay=0.2)
  after = controller.update(_delayed_turn(), v_ego=8.0, current_curvature=0.0, actuator_delay=0.8)

  assert abs(before.path_offset) < 1e-9
  assert abs(before.path_angle) < 1e-9
  assert abs(before.curvature) < 1e-9
  assert abs(before.curvature_rate) < 1e-9
  assert after.path_offset > 0.0
  assert after.path_angle > 0.0


def test_departure_model_creates_fast_error_while_vehicle_is_stopped():
  path = FordPathController().update(_path(0.02, speed=2.0), v_ego=0.0, current_curvature=0.0, actuator_delay=0.5)

  assert path.valid
  assert path.path_offset > 0.0
  assert path.path_angle > 0.0


def test_c2_lead_is_continuous_and_symmetric():
  controller = FordPathController()
  entering = controller.update(_path(0.004, 0.0004, speed=10.0), v_ego=10.0, actuator_delay=0.4)
  leaving = controller.update(_path(0.008, -0.0004, speed=10.0), v_ego=10.0, actuator_delay=0.4)

  assert entering.curvature_rate > 0.0
  assert leaving.curvature_rate < 0.0
  assert entering.curvature > 0.004 + 0.0004 * 4.0
  assert leaving.curvature < 0.008 - 0.0004 * 4.0


def test_curve_exit_drains_c2_without_an_unwind_state():
  path = FordPathController().update(_path(0.008, -0.001, speed=8.0), v_ego=8.0,
                                     current_curvature=0.008, actuator_delay=0.5)

  assert path.curvature_rate < 0.0
  assert path.curvature < 0.0


def test_s_turn_uses_c0_c1_against_retained_vehicle_curvature():
  path = FordPathController().update(_path(-0.008), v_ego=8.0, current_curvature=0.008, actuator_delay=0.4)

  assert path.path_offset < 0.0
  assert path.path_angle < 0.0
  assert path.curvature < 0.0


def test_desired_curvature_noise_does_not_move_aligned_path():
  controller = FordPathController()
  quiet = controller.update(_path(0.0), -0.003, v_ego=15.0, current_curvature=0.0, actuator_delay=0.4)
  noisy = controller.update(_path(0.0), 0.003, v_ego=15.0, current_curvature=0.0, actuator_delay=0.4)

  assert quiet == noisy
  assert quiet.path_offset == 0.0
  assert quiet.path_angle == 0.0


def test_c2_and_c3_are_local_path_geometry_with_c2_lead():
  path = encode_ford_path(_path(0.008), 0.2, v_ego=8.0, actuator_delay=0.4)

  assert np.isclose(path.curvature, 0.008, atol=2e-5)
  assert abs(path.curvature_rate) < 6e-6


def test_invalid_or_inactive_returns_no_path():
  controller = FordPathController()

  assert not controller.update(_path(0.0), v_ego=12.0, active=False).valid
  assert not controller.update(None, v_ego=12.0).valid
