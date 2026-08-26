from types import SimpleNamespace

import numpy as np

from openpilot.selfdrive.controls.lib.ford_path import FordPathController, encode_ford_path


def _arc(curvature):
  x = np.linspace(0.0, 20.0, 41)
  if curvature == 0.0:
    y = np.zeros_like(x)
    heading = np.zeros_like(x)
  else:
    kx = np.clip(curvature * x, -0.99, 0.99)
    y = (1.0 - np.sqrt(1.0 - kx ** 2)) / curvature
    heading = np.arcsin(kx)
  return SimpleNamespace(
    position=SimpleNamespace(x=x.tolist(), y=y.tolist()),
    orientation=SimpleNamespace(z=heading.tolist()),
  )


def test_aligned_curve_has_zero_path_and_heading_error():
  path = FordPathController().update(_arc(0.008), v_ego=8.0, current_curvature=0.008)

  assert path.valid
  assert abs(path.path_offset) < 1e-8
  assert abs(path.path_angle) < 1e-8
  assert path.curvature > 0.007


def test_curve_entry_is_target_pose_minus_straight_vehicle_prediction():
  path = FordPathController().update(_arc(0.008), v_ego=8.0, current_curvature=0.0)

  assert path.path_offset > 0.15
  assert path.path_angle > 0.05
  assert path.curvature > 0.007


def test_curve_exit_countersteers_measured_wheel_curvature():
  path = FordPathController().update(_arc(0.0), v_ego=8.0, current_curvature=0.008)

  assert path.path_offset < -0.15
  assert path.path_angle < -0.05
  assert path.curvature == 0.0


def test_s_turn_error_includes_old_wheel_direction():
  path = FordPathController().update(_arc(-0.008), v_ego=8.0, current_curvature=0.008)
  entry = FordPathController().update(_arc(-0.008), v_ego=8.0, current_curvature=0.0)

  assert path.path_offset < entry.path_offset
  assert path.path_angle < entry.path_angle
  assert path.curvature < 0.0


def test_desired_curvature_noise_does_not_move_straight_aligned_path():
  controller = FordPathController()
  quiet = controller.update(_arc(0.0), -0.003, v_ego=15.0, current_curvature=0.0)
  noisy = controller.update(_arc(0.0), 0.003, v_ego=15.0, current_curvature=0.0)

  assert quiet == noisy
  assert quiet.path_offset == 0.0
  assert quiet.path_angle == 0.0


def test_c2_and_c3_are_target_path_geometry_at_same_reference():
  path = encode_ford_path(_arc(0.008), 0.2, current_curvature=0.0)

  assert np.isclose(path.curvature, 0.008, atol=2e-5)
  assert abs(path.curvature_rate) < 2e-6


def test_invalid_or_inactive_returns_no_path():
  controller = FordPathController()

  assert not controller.update(_arc(0.0), v_ego=12.0, current_curvature=0.0, active=False).valid
  assert not controller.update(None, v_ego=12.0, current_curvature=0.0).valid
