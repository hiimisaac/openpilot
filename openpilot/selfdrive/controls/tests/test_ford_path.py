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


def test_path_fields_do_not_feed_measured_curvature_back_into_pscm():
  controller = FordPathController()
  straight_wheel = controller.update(_arc(0.008), v_ego=8.0, applied_curvature=0.0)
  turned_wheel = controller.update(_arc(0.008), v_ego=8.0, applied_curvature=0.008)

  assert straight_wheel.valid
  assert straight_wheel.path_offset == turned_wheel.path_offset
  assert straight_wheel.path_angle == turned_wheel.path_angle
  assert straight_wheel.path_offset > 0.15
  assert straight_wheel.path_angle > 0.05


def test_curve_entry_sends_model_offset_heading_and_curvature_together():
  path = FordPathController().update(_arc(0.008), v_ego=8.0, applied_curvature=0.0)

  assert path.path_offset > 0.15
  assert path.path_angle > 0.05
  assert path.curvature > 0.007


def test_curve_exit_countersteers_retained_c2_directly():
  controller = FordPathController(dt=0.05)
  for _ in range(40):
    controller.update(_arc(0.008), v_ego=8.0, applied_curvature=0.008)
  path = controller.update(_arc(0.0), v_ego=8.0, applied_curvature=0.008)

  assert path.path_offset == 0.0
  assert path.path_angle == 0.0
  assert path.curvature < 0.0


def test_s_turn_uses_c0_c1_immediately_and_drains_old_c2():
  controller = FordPathController(dt=0.05)
  for _ in range(40):
    controller.update(_arc(0.008), v_ego=8.0, applied_curvature=0.008)
  path = controller.update(_arc(-0.008), v_ego=8.0, applied_curvature=0.008)

  assert path.path_offset < -0.15
  assert path.path_angle < -0.05
  assert path.curvature < 0.0


def test_desired_curvature_noise_does_not_move_straight_aligned_path():
  controller = FordPathController()
  quiet = controller.update(_arc(0.0), -0.003, v_ego=15.0, applied_curvature=0.0)
  noisy = controller.update(_arc(0.0), 0.003, v_ego=15.0, applied_curvature=0.0)

  assert quiet == noisy
  assert quiet.path_offset == 0.0
  assert quiet.path_angle == 0.0


def test_c2_and_c3_are_target_path_geometry_at_same_reference():
  path = encode_ford_path(_arc(0.008), 0.2)

  assert np.isclose(path.curvature, 0.008, atol=2e-5)
  assert abs(path.curvature_rate) < 6e-6


def test_invalid_or_inactive_returns_no_path():
  controller = FordPathController()

  assert not controller.update(_arc(0.0), v_ego=12.0, applied_curvature=0.0, active=False).valid
  assert not controller.update(None, v_ego=12.0, applied_curvature=0.0).valid
