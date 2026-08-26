from types import SimpleNamespace

import numpy as np

from openpilot.selfdrive.controls.lib.ford_path import FordPathController, encode_ford_path


def _model(coefficients=(0.0, 0.0, 0.0, 0.0)):
  x = np.linspace(0.0, 20.0, 41)
  y = np.polynomial.polynomial.polyval(x, coefficients)
  slope = np.polynomial.polynomial.polyval(x, np.polynomial.polynomial.polyder(coefficients))
  return SimpleNamespace(
    position=SimpleNamespace(x=x.tolist(), y=y.tolist()),
    orientation=SimpleNamespace(z=np.arctan(slope).tolist()),
  )


def test_all_fields_describe_one_cubic_at_one_preview_point():
  coefficients = (0.1, 0.02, 0.001, 0.00001)
  path = encode_ford_path(_model(coefficients), 0.2, v_ego=12.0)

  x = 7.0
  slope = coefficients[1] + 2.0 * coefficients[2] * x + 3.0 * coefficients[3] * x ** 2
  second = 2.0 * coefficients[2] + 6.0 * coefficients[3] * x
  third = 6.0 * coefficients[3]
  expected_curvature = second / (1.0 + slope ** 2) ** 1.5
  expected_rate = third / (1.0 + slope ** 2) ** 2 - 3.0 * slope * second ** 2 / (1.0 + slope ** 2) ** 3

  assert path.valid
  assert np.isclose(path.path_offset, np.polynomial.polynomial.polyval(x, coefficients), atol=1e-8)
  assert np.isclose(path.path_angle, np.arctan(slope), atol=1e-8)
  assert np.isclose(path.curvature, expected_curvature, atol=1e-8)
  assert np.isclose(path.curvature_rate, expected_rate, atol=1e-8)


def test_straight_path_does_not_translate_desired_curvature_noise_into_c1():
  controller = FordPathController(dt=0.05)
  quiet = controller.update(_model(), -0.003, v_ego=15.0, applied_curvature=0.0)
  noisy = controller.update(_model(), 0.003, v_ego=15.0, applied_curvature=0.0)

  assert quiet == noisy
  assert quiet.path_angle == 0.0
  assert quiet.curvature == 0.0


def test_small_c2_retention_does_not_chatter_straight_path():
  controller = FordPathController(dt=0.05)
  controller.c2_response = 0.0015

  path = controller.update(_model(), v_ego=15.0, applied_curvature=0.0015)

  assert path.path_angle == 0.0


def test_turn_entry_uses_geometric_offset_and_heading_without_c2_fill_correction():
  path = FordPathController(dt=0.05).update(_model((0.0, 0.0, 0.004, 0.0)), v_ego=8.0, applied_curvature=0.0)

  assert path.path_offset > 0.15
  assert path.path_angle > 0.05
  assert path.curvature > 0.007


def test_c1_countersteers_retained_c2_during_unwind():
  controller = FordPathController(dt=0.05)
  curve = _model((0.0, 0.0, 0.004, 0.0))
  for _ in range(40):
    controller.update(curve, v_ego=15.0, applied_curvature=0.008)

  unwind = controller.update(_model(), v_ego=15.0, applied_curvature=0.0)

  assert unwind.curvature == 0.0
  assert unwind.path_offset == 0.0
  assert unwind.path_angle < -0.05


def test_s_turn_countersteers_old_c2_in_addition_to_new_polynomial():
  controller = FordPathController(dt=0.05)
  right = _model((0.0, 0.0, 0.004, 0.0))
  left = _model((0.0, 0.0, -0.004, 0.0))
  for _ in range(40):
    controller.update(right, v_ego=8.0, applied_curvature=0.008)

  reverse = controller.update(left, v_ego=8.0, applied_curvature=-0.008)

  geometric = encode_ford_path(left, 0.2, v_ego=8.0)
  assert reverse.curvature < 0.0
  assert reverse.path_offset < 0.0
  assert reverse.path_angle < geometric.path_angle


def test_invalid_or_inactive_resets_c2_state():
  controller = FordPathController(dt=0.05)
  controller.update(_model((0.0, 0.0, 0.004, 0.0)), v_ego=12.0, applied_curvature=0.008)

  assert not controller.update(_model(), v_ego=12.0, active=False).valid
  assert controller.c2_response == 0.0
  assert not controller.update(None, v_ego=12.0).valid
