import math
from types import SimpleNamespace

from openpilot.selfdrive.controls.lib.lateral_path import model_lateral_path


def polynomial_model(curvature: float, curvature_rate: float = 0.0,
                     path_offset: float = 0.0, path_angle: float = 0.0):
  distances = (0.0, 1.75, 2.5, 3.5, 5.0, 7.0, 12.0, 20.0)
  return SimpleNamespace(
    position=SimpleNamespace(
      x=list(distances),
      y=[path_offset + path_angle * s + 0.5 * curvature * s ** 2 + curvature_rate * s ** 3 / 6.0
         for s in distances],
    ),
    orientation=SimpleNamespace(
      z=[path_angle + curvature * s + 0.5 * curvature_rate * s ** 2 for s in distances],
    ),
  )


def test_model_lateral_path_builds_complete_polynomial_target():
  target = model_lateral_path(polynomial_model(0.01, 0.0004, 0.12, -0.015), 0.01, 7.0)

  assert target.valid
  assert math.isclose(target.path_offset, 0.12)
  assert math.isclose(target.path_angle, -0.015)
  assert math.isclose(target.curvature, 0.01)
  assert math.isclose(target.curvature_rate, 0.0004, rel_tol=0.01)


def test_model_lateral_path_is_spatial_and_speed_independent():
  slow_target = model_lateral_path(polynomial_model(0.01, 0.0002), 0.01, 4.0)
  fast_target = model_lateral_path(polynomial_model(0.01, 0.0002), 0.01, 25.0)

  assert slow_target == fast_target


def test_model_lateral_path_does_not_turn_action_mismatch_into_curvature_rate():
  target = model_lateral_path(polynomial_model(0.01), 0.004, 7.0)

  assert target.valid
  assert target.curvature == 0.004
  assert math.isclose(target.curvature_rate, 0.0, abs_tol=1e-5)


def test_model_lateral_path_preserves_action_when_model_is_invalid():
  target = model_lateral_path(None, -0.004, 7.0)

  assert not target.valid
  assert target.path_offset == 0.0
  assert target.path_angle == 0.0
  assert math.isclose(target.curvature, -0.004)
  assert target.curvature_rate == 0.0
