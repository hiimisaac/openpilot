import math
from types import SimpleNamespace

from openpilot.selfdrive.controls.lib.lateral_path import model_lateral_path


def polynomial_model(curvature: float, curvature_rate: float = 0.0):
  distances = tuple(index * 0.5 for index in range(61))
  return SimpleNamespace(
    position=SimpleNamespace(
      x=list(distances),
      y=[0.5 * curvature * s ** 2 + curvature_rate * s ** 3 / 6.0 for s in distances],
    ),
    orientation=SimpleNamespace(
      z=[curvature * s + 0.5 * curvature_rate * s ** 2 for s in distances],
    ),
  )


def delayed_turn_model(onset: float = 10.0, curvature_rate: float = 0.001):
  distances = tuple(index * 0.5 for index in range(61))
  turn_distances = [max(distance - onset, 0.0) for distance in distances]
  return SimpleNamespace(
    position=SimpleNamespace(
      x=list(distances),
      y=[curvature_rate * distance ** 3 / 6.0 for distance in turn_distances],
    ),
    orientation=SimpleNamespace(
      z=[curvature_rate * distance ** 2 / 2.0 for distance in turn_distances],
    ),
  )


def test_model_lateral_path_builds_complete_polynomial_target():
  target = model_lateral_path(polynomial_model(0.004, 0.0004), 0.004, 20.0)

  assert target.valid
  assert math.isclose(target.path_offset, 0.0, abs_tol=1e-12)
  assert math.isclose(target.path_angle, 0.0, abs_tol=1e-12)
  assert math.isclose(target.curvature, 0.004)
  assert math.isclose(target.curvature_rate, 0.0004, rel_tol=0.01)


def test_model_lateral_path_uses_one_second_preview_for_distant_turn_onset():
  low_speed = model_lateral_path(delayed_turn_model(), 0.0, 5.0)
  high_speed = model_lateral_path(delayed_turn_model(), 0.0, 25.0)

  assert low_speed.valid and high_speed.valid
  assert math.isclose(low_speed.path_offset, 0.0, abs_tol=1e-12)
  assert math.isclose(low_speed.path_angle, 0.0, abs_tol=1e-12)
  assert math.isclose(low_speed.curvature_rate, 0.0, abs_tol=1e-12)
  assert abs(high_speed.path_offset) + abs(high_speed.path_angle) + abs(high_speed.curvature_rate) > 0.001


def test_model_lateral_path_uses_valid_prefix_when_tight_turn_recedes_in_x():
  model = polynomial_model(0.004, 0.0004)
  model.position.x = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 9.0, 8.0]
  model.position.y = [0.5 * 0.004 * x ** 2 + 0.0004 * x ** 3 / 6.0 for x in model.position.x]
  model.orientation.z = [0.004 * x + 0.5 * 0.0004 * x ** 2 for x in model.position.x]

  target = model_lateral_path(model, 0.004, 10.0)

  assert target.valid
  assert math.isclose(target.curvature, 0.004)
  assert math.isclose(target.curvature_rate, 0.0004, rel_tol=0.05)


def test_model_lateral_path_gives_c2_full_ownership_of_ordinary_curvature():
  target = model_lateral_path(polynomial_model(0.004), 0.004, 20.0, desired_angle_deg=20.0)

  assert math.isclose(target.curvature, 0.004)


def test_model_lateral_path_fades_c2_continuously_for_large_steering():
  halfway = model_lateral_path(polynomial_model(0.004), 0.004, 20.0, desired_angle_deg=57.5)
  large_turn = model_lateral_path(polynomial_model(0.004), 0.004, 20.0, desired_angle_deg=80.0)

  assert math.isclose(halfway.curvature, 0.002)
  assert large_turn.curvature == 0.0
  assert abs(large_turn.path_offset) + abs(large_turn.path_angle) + abs(large_turn.curvature_rate) > 0.0


def test_model_lateral_path_fades_c2_for_large_model_curvature():
  target = model_lateral_path(polynomial_model(0.018), 0.018, 20.0, desired_angle_deg=0.0)

  assert target.curvature == 0.0


def test_model_lateral_path_refits_large_turn_into_the_coefficient_envelope():
  target = model_lateral_path(polynomial_model(0.018), 0.018, 20.0, desired_angle_deg=80.0)

  assert -5.11 <= target.path_offset <= 5.12
  assert -0.5235 <= target.path_angle <= 0.5
  assert -0.02 <= target.curvature <= 0.02
  assert -0.001023 <= target.curvature_rate <= 0.001024
  assert math.isclose(target.curvature_rate, 0.001024)


def test_model_lateral_path_has_no_c2_memory():
  large_turn = model_lateral_path(polynomial_model(0.018), 0.018, 20.0, desired_angle_deg=80.0)
  ordinary = model_lateral_path(polynomial_model(0.004), 0.004, 20.0, desired_angle_deg=20.0)

  assert large_turn.curvature == 0.0
  assert math.isclose(ordinary.curvature, 0.004)


def test_model_lateral_path_preserves_action_when_model_is_invalid():
  target = model_lateral_path(None, -0.004, 7.0)

  assert not target.valid
  assert target.path_offset == 0.0
  assert target.path_angle == 0.0
  assert math.isclose(target.curvature, -0.004)
  assert target.curvature_rate == 0.0
