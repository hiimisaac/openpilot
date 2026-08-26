import math
from types import SimpleNamespace

import pytest

from openpilot.selfdrive.controls.lib.lateral_path import (
  SPATIAL_CURVATURE_RATE_LIMITS, FrenetErrorObserver, PersistentLateralPath, _PathSample, _ReferencePath,
)


DT = 0.01
NS = 1_000_000_000


def model_path(xs, ys, headings, timestamp_ns: int, frame_id: int = 1):
  return SimpleNamespace(
    frameId=frame_id,
    timestampEof=timestamp_ns,
    position=SimpleNamespace(x=list(xs), y=list(ys)),
    orientation=SimpleNamespace(z=list(headings)),
  )


def polynomial_model(curvature: float, curvature_rate: float, timestamp_ns: int, frame_id: int = 1):
  distances = [index * 0.25 for index in range(161)]
  return model_path(
    distances,
    [0.5 * curvature * s ** 2 + curvature_rate * s ** 3 / 6.0 for s in distances],
    [curvature * s + 0.5 * curvature_rate * s ** 2 for s in distances],
    timestamp_ns,
    frame_id,
  )


def circle_model(radius: float, timestamp_ns: int, frame_id: int = 1, arc_deg: float = 120.0):
  direction = math.copysign(1.0, radius)
  radius = abs(radius)
  angles = [math.radians(arc_deg) * index / 160.0 for index in range(161)]
  return model_path(
    [radius * math.sin(angle) for angle in angles],
    [direction * radius * (1.0 - math.cos(angle)) for angle in angles],
    [direction * angle for angle in angles],
    timestamp_ns,
    frame_id,
  )


def straight_model(timestamp_ns: int, frame_id: int = 1):
  distances = [index * 0.25 for index in range(161)]
  return model_path(distances, [0.0] * len(distances), [0.0] * len(distances), timestamp_ns, frame_id)


def delayed_clothoid_model(timestamp_ns: int, frame_id: int = 1, onset: float = 2.0):
  spacing = 0.25
  distances = [index * spacing for index in range(161)]
  xs = [0.0]
  ys = [0.0]
  headings = [0.0]
  for distance in distances[1:]:
    previous_distance = distance - spacing
    midpoint = 0.5 * (distance + previous_distance)
    heading = 0.004 * max(midpoint - onset, 0.0) ** 2
    xs.append(xs[-1] + spacing * math.cos(heading))
    ys.append(ys[-1] + spacing * math.sin(heading))
    headings.append(0.004 * max(distance - onset, 0.0) ** 2)
  return model_path(xs, ys, headings, timestamp_ns, frame_id)


def parametric_clothoid_model(initial_heading: float, curvature: float, curvature_rate: float,
                              timestamp_ns: int, frame_id: int = 1):
  spacing = 0.25
  stations = [index * spacing for index in range(161)]
  headings = [initial_heading + curvature * station + 0.5 * curvature_rate * station ** 2 for station in stations]
  xs = [0.0]
  ys = [0.0]
  for first, second in zip(headings, headings[1:], strict=False):
    midpoint = 0.5 * (first + second)
    xs.append(xs[-1] + spacing * math.cos(midpoint))
    ys.append(ys[-1] + spacing * math.sin(midpoint))
  return model_path(xs, ys, headings, timestamp_ns, frame_id)


def step(controller: PersistentLateralPath, timestamp_ns: int, *, model=None,
         speed=10.0, yaw_rate=0.0, active=True):
  return controller.update(
    model,
    active=active,
    mono_time_ns=timestamp_ns,
    v_ego=speed,
    yaw_rate=yaw_rate,
  )


def test_frenet_observer_predicts_patent_kinematics():
  observer = FrenetErrorObserver()
  observer.update(NS, speed=10.0, reference_curvature=0.01, yaw_rate=0.05, measurement=(0.2, 0.03))

  estimate = None
  for index in range(1, 11):
    estimate = observer.update(NS + index * 10_000_000, speed=10.0, reference_curvature=0.01, yaw_rate=0.05)

  assert estimate is not None
  assert estimate[0] == pytest.approx(0.23225)
  assert estimate[1] == pytest.approx(0.035)


def test_frenet_observer_correction_is_bounded_not_a_measurement_snap():
  observer = FrenetErrorObserver()
  observer.update(NS, speed=0.0, reference_curvature=0.0, yaw_rate=0.0, measurement=(0.0, 0.0))

  estimate = observer.update(NS + 50_000_000, speed=0.0, reference_curvature=0.0, yaw_rate=0.0,
                             measurement=(0.2, 0.1))

  assert estimate is not None
  assert 0.0 < estimate[0] < 0.2
  assert 0.0 < estimate[1] < 0.1


def test_frenet_observer_reset_clears_covariance_and_state():
  observer = FrenetErrorObserver()
  observer.update(NS, speed=5.0, reference_curvature=0.01, yaw_rate=0.0, measurement=(0.1, 0.02))
  observer.reset()

  assert observer.state is None
  assert observer.update(NS + 50_000_000, speed=5.0, reference_curvature=0.01, yaw_rate=0.0) is None


def test_constant_curve_is_one_clean_local_jet():
  controller = PersistentLateralPath()
  target = step(controller, NS, model=polynomial_model(0.004, 0.0004, NS))

  assert target.valid
  assert target.path_offset == pytest.approx(0.0, abs=1e-7)
  assert target.path_angle == pytest.approx(0.0, abs=1e-7)
  assert target.curvature == pytest.approx(0.004, rel=0.02)
  assert target.curvature_rate == pytest.approx(0.0004, rel=0.06)


def test_curvature_is_evaluated_at_current_station_not_averaged_from_future_turn():
  controller = PersistentLateralPath()
  target = step(controller, NS, model=delayed_clothoid_model(NS))

  assert target.valid
  assert target.curvature == pytest.approx(0.0, abs=1e-7)
  assert target.curvature_rate == pytest.approx(0.0, abs=1e-7)


def test_position_geometry_owns_the_complete_local_jet():
  consistent = PersistentLateralPath()
  inconsistent = PersistentLateralPath()
  path = circle_model(80.0, NS)
  inconsistent_path = circle_model(80.0, NS)
  inconsistent_path.orientation.z = [0.0] * len(inconsistent_path.orientation.z)

  expected = step(consistent, NS, model=path)
  target = step(inconsistent, NS, model=inconsistent_path)

  assert target.valid
  assert target.path_offset == pytest.approx(expected.path_offset, abs=1e-8)
  assert target.path_angle == pytest.approx(expected.path_angle, abs=1e-8)
  assert target.curvature == pytest.approx(expected.curvature, abs=1e-8)
  assert target.curvature_rate == pytest.approx(expected.curvature_rate, abs=1e-8)


@pytest.mark.parametrize("heading", [0.0, 0.2, -0.2, 0.45, -0.45])
def test_curvature_rate_is_expressed_in_vehicle_longitudinal_distance(heading: float):
  controller = PersistentLateralPath()
  curvature_rate_ds = 0.0002
  target = step(
    controller, NS,
    model=parametric_clothoid_model(heading, 0.004, curvature_rate_ds, NS),
  )

  assert target.valid
  assert target.curvature == pytest.approx(0.004, rel=0.02)
  assert target.curvature_rate == pytest.approx(curvature_rate_ds / math.cos(heading), rel=0.005)


def test_bounded_wire_curvature_advances_at_the_transmitted_curvature_rate():
  controller = PersistentLateralPath()
  first = step(controller, NS, model=polynomial_model(0.0, 0.01, NS), speed=10.0)
  advanced = step(controller, NS + 50_000_000, speed=10.0)

  assert first.valid and advanced.valid
  assert first.curvature_rate == pytest.approx(SPATIAL_CURVATURE_RATE_LIMITS[1], abs=1e-12)
  assert advanced.curvature - first.curvature == pytest.approx(first.curvature_rate * 0.5, abs=2e-6)
  assert advanced.curvature_rate == pytest.approx(first.curvature_rate, rel=1e-5)


def test_projection_hint_prevents_branch_jump_at_self_intersection():
  path = _ReferencePath([
    _PathSample(-2.0, -2.0, math.pi / 4.0),
    _PathSample(0.0, 0.0, math.pi / 4.0),
    _PathSample(2.0, 2.0, math.pi / 4.0),
    _PathSample(4.0, 0.0, -math.pi / 4.0),
    _PathSample(2.0, -2.0, -3.0 * math.pi / 4.0),
    _PathSample(0.0, 0.0, 3.0 * math.pi / 4.0),
    _PathSample(-2.0, 2.0, 3.0 * math.pi / 4.0),
  ])

  first_branch = path.project(0.0, 0.0, heading=math.pi / 4.0, station_hint=2.8, min_station=2.0, max_station=4.0)
  second_branch = path.project(0.0, 0.0, heading=3.0 * math.pi / 4.0, station_hint=14.1, min_station=13.0, max_station=15.0)

  assert first_branch.station < 4.0
  assert second_branch.station > 13.0


def test_large_driver_motion_preserves_observer_across_segment_promotion():
  controller = PersistentLateralPath()
  assert step(controller, NS, model=circle_model(25.0, NS), speed=8.0).valid
  before = None
  for index in range(1, 11):
    before = step(controller, NS + index * 10_000_000, speed=8.0, yaw_rate=0.0)
  assert before is not None
  assert abs(before.path_offset) > 1e-3
  assert abs(before.path_angle) > math.radians(0.5)

  target = step(
    controller,
    NS + 150_000_000,
    model=straight_model(NS + 150_000_000, frame_id=2),
    speed=0.0,
    yaw_rate=8.0,
  )

  assert target.valid
  assert target.path_offset * before.path_offset > 0.0
  assert abs(target.path_offset) > 1e-3
  assert target.curvature == before.curvature


def test_model_is_assimilated_only_when_source_key_changes():
  controller = PersistentLateralPath()
  step(controller, NS, model=straight_model(NS))
  mutated = circle_model(20.0, NS, frame_id=1)

  target = step(controller, NS + int(DT * NS), model=mutated, speed=0.0)

  assert target.valid
  assert target.curvature == pytest.approx(0.0, abs=1e-6)


def test_engagement_bootstraps_from_latest_valid_model_between_model_frames():
  controller = PersistentLateralPath()
  latest_model = circle_model(50.0, NS)
  assert not step(controller, NS, model=latest_model, active=False).valid

  target = step(controller, NS + 10_000_000, model=latest_model, active=True)

  assert target.valid
  assert target.curvature == pytest.approx(0.02, rel=0.03)


def test_duplicate_model_frames_do_not_extend_source_freshness():
  controller = PersistentLateralPath()
  model = straight_model(NS)
  target = step(controller, NS, model=model)
  assert target.valid

  for index in range(1, 8):
    target = step(controller, NS + index * 50_000_000, model=model)

  assert not target.valid


def test_delayed_model_frame_is_compensated_to_current_pose():
  radius = 100.0
  speed = 10.0
  yaw_rate = speed / radius
  on_time = PersistentLateralPath()
  delayed = PersistentLateralPath()
  step(on_time, NS, speed=speed, yaw_rate=yaw_rate)
  step(delayed, NS, speed=speed, yaw_rate=yaw_rate)

  timestamp_ns = NS + 100_000_000
  on_time_target = step(
    on_time, timestamp_ns,
    model=circle_model(radius, timestamp_ns, frame_id=2),
    speed=speed, yaw_rate=yaw_rate,
  )
  delayed_target = step(
    delayed, timestamp_ns,
    model=circle_model(radius, NS, frame_id=2),
    speed=speed, yaw_rate=yaw_rate,
  )

  assert delayed_target.valid
  assert delayed_target.path_offset == pytest.approx(on_time_target.path_offset, abs=2e-3)
  assert delayed_target.path_angle == pytest.approx(on_time_target.path_angle, abs=2e-4)
  assert delayed_target.curvature == pytest.approx(on_time_target.curvature, abs=2e-4)
  assert delayed_target.curvature_rate == pytest.approx(on_time_target.curvature_rate, abs=2e-4)


def test_too_old_model_timestamp_fails_closed():
  controller = PersistentLateralPath()
  step(controller, NS)

  target = step(
    controller, NS + 100_000_000,
    model=straight_model(NS - 100_000_000, frame_id=2),
  )

  assert not target.valid


def test_ego_relative_replans_do_not_erase_observed_frenet_error():
  controller = PersistentLateralPath()
  step(controller, NS, model=circle_model(20.0, NS), speed=10.0)

  before = None
  for index in range(1, 6):
    before = step(controller, NS + index * 10_000_000, speed=10.0, yaw_rate=0.0)
  assert before is not None
  after = step(
    controller,
    NS + 50_000_000,
    model=circle_model(20.0, NS + 50_000_000, frame_id=2),
    speed=10.0,
    yaw_rate=0.0,
  )

  assert abs(before.path_offset) > 0.002
  assert abs(before.path_angle) > math.radians(0.5)
  assert after.path_offset * before.path_offset > 0.0
  assert after.path_angle * before.path_angle > 0.0
  assert abs(after.path_offset) > 0.7 * abs(before.path_offset)
  assert abs(after.path_angle) > 0.7 * abs(before.path_angle)


def test_fresh_far_field_replaces_old_plan_without_resetting_near_field():
  controller = PersistentLateralPath()
  step(controller, NS, model=straight_model(NS), speed=10.0)
  changed = polynomial_model(0.0, 0.001, NS + 50_000_000, frame_id=2)

  at_splice = step(controller, NS + 50_000_000, model=changed, speed=10.0)
  assert at_splice.path_offset == pytest.approx(0.0, abs=1e-4)
  assert at_splice.path_angle == pytest.approx(0.0, abs=3e-4)

  target = at_splice
  for index in range(1, 101):
    timestamp_ns = NS + 50_000_000 + index * 10_000_000
    model_updated = index % 5 == 0
    target = step(
      controller,
      timestamp_ns,
      model=polynomial_model(0.0, 0.001, timestamp_ns, frame_id=index // 5 + 2) if model_updated else None,
      speed=10.0,
    )
  assert target.valid
  assert target.path_offset > 0.005
  assert target.path_angle > 0.003


def test_replan_only_updates_pending_segment_and_preserves_curvature_at_boundary():
  controller = PersistentLateralPath()
  step(controller, NS, model=straight_model(NS), speed=10.0)

  pending = step(
    controller,
    NS + 50_000_000,
    model=polynomial_model(0.01, 0.0004, NS + 50_000_000, frame_id=2),
    speed=10.0,
  )
  before_boundary = step(controller, NS + 90_000_000, speed=10.0)
  after_boundary = step(controller, NS + 100_000_000, speed=10.0)

  assert pending.curvature == pytest.approx(0.0, abs=1e-7)
  assert pending.curvature_rate == pytest.approx(0.0, abs=1e-7)
  assert before_boundary.curvature == pytest.approx(0.0, abs=1e-7)
  assert before_boundary.curvature_rate == pytest.approx(0.0, abs=1e-7)
  assert after_boundary.curvature == pytest.approx(0.0, abs=2e-6)
  assert after_boundary.curvature_rate == pytest.approx(SPATIAL_CURVATURE_RATE_LIMITS[1], abs=1e-9)


def test_low_speed_segment_stays_short_enough_to_promote_new_turn():
  controller = PersistentLateralPath()
  step(controller, NS, model=straight_model(NS), speed=2.0)
  step(
    controller,
    NS + 50_000_000,
    model=polynomial_model(0.01, 0.0004, NS + 50_000_000, frame_id=2),
    speed=2.0,
  )

  before_boundary = step(controller, NS + 240_000_000, speed=2.0)
  promoted = step(controller, NS + 250_000_000, speed=2.0)

  assert before_boundary.curvature == pytest.approx(0.0, abs=1e-7)
  assert promoted.curvature == pytest.approx(0.0, abs=2e-6)
  assert promoted.curvature_rate == pytest.approx(SPATIAL_CURVATURE_RATE_LIMITS[1], abs=1e-9)


def test_pending_geometry_cannot_change_any_active_coefficient():
  baseline = PersistentLateralPath()
  changing = PersistentLateralPath()
  step(baseline, NS, model=straight_model(NS), speed=10.0)
  step(changing, NS, model=straight_model(NS), speed=10.0)

  for index in range(1, 10):
    timestamp_ns = NS + index * 10_000_000
    model_updated = index % 5 == 0
    baseline_target = step(
      baseline,
      timestamp_ns,
      model=straight_model(timestamp_ns, frame_id=index // 5 + 1) if model_updated else None,
      speed=10.0,
    )
    changing_target = step(
      changing,
      timestamp_ns,
      model=polynomial_model(0.015, -0.0007, timestamp_ns, frame_id=index // 5 + 1) if model_updated else None,
      speed=10.0,
    )
    assert changing_target == baseline_target


def test_latest_pending_segment_wins_before_promotion():
  controller = PersistentLateralPath()
  step(controller, NS, model=straight_model(NS), speed=10.0)
  step(
    controller,
    NS + 50_000_000,
    model=polynomial_model(0.01, 0.0002, NS + 50_000_000, frame_id=2),
    speed=10.0,
  )
  still_active = step(
    controller,
    NS + 90_000_000,
    model=polynomial_model(-0.008, -0.0003, NS + 90_000_000, frame_id=3),
    speed=10.0,
  )
  promoted = step(controller, NS + 100_000_000, speed=10.0)

  assert still_active.curvature == pytest.approx(0.0, abs=1e-7)
  assert promoted.curvature == pytest.approx(0.0, abs=2e-6)
  assert promoted.curvature_rate == pytest.approx(SPATIAL_CURVATURE_RATE_LIMITS[0], abs=1e-9)


def test_fresh_model_at_boundary_is_not_delayed_by_a_fallback_segment():
  controller = PersistentLateralPath()
  step(controller, NS, model=straight_model(NS), speed=10.0)

  promoted = step(
    controller,
    NS + 100_000_000,
    model=polynomial_model(0.01, 0.0004, NS + 100_000_000, frame_id=2),
    speed=10.0,
  )

  assert promoted.curvature == pytest.approx(0.0, abs=2e-6)
  assert promoted.curvature_rate == pytest.approx(SPATIAL_CURVATURE_RATE_LIMITS[1], abs=1e-9)


def test_fresh_model_just_after_boundary_replaces_unconsumed_fallback():
  controller = PersistentLateralPath()
  step(controller, NS, model=straight_model(NS), speed=10.0)
  crossed_without_model = step(controller, NS + 100_000_000, speed=10.0)
  recovered = step(
    controller,
    NS + 110_000_000,
    model=polynomial_model(0.01, 0.0004, NS + 110_000_000, frame_id=2),
    speed=10.0,
  )

  assert crossed_without_model.curvature_rate == pytest.approx(0.0, abs=1e-9)
  assert recovered.curvature == pytest.approx(0.0, abs=2e-6)
  assert recovered.curvature_rate == pytest.approx(SPATIAL_CURVATURE_RATE_LIMITS[1], abs=1e-9)


def test_repeated_replans_do_not_reverse_the_requested_curvature_ramp():
  controller = PersistentLateralPath()
  step(controller, NS, model=straight_model(NS), speed=10.0)
  targets = []

  for index in range(1, 81):
    timestamp_ns = NS + index * 10_000_000
    model_updated = index % 5 == 0
    target = step(
      controller,
      timestamp_ns,
      model=polynomial_model(0.0, 0.001, timestamp_ns, frame_id=index // 5 + 1) if model_updated else None,
      speed=10.0,
    )
    if index >= 10:
      targets.append(target)

  assert all(target.valid for target in targets)
  assert all(target.curvature >= -1e-6 for target in targets)
  assert all(target.curvature_rate >= -1e-6 for target in targets)
  assert any(target.curvature_rate > 0.0008 for target in targets)


@pytest.mark.parametrize("radius", [20.0, -20.0])
def test_parametric_path_survives_past_world_ninety_degrees(radius: float):
  controller = PersistentLateralPath()
  timestamp_ns = NS
  speed = 20.0
  yaw_rate = speed / radius
  target = step(controller, timestamp_ns, model=circle_model(radius, timestamp_ns),
                speed=speed, yaw_rate=yaw_rate)

  for index in range(1, 181):
    timestamp_ns = NS + index * 10_000_000
    model_updated = index % 5 == 0
    target = step(
      controller,
      timestamp_ns,
      model=circle_model(radius, timestamp_ns, frame_id=index // 5 + 1) if model_updated else None,
      speed=speed,
      yaw_rate=yaw_rate,
    )

  assert target.valid
  assert target.path_offset == pytest.approx(0.0, abs=0.025)
  assert target.path_angle == pytest.approx(0.0, abs=math.radians(0.25))
  assert target.curvature == pytest.approx(math.copysign(0.02, radius), abs=0.001)


def test_c2_is_reference_geometry_not_state_and_has_no_accumulator():
  controller = PersistentLateralPath()
  step(controller, NS, model=circle_model(20.0, NS))
  controller.reset()

  target = step(controller, NS + 50_000_000, model=straight_model(NS + 50_000_000, 2))

  assert target.valid
  assert target.curvature == pytest.approx(0.0, abs=1e-7)
  assert target.curvature_rate == pytest.approx(0.0, abs=1e-7)


def test_undertracking_relatches_without_controller_mode():
  controller = PersistentLateralPath()
  step(controller, NS, model=circle_model(25.0, NS), speed=8.0)
  first_error = None
  for index in range(1, 11):
    first_error = step(controller, NS + index * 10_000_000, speed=8.0, yaw_rate=0.0)
  caught_up = None
  for index in range(11, 21):
    caught_up = step(controller, NS + index * 10_000_000, speed=8.0, yaw_rate=2.0 * 8.0 / 25.0)
  second_error = None
  for index in range(21, 31):
    second_error = step(controller, NS + index * 10_000_000, speed=8.0, yaw_rate=0.0)

  assert first_error is not None and caught_up is not None and second_error is not None
  assert first_error.path_offset > 0.0 and first_error.path_angle > 0.0
  assert abs(caught_up.path_angle) < abs(first_error.path_angle)
  assert second_error.path_angle > caught_up.path_angle


def test_active_ego_motion_preserves_reference_and_observer_state():
  controller = PersistentLateralPath()
  step(controller, NS, model=circle_model(25.0, NS), speed=8.0)

  during_override = None
  for index in range(1, 21):
    # selfdrive State.overriding remains active; driver motion must propagate the
    # observer rather than clearing and re-rooting it.
    during_override = step(controller, NS + index * 10_000_000, speed=8.0, yaw_rate=2.0 * 8.0 / 25.0)
  resumed = step(
    controller, NS + 250_000_000,
    model=circle_model(25.0, NS + 250_000_000, frame_id=2),
    speed=8.0, yaw_rate=8.0 / 25.0,
  )

  assert during_override is not None and during_override.valid
  assert resumed.valid
  assert abs(resumed.path_offset) > 1e-3
  assert abs(resumed.path_angle) > math.radians(0.1)


def test_invalid_fresh_model_is_ignored_while_retained_reference_is_fresh():
  controller = PersistentLateralPath()
  original = step(controller, NS, model=circle_model(25.0, NS))
  bad = straight_model(NS + 50_000_000, 2)
  bad.position.y[5] = math.nan

  retained = step(controller, NS + 50_000_000, model=bad)

  assert original.valid and retained.valid
  assert retained.curvature == pytest.approx(original.curvature, rel=0.05)


def test_inactive_control_clears_reference():
  controller = PersistentLateralPath()
  assert step(controller, NS, model=circle_model(25.0, NS)).valid

  inactive = step(controller, NS + 10_000_000, active=False)
  assert not inactive.valid
  assert inactive.path_offset == inactive.path_angle == inactive.curvature_rate == 0.0


def test_nonfinite_model_fails_closed():
  controller = PersistentLateralPath()
  bad = straight_model(NS)
  bad.position.y[5] = math.nan

  target = step(controller, NS, model=bad)

  assert not target.valid


def test_positive_negative_path_symmetry():
  positive = PersistentLateralPath()
  negative = PersistentLateralPath()
  positive_target = step(positive, NS, model=polynomial_model(0.006, 0.0003, NS))
  negative_target = step(negative, NS, model=polynomial_model(-0.006, -0.0003, NS))

  assert negative_target.path_offset == pytest.approx(-positive_target.path_offset, abs=1e-8)
  assert negative_target.path_angle == pytest.approx(-positive_target.path_angle, abs=1e-8)
  assert negative_target.curvature == pytest.approx(-positive_target.curvature, rel=1e-5)
  assert negative_target.curvature_rate == pytest.approx(-positive_target.curvature_rate, rel=1e-5)


def test_frenet_offset_and_heading_use_model_right_positive_units():
  controller = PersistentLateralPath()
  distances = [index * 0.25 for index in range(161)]
  heading = math.radians(5.0)
  target = step(
    controller, NS,
    model=model_path(
      [distance * math.cos(heading) for distance in distances],
      [0.4 + distance * math.sin(heading) for distance in distances],
      [heading] * len(distances),
      NS,
    ),
  )

  assert target.valid
  assert target.path_offset == pytest.approx(0.4 * math.cos(heading), abs=1e-3)
  assert target.path_angle == pytest.approx(heading, abs=1e-4)


def test_wire_bounds_are_the_only_output_clamps():
  controller = PersistentLateralPath()
  target = step(controller, NS, model=circle_model(10.0, NS))

  assert target.valid
  assert target.curvature == 0.02
  assert -5.11 <= target.path_offset <= 5.12
  assert -0.5235 <= target.path_angle <= 0.5
  assert -0.001023 <= target.curvature_rate <= 0.001024
