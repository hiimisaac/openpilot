import os
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("SCALE", "1")

import numpy as np
import pyray as rl

from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.ui.mici.onroad import model_renderer as mici_model_renderer
from openpilot.selfdrive.ui.onroad import intent_overlay
from openpilot.selfdrive.ui.onroad import model_renderer as big_model_renderer
from openpilot.selfdrive.ui.onroad.intent_overlay import IntentOverlay, LaneChangePhase, StopKind, derive_intent_state
from openpilot.selfdrive.ui.ui_state import UIStatus


def message(**kwargs):
  return SimpleNamespace(**kwargs)


def empty_model():
  return message(
    action=message(shouldStop=False),
    position=message(t=[], x=[], y=[], z=[]),
    velocity=message(t=[], x=[]),
    orientation=message(t=[], z=[]),
    meta=message(laneChangeState='off', laneChangeDirection='none'),
    laneLines=[],
    laneLineProbs=[],
  )


def empty_car_state():
  return message(leftBlindspot=False, rightBlindspot=False)


def empty_plan():
  return message(
    shouldStop=False,
    longitudinalPlanSource='cruise',
    speeds=[],
    aTarget=0.0,
  )


class FakeSubMaster:
  def __init__(self, **messages):
    self.services = list(messages)
    self.data = messages
    self.alive = dict.fromkeys(self.services, True)
    self.valid = dict.fromkeys(self.services, True)

  def __getitem__(self, service):
    return self.data[service]


def renderer_submaster():
  sm = FakeSubMaster(
    liveCalibration=message(height=[1.25]),
    modelV2=empty_model(),
    radarState=message(),
    carParams=message(),
    carState=empty_car_state(),
    carControl=message(latActive=True, longActive=True),
    carOutput=message(actuatorsOutput=message(torque=0.0)),
    selfdriveState=message(experimentalMode=False),
  )
  sm.recv_frame = {'liveCalibration': 1, 'modelV2': 1}
  sm.updated = dict.fromkeys(sm.services, False)
  sm.valid['radarState'] = False
  return sm


def test_control_plan_stop_uses_published_speed_horizon():
  speeds = np.linspace(5.1, 0.3, 17)
  plan = message(
    shouldStop=False,
    longitudinalPlanSource='lead0',
    speeds=speeds,
    aTarget=-1.0,
  )

  state = derive_intent_state(empty_model(), plan, empty_car_state())

  assert state.stop is not None
  assert state.stop.kind == StopKind.CONTROL
  expected_distance = np.trapezoid(speeds, ModelConstants.T_IDXS[:len(speeds)])
  assert np.isclose(state.stop.distance, expected_distance)


def test_e2e_stop_uses_model_trajectory_not_mpc_speeds():
  model = empty_model()
  model.action.shouldStop = True
  model.position = message(t=[0.0, 1.0, 2.0, 3.0], x=[0.0, 5.0, 8.0, 9.0], y=[0.0] * 4, z=[0.0] * 4)
  model.velocity = message(t=[0.0, 1.0, 2.0, 3.0], x=[5.0, 3.0, 1.0, 0.2])
  plan = message(
    shouldStop=True,
    longitudinalPlanSource='e2e',
    speeds=np.linspace(20.0, 0.0, 17),
    aTarget=-1.0,
  )

  state = derive_intent_state(model, plan, empty_car_state())

  assert state.stop is not None
  assert state.stop.kind == StopKind.MODEL
  assert np.isclose(state.stop.distance, 8.875)


def test_e2e_stop_distance_follows_curved_model_path_length():
  model = empty_model()
  model.action.shouldStop = True
  model.position = message(t=[0.0, 1.0, 2.0], x=[0.0, 3.0, 6.0], y=[0.0, 4.0, 4.0], z=[0.0] * 3)
  model.velocity = message(t=[0.0, 1.0, 2.0], x=[5.0, 1.0, 0.2])
  plan = message(shouldStop=True, longitudinalPlanSource='e2e', speeds=[], aTarget=-1.0)

  state = derive_intent_state(model, plan, empty_car_state())

  assert state.stop is not None
  assert np.isclose(state.stop.distance, 7.625)


def test_future_poses_follow_model_time_and_unwrap_yaw():
  model = empty_model()
  model.position = message(
    t=[0.0, 0.5, 1.5, 2.5, 3.5],
    x=[0.0, 2.0, 7.0, 13.0, 20.0],
    y=[0.0, 0.1, 0.6, 1.5, 2.8],
    z=[0.0] * 5,
  )
  model.orientation = message(
    t=[0.0, 0.5, 1.5, 2.5, 3.5],
    z=[3.0, 3.1, -3.0, -2.9, -2.8],
  )

  state = derive_intent_state(model, empty_plan(), empty_car_state())

  assert [pose.t for pose in state.future_poses] == [1.0, 2.0, 3.0]
  assert np.isclose(state.future_poses[0].x, 4.5)
  assert np.isclose(state.future_poses[1].y, 1.05)
  assert state.future_poses[0].yaw > 3.0


def test_pre_lane_change_marks_matching_blindspot_blocked():
  model = empty_model()
  model.meta = message(laneChangeState='preLaneChange', laneChangeDirection='left')
  car_state = message(leftBlindspot=True, rightBlindspot=False)

  state = derive_intent_state(model, empty_plan(), car_state)

  assert state.lane_change is not None
  assert state.lane_change.phase == LaneChangePhase.CANDIDATE
  assert state.lane_change.boundary_indices == (0, 1)
  assert state.lane_change.blocked


def test_active_lane_change_does_not_relabel_new_blindspot_as_blocked():
  model = empty_model()
  model.meta = message(laneChangeState='laneChangeStarting', laneChangeDirection='right')
  car_state = message(leftBlindspot=False, rightBlindspot=True)

  state = derive_intent_state(model, empty_plan(), car_state)

  assert state.lane_change is not None
  assert state.lane_change.phase == LaneChangePhase.ACTIVE
  assert state.lane_change.boundary_indices == (2, 3)
  assert not state.lane_change.blocked


def test_controlling_lead_matches_longitudinal_source():
  for source, lead_index in (('lead0', 0), ('lead1', 1), ('cruise', None), ('e2e', None)):
    plan = empty_plan()
    plan.longitudinalPlanSource = source

    state = derive_intent_state(empty_model(), plan, empty_car_state())

    assert state.controlling_lead_index == lead_index


def test_projection_uses_absolute_big_coordinates_and_rect_local_compact_coordinates():
  rect = rl.Rectangle(100, 50, 300, 180)
  transform = np.eye(3)
  big = IntentOverlay(compact=False)
  compact = IntentOverlay(compact=True)
  big.set_transform(transform)
  compact.set_transform(transform)

  assert np.allclose(big._project(rect, (120.0, 80.0, 1.0)), (120.0, 80.0))
  assert np.allclose(compact._project(rect, (20.0, 30.0, 1.0)), (120.0, 80.0))


def test_path_sample_places_travel_distance_along_curved_model_path():
  model = empty_model()
  model.position = message(x=[0.0, 3.0, 6.0], y=[0.0, 4.0, 4.0], z=[0.0, 0.0, 0.0])

  sample = IntentOverlay._path_sample(model, 5.0)

  assert sample is not None
  assert np.allclose(sample[:3], (3.0, 4.0, 0.0))


def test_big_renderer_forwards_transform_and_current_intent_state():
  renderer = big_model_renderer.ModelRenderer()
  transform = np.eye(3)
  renderer.set_transform(transform)
  renderer._transform_dirty = False
  renderer._longitudinal_control = True
  sm = renderer_submaster()
  rect = rl.Rectangle(0, 0, 400, 220)

  assert not renderer._intent_overlay._compact
  assert np.array_equal(renderer._intent_overlay._car_space_transform, transform)
  with patch.object(big_model_renderer.ui_state, 'sm', sm), \
       patch.object(big_model_renderer.ui_state, 'started_frame', 0), \
       patch.object(big_model_renderer.ui_state, 'driving_intent_enabled', True), \
       patch.object(renderer, '_draw_lane_lines'), \
       patch.object(renderer, '_draw_path'), \
       patch.object(renderer._intent_overlay, 'render_intent') as render:
    renderer._render(rect)

  render.assert_called_once_with(rect, sm, enabled=True, longitudinal_control=True, path_offset_z=1.25)


def test_compact_renderer_forwards_transform_and_current_intent_state():
  renderer = mici_model_renderer.ModelRenderer()
  transform = np.eye(3)
  renderer.set_transform(transform)
  renderer._transform_dirty = False
  renderer._longitudinal_control = True
  sm = renderer_submaster()
  rect = rl.Rectangle(100, 50, 400, 220)

  assert renderer._intent_overlay._compact
  assert np.array_equal(renderer._intent_overlay._car_space_transform, transform)
  with patch.object(mici_model_renderer.ui_state, 'sm', sm), \
       patch.object(mici_model_renderer.ui_state, 'started_frame', 0), \
       patch.object(mici_model_renderer.ui_state, 'status', UIStatus.ENGAGED), \
       patch.object(mici_model_renderer.ui_state, 'driving_intent_enabled', True), \
       patch.object(renderer, '_draw_lane_lines'), \
       patch.object(renderer, '_draw_path'), \
       patch.object(renderer._intent_overlay, 'render_intent') as render:
    renderer._render(rect)

  render.assert_called_once_with(rect, sm, enabled=True, longitudinal_control=True, path_offset_z=1.25)


def test_overlay_draws_trajectory_lane_target_stop_and_controlling_lead():
  times = [0.0, 1.0, 2.0, 3.0]
  model = empty_model()
  model.action.shouldStop = False
  model.position = message(t=times, x=[1.0, 6.0, 12.0, 18.0], y=[0.0, 0.2, 0.7, 1.4], z=[1.0] * 4)
  model.velocity = message(t=times, x=[6.0, 5.0, 3.0, 1.0])
  model.orientation = message(t=times, z=[0.0, 0.03, 0.08, 0.14])
  model.meta = message(laneChangeState='preLaneChange', laneChangeDirection='left')
  lane_x = [2.0, 8.0, 16.0, 30.0]
  model.laneLines = [
    message(x=lane_x, y=[offset] * 4, z=[1.0] * 4)
    for offset in (5.2, 1.8, -1.8, -5.2)
  ]
  model.laneLineProbs = [0.9] * 4

  plan = message(
    shouldStop=True,
    longitudinalPlanSource='lead0',
    speeds=np.linspace(5.1, 0.3, 17),
    aTarget=-1.0,
  )
  car_state = message(leftBlindspot=False, rightBlindspot=False)
  car_control = message(latActive=True, longActive=True)
  radar_state = message(
    leadOne=message(present=True, dRel=12.0, yRel=0.2, vRel=-1.0),
    leadTwo=message(present=False),
  )

  sm = FakeSubMaster(modelV2=model, longitudinalPlan=plan, carState=car_state,
                     carControl=car_control, radarState=radar_state)

  overlay = IntentOverlay(compact=False)
  overlay.set_transform(np.array([
    [200.0, 20.0, 0.0],
    [100.0, 0.0, -45.0],
    [1.0, 0.0, 0.0],
  ]))
  with patch.object(intent_overlay, 'draw_polygon') as draw_polygon, \
       patch.object(intent_overlay.rl, 'draw_line_ex') as draw_line, \
       patch.object(intent_overlay.rl, 'draw_triangle_fan') as draw_chevron, \
       patch.object(intent_overlay.rl, 'get_time', return_value=0.0):
    state = overlay.render_intent(rl.Rectangle(0, 0, 400, 220), sm, enabled=True,
                                  longitudinal_control=True, path_offset_z=1.0)

  assert state is not None
  assert draw_polygon.called
  assert draw_line.call_count >= 10
  assert draw_chevron.call_count >= 2


def test_geometry_smoothing_interpolates_without_lagging_first_frame():
  overlay = IntentOverlay(compact=False)
  initial = np.zeros((3, 3), dtype=np.float64)
  target = np.ones((3, 3), dtype=np.float64)

  first = overlay._smooth_geometry(None, initial)
  second = overlay._smooth_geometry(first, target)
  third = overlay._smooth_geometry(second, target)

  assert np.array_equal(first, initial)
  assert np.all((second > initial) & (second < target))
  assert np.all((third > second) & (third < target))


def test_big_overlay_draws_smoothly_animated_path_chevrons():
  times = [0.0, 1.0, 2.0, 3.0, 4.0]
  model = empty_model()
  model.position = message(t=times, x=[1.0, 9.0, 18.0, 28.0, 40.0],
                           y=[0.0, 0.1, 0.5, 1.1, 1.9], z=[1.0] * 5)
  model.velocity = message(t=times, x=[10.0, 10.0, 9.0, 8.0, 7.0])
  model.orientation = message(t=times, z=[0.0, 0.02, 0.05, 0.08, 0.1])
  sm = FakeSubMaster(
    modelV2=model,
    longitudinalPlan=empty_plan(),
    carState=empty_car_state(),
    carControl=message(latActive=True, longActive=False),
  )
  overlay = IntentOverlay(compact=False)
  overlay.set_transform(np.array([
    [200.0, 20.0, 0.0],
    [100.0, 0.0, -45.0],
    [1.0, 0.0, 0.0],
  ]))
  overlay._alpha_filter.x = 1.0

  frames = []
  for now in (0.0, 0.25):
    with patch.object(intent_overlay, 'draw_polygon'), \
         patch.object(intent_overlay.rl, 'draw_line_ex'), \
         patch.object(intent_overlay.rl, 'draw_circle'), \
         patch.object(intent_overlay.rl, 'draw_triangle_fan') as draw_chevron, \
         patch.object(intent_overlay.rl, 'get_time', return_value=now):
      overlay.render_intent(rl.Rectangle(0, 0, 400, 220), sm, enabled=True,
                            longitudinal_control=False, path_offset_z=1.0)
    frames.append([call.args[0] for call in draw_chevron.call_args_list])

  assert len(frames[0]) >= 6
  assert frames[0] != frames[1]
