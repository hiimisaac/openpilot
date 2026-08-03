import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pyray as rl

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.shader_polygon import draw_polygon
from openpilot.system.ui.widgets import Widget


STOP_SPEED = 0.3
MIN_STOP_DISTANCE = 1.5
MAX_DRAW_DISTANCE = 100.0
LANE_PROB_THRESHOLD = 0.25
INTENT_CYAN = rl.Color(70, 220, 255, 255)
BLOCKED_RED = rl.Color(255, 75, 85, 255)


class StopKind(StrEnum):
  CONTROL = "control"
  MODEL = "model"


class LaneChangePhase(StrEnum):
  CANDIDATE = "candidate"
  ACTIVE = "active"
  FINISHING = "finishing"


@dataclass(frozen=True)
class StopIntent:
  distance: float
  kind: StopKind


@dataclass(frozen=True)
class FuturePose:
  t: float
  x: float
  y: float
  z: float
  yaw: float


@dataclass(frozen=True)
class LaneChangeIntent:
  direction: str
  phase: LaneChangePhase
  boundary_indices: tuple[int, int]
  blocked: bool


@dataclass(frozen=True)
class IntentState:
  stop: StopIntent | None = None
  future_poses: tuple[FuturePose, ...] = ()
  lane_change: LaneChangeIntent | None = None
  controlling_lead_index: int | None = None


def _enum_name(value) -> str:
  return str(value).rsplit('.', maxsplit=1)[-1]


def _control_stop_distance(plan) -> float | None:
  if not plan.shouldStop or _enum_name(plan.longitudinalPlanSource) == 'e2e':
    return None

  speeds = np.asarray(plan.speeds, dtype=np.float64)
  if not 2 <= len(speeds) <= len(ModelConstants.T_IDXS):
    return None

  times = np.asarray(ModelConstants.T_IDXS[:len(speeds)], dtype=np.float64)
  if not np.all(np.isfinite(speeds)) or np.any(speeds < 0.0) or speeds[0] <= STOP_SPEED:
    return None

  stop_indices = np.flatnonzero(speeds[1:] <= STOP_SPEED)
  if not len(stop_indices):
    return None

  stop_idx = int(stop_indices[0] + 1)
  v0, v1 = speeds[stop_idx - 1:stop_idx + 1]
  t0, t1 = times[stop_idx - 1:stop_idx + 1]
  ratio = np.clip((v0 - STOP_SPEED) / max(v0 - v1, 1e-6), 0.0, 1.0)
  stop_time = t0 + ratio * (t1 - t0)

  distance = float(np.trapezoid(speeds[:stop_idx], times[:stop_idx]))
  distance += float((v0 + STOP_SPEED) * 0.5 * (stop_time - t0))
  return distance if distance >= MIN_STOP_DISTANCE else None


def _model_stop_distance(model, plan) -> float | None:
  if _enum_name(plan.longitudinalPlanSource) != 'e2e' or not model.action.shouldStop:
    return None

  velocity_t = np.asarray(model.velocity.t, dtype=np.float64)
  velocity_x = np.asarray(model.velocity.x, dtype=np.float64)
  position_t = np.asarray(model.position.t, dtype=np.float64)
  position_x = np.asarray(model.position.x, dtype=np.float64)
  position_y = np.asarray(model.position.y, dtype=np.float64)
  position_z = np.asarray(model.position.z, dtype=np.float64)
  arrays = (velocity_t, velocity_x, position_t, position_x, position_y, position_z)
  if any(len(a) < 2 for a in arrays) or len(velocity_t) != len(velocity_x) or len(position_t) != len(position_x):
    return None
  if len({len(position_t), len(position_x), len(position_y), len(position_z)}) != 1:
    return None
  if not all(np.all(np.isfinite(a)) for a in arrays):
    return None
  if np.any(np.diff(velocity_t) <= 0.0) or np.any(np.diff(position_t) <= 0.0) or velocity_x[0] <= STOP_SPEED:
    return None

  stop_indices = np.flatnonzero(velocity_x[1:] <= STOP_SPEED)
  if not len(stop_indices):
    return None

  stop_idx = int(stop_indices[0] + 1)
  v0, v1 = velocity_x[stop_idx - 1:stop_idx + 1]
  t0, t1 = velocity_t[stop_idx - 1:stop_idx + 1]
  ratio = np.clip((v0 - STOP_SPEED) / max(v0 - v1, 1e-6), 0.0, 1.0)
  stop_time = t0 + ratio * (t1 - t0)
  if not position_t[0] <= stop_time <= position_t[-1]:
    return None

  position = np.column_stack((position_x, position_y, position_z))
  arc_distance = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(position, axis=0), axis=1))))
  distance = float(np.interp(stop_time, position_t, arc_distance))
  return distance if distance >= MIN_STOP_DISTANCE else None


def _future_poses(model) -> tuple[FuturePose, ...]:
  position_t = np.asarray(model.position.t, dtype=np.float64)
  position_x = np.asarray(model.position.x, dtype=np.float64)
  position_y = np.asarray(model.position.y, dtype=np.float64)
  position_z = np.asarray(model.position.z, dtype=np.float64)
  orientation_t = np.asarray(model.orientation.t, dtype=np.float64)
  orientation_z = np.asarray(model.orientation.z, dtype=np.float64)
  position_arrays = (position_t, position_x, position_y, position_z)
  orientation_arrays = (orientation_t, orientation_z)
  if any(len(a) < 2 for a in (*position_arrays, *orientation_arrays)):
    return ()
  if len({len(a) for a in position_arrays}) != 1 or len(orientation_t) != len(orientation_z):
    return ()
  if not all(np.all(np.isfinite(a)) for a in (*position_arrays, *orientation_arrays)):
    return ()
  if np.any(np.diff(position_t) <= 0.0) or np.any(np.diff(orientation_t) <= 0.0):
    return ()

  yaw = np.unwrap(orientation_z)
  poses = []
  for t in (1.0, 2.0, 3.0):
    if not (position_t[0] <= t <= position_t[-1] and orientation_t[0] <= t <= orientation_t[-1]):
      continue
    poses.append(FuturePose(
      t=t,
      x=float(np.interp(t, position_t, position_x)),
      y=float(np.interp(t, position_t, position_y)),
      z=float(np.interp(t, position_t, position_z)),
      yaw=float(np.interp(t, orientation_t, yaw)),
    ))
  return tuple(poses)


def _lane_change_intent(model, car_state) -> LaneChangeIntent | None:
  state = _enum_name(model.meta.laneChangeState)
  direction = _enum_name(model.meta.laneChangeDirection)
  if direction not in ('left', 'right'):
    return None

  phases = {
    'preLaneChange': LaneChangePhase.CANDIDATE,
    'laneChangeStarting': LaneChangePhase.ACTIVE,
    'laneChangeFinishing': LaneChangePhase.FINISHING,
  }
  phase = phases.get(state)
  if phase is None:
    return None

  boundary_indices = (0, 1) if direction == 'left' else (2, 3)
  blindspot = car_state.leftBlindspot if direction == 'left' else car_state.rightBlindspot
  return LaneChangeIntent(
    direction=direction,
    phase=phase,
    boundary_indices=boundary_indices,
    blocked=bool(phase == LaneChangePhase.CANDIDATE and blindspot),
  )


def derive_intent_state(model, plan, car_state) -> IntentState:
  stop_distance = _control_stop_distance(plan) if plan is not None else None
  stop_kind = StopKind.CONTROL
  if stop_distance is None and plan is not None:
    stop_distance = _model_stop_distance(model, plan)
    stop_kind = StopKind.MODEL
  stop = StopIntent(stop_distance, stop_kind) if stop_distance is not None else None
  controlling_lead_index = None if plan is None else {
    'lead0': 0,
    'lead1': 1,
  }.get(_enum_name(plan.longitudinalPlanSource))
  return IntentState(
    stop=stop,
    future_poses=_future_poses(model),
    lane_change=_lane_change_intent(model, car_state),
    controlling_lead_index=controlling_lead_index,
  )


class IntentOverlay(Widget):
  """Render calibrated driving intent from existing model and control messages."""

  def __init__(self, compact: bool):
    super().__init__()
    self.set_enabled(False)
    self._compact = compact
    self._car_space_transform = np.zeros((3, 3), dtype=np.float64)
    self._alpha_filter = FirstOrderFilter(0.0, 0.15, 1 / gui_app.target_fps)
    self._sm = None
    self._intent_enabled = False
    self._longitudinal_control = False
    self._path_offset_z = 0.0
    self._render_state: IntentState | None = None

  def set_transform(self, transform: np.ndarray) -> None:
    self._car_space_transform = np.asarray(transform, dtype=np.float64)

  @staticmethod
  def _message_valid(sm, service: str) -> bool:
    services = getattr(sm, 'services', sm)
    return bool(service in services and sm.alive.get(service, False) and sm.valid.get(service, False))

  @staticmethod
  def _color(color: rl.Color, alpha: float) -> rl.Color:
    return rl.Color(color.r, color.g, color.b, int(np.clip(alpha, 0.0, 255.0)))

  def _project(self, rect: rl.Rectangle, point: tuple[float, float, float]) -> tuple[float, float] | None:
    projected = self._car_space_transform @ np.asarray(point, dtype=np.float64)
    if not np.all(np.isfinite(projected)) or abs(projected[2]) < 1e-6:
      return None

    offset_x = rect.x if self._compact else 0.0
    offset_y = rect.y if self._compact else 0.0
    x = float(projected[0] / projected[2] + offset_x)
    y = float(projected[1] / projected[2] + offset_y)
    if not (rect.x <= x <= rect.x + rect.width and rect.y <= y <= rect.y + rect.height):
      return None
    return x, y

  @staticmethod
  def _path_sample(model, distance: float) -> tuple[float, float, float, float] | None:
    x = np.asarray(model.position.x, dtype=np.float64)
    y = np.asarray(model.position.y, dtype=np.float64)
    z = np.asarray(model.position.z, dtype=np.float64)
    if len(x) < 2 or len(x) != len(y) or len(x) != len(z):
      return None
    if not all(np.all(np.isfinite(a)) for a in (x, y, z)) or np.any(np.diff(x) <= 0.0):
      return None
    path = np.column_stack((x, y, z))
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc_distance = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if np.any(segment_lengths <= 1e-6) or not 0.0 <= distance <= arc_distance[-1]:
      return None

    idx = int(np.clip(np.searchsorted(arc_distance, distance), 1, len(x) - 1))
    yaw = math.atan2(y[idx] - y[idx - 1], x[idx] - x[idx - 1])
    return (
      float(np.interp(distance, arc_distance, x)),
      float(np.interp(distance, arc_distance, y)),
      float(np.interp(distance, arc_distance, z)),
      yaw,
    )

  @staticmethod
  def _path_height(model, forward_distance: float) -> float:
    x = np.asarray(model.position.x, dtype=np.float64)
    z = np.asarray(model.position.z, dtype=np.float64)
    if len(x) < 2 or len(x) != len(z) or not all(np.all(np.isfinite(a)) for a in (x, z)) or np.any(np.diff(x) <= 0.0):
      return 0.0
    return float(np.interp(forward_distance, x, z))

  def _draw_lane_target(self, rect: rl.Rectangle, model, lane_change: LaneChangeIntent, alpha: float) -> None:
    first_idx, second_idx = lane_change.boundary_indices
    if len(model.laneLines) <= second_idx or len(model.laneLineProbs) <= second_idx:
      return
    if min(model.laneLineProbs[first_idx], model.laneLineProbs[second_idx]) < LANE_PROB_THRESHOLD:
      return

    boundary_data = []
    for line_idx in lane_change.boundary_indices:
      line = model.laneLines[line_idx]
      x = np.asarray(line.x, dtype=np.float64)
      y = np.asarray(line.y, dtype=np.float64)
      z = np.asarray(line.z, dtype=np.float64)
      if len(x) < 2 or len(x) != len(y) or len(x) != len(z):
        return
      if not all(np.all(np.isfinite(a)) for a in (x, y, z)):
        return
      boundary_data.append((x, y, z))

    if len(boundary_data[0][0]) != len(boundary_data[1][0]):
      return

    first: list[tuple[float, float]] = []
    second: list[tuple[float, float]] = []
    for first_point, second_point in zip(zip(*boundary_data[0], strict=True), zip(*boundary_data[1], strict=True), strict=True):
      if not all(1.0 <= point[0] <= MAX_DRAW_DISTANCE for point in (first_point, second_point)):
        continue
      first_projected = self._project(rect, tuple(float(value) for value in first_point))
      second_projected = self._project(rect, tuple(float(value) for value in second_point))
      if first_projected is not None and second_projected is not None:
        first.append(first_projected)
        second.append(second_projected)

    if len(first) < 2:
      return

    point_count = len(first)
    polygon = np.asarray(first + list(reversed(second)), dtype=np.float32)
    color = BLOCKED_RED if lane_change.blocked else INTENT_CYAN
    phase_alpha = {
      LaneChangePhase.CANDIDATE: 28,
      LaneChangePhase.ACTIVE: 42,
      LaneChangePhase.FINISHING: 20,
    }[lane_change.phase]
    if lane_change.blocked:
      phase_alpha = 48
    draw_polygon(rect, polygon, self._color(color, phase_alpha * alpha))

    outer_boundary = first if lane_change.direction == 'left' else second
    line_width = 2.0 if self._compact else 5.0
    for start, end in zip(outer_boundary[:-1], outer_boundary[1:], strict=True):
      rl.draw_line_ex(rl.Vector2(*start), rl.Vector2(*end), line_width, self._color(color, 205 * alpha))

    if lane_change.blocked:
      marker_idx = min(max(point_count // 3, 0), point_count - 1)
      center = np.mean((first[marker_idx], second[marker_idx]), axis=0)
      size = 6.0 if self._compact else 16.0
      width = 2.0 if self._compact else 4.0
      rl.draw_line_ex(rl.Vector2(center[0] - size, center[1] - size),
                      rl.Vector2(center[0] + size, center[1] + size), width, self._color(color, 230 * alpha))
      rl.draw_line_ex(rl.Vector2(center[0] - size, center[1] + size),
                      rl.Vector2(center[0] + size, center[1] - size), width, self._color(color, 230 * alpha))

  def _draw_future_poses(self, rect: rl.Rectangle, poses: tuple[FuturePose, ...], path_offset_z: float, alpha: float) -> None:
    previous_center = None
    for i, pose in enumerate(poses):
      lateral_x = -math.sin(pose.yaw) * 0.65
      lateral_y = math.cos(pose.yaw) * 0.65
      z = pose.z + path_offset_z
      left = self._project(rect, (pose.x + lateral_x, pose.y + lateral_y, z))
      right = self._project(rect, (pose.x - lateral_x, pose.y - lateral_y, z))
      center = self._project(rect, (pose.x, pose.y, z))
      if left is None or right is None or center is None:
        continue
      if self._compact and previous_center is not None and np.linalg.norm(np.subtract(center, previous_center)) < 12.0:
        continue

      opacity = (220, 175, 135)[min(i, 2)] * alpha
      glow_width = 3.0 if self._compact else 10.0
      core_width = 1.5 if self._compact else 4.0
      rl.draw_line_ex(rl.Vector2(*left), rl.Vector2(*right), glow_width, self._color(INTENT_CYAN, opacity * 0.22))
      rl.draw_line_ex(rl.Vector2(*left), rl.Vector2(*right), core_width, self._color(INTENT_CYAN, opacity))
      radius = 2.0 if self._compact else 6.0
      rl.draw_circle(int(center[0]), int(center[1]), radius, self._color(INTENT_CYAN, opacity))
      previous_center = center

  def _draw_stop(self, rect: rl.Rectangle, model, stop: StopIntent, path_offset_z: float, alpha: float) -> None:
    sample = self._path_sample(model, stop.distance)
    if sample is None:
      return
    x, y, z, yaw = sample
    lateral_x = -math.sin(yaw) * 1.1
    lateral_y = math.cos(yaw) * 1.1
    forward_x = math.cos(yaw) * 0.3
    forward_y = math.sin(yaw) * 0.3
    color = INTENT_CYAN
    glow_width = 5.0 if self._compact else 18.0
    core_width = 2.0 if self._compact else 7.0
    offsets = (0.0, 0.3) if stop.kind == StopKind.CONTROL else (0.0,)
    core_alpha = 225 if stop.kind == StopKind.CONTROL else 155
    for longitudinal_offset in offsets:
      offset_x = forward_x * (longitudinal_offset / 0.3)
      offset_y = forward_y * (longitudinal_offset / 0.3)
      left = self._project(rect, (x + offset_x + lateral_x, y + offset_y + lateral_y, z + path_offset_z))
      right = self._project(rect, (x + offset_x - lateral_x, y + offset_y - lateral_y, z + path_offset_z))
      if left is None or right is None:
        continue
      rl.draw_line_ex(rl.Vector2(*left), rl.Vector2(*right), glow_width, self._color(color, 45 * alpha))
      rl.draw_line_ex(rl.Vector2(*left), rl.Vector2(*right), core_width, self._color(color, core_alpha * alpha))

  def _draw_controlling_lead(self, rect: rl.Rectangle, model, radar_state, lead_index: int,
                             path_offset_z: float, alpha: float) -> None:
    leads = (radar_state.leadOne, radar_state.leadTwo)
    if lead_index >= len(leads) or not leads[lead_index].present:
      return
    lead = leads[lead_index]
    z = self._path_height(model, float(lead.dRel))
    point = self._project(rect, (float(lead.dRel), float(-lead.yRel), z + path_offset_z))
    if point is None:
      return

    scale = 1.0 if self._compact else 2.35
    size = float(np.clip((25.0 * 30.0) / (lead.dRel / 3.0 + 30.0), 15.0, 30.0) * scale)
    x, y = point
    outline = (
      (x + size * 1.45, y + size * 1.08),
      (x, y - size * 0.12),
      (x - size * 1.45, y + size * 1.08),
    )
    width = 2.0 if self._compact else 6.0
    for start, end in zip(outline, (*outline[1:], outline[0]), strict=True):
      rl.draw_line_ex(rl.Vector2(*start), rl.Vector2(*end), width, self._color(INTENT_CYAN, 225 * alpha))

    if self._compact:
      chevron = [(x + size * 1.15, y + size), (x, y), (x - size * 1.15, y + size)]
      fill_alpha = int(np.clip(255.0 * (1.0 - lead.dRel / 40.0) + max(-lead.vRel, 0.0) * 25.5, 0.0, 255.0))
      rl.draw_triangle_fan(chevron, len(chevron), rl.Color(201, 34, 49, int(fill_alpha * alpha)))

  def render_intent(self, rect: rl.Rectangle, sm, *, enabled: bool, longitudinal_control: bool,
                    path_offset_z: float) -> IntentState | None:
    self._sm = sm
    self._intent_enabled = enabled
    self._longitudinal_control = longitudinal_control
    self._path_offset_z = path_offset_z
    self._render_state = None
    super().render(rect)
    return self._render_state

  def _render(self, rect: rl.Rectangle) -> None:
    sm = self._sm
    required = ('modelV2', 'carState', 'carControl')
    if not all(self._message_valid(sm, service) for service in required):
      self._alpha_filter.x = 0.0
      return

    car_control = sm['carControl']
    active = self._intent_enabled and (car_control.latActive or (self._longitudinal_control and car_control.longActive))
    alpha = self._alpha_filter.update(float(active))
    if alpha < 1e-2:
      return

    model = sm['modelV2']
    plan_valid = self._message_valid(sm, 'longitudinalPlan')
    plan = sm['longitudinalPlan'] if plan_valid else None
    state = derive_intent_state(model, plan, sm['carState'])

    if car_control.latActive and state.lane_change is not None:
      self._draw_lane_target(rect, model, state.lane_change, alpha)
    if car_control.latActive:
      self._draw_future_poses(rect, state.future_poses, self._path_offset_z, alpha)
    if self._longitudinal_control and car_control.longActive and state.stop is not None:
      self._draw_stop(rect, model, state.stop, self._path_offset_z, alpha)
    if (self._longitudinal_control and car_control.longActive and state.controlling_lead_index is not None and
        self._message_valid(sm, 'radarState')):
      self._draw_controlling_lead(rect, model, sm['radarState'], state.controlling_lead_index, self._path_offset_z, alpha)
    self._render_state = state
