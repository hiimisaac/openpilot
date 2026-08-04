import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pyray as rl

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.shader_polygon import Gradient, draw_polygon
from openpilot.system.ui.widgets import Widget


STOP_SPEED = 0.3
MIN_STOP_DISTANCE = 1.5
MAX_DRAW_DISTANCE = 100.0
LANE_PROB_THRESHOLD = 0.25
TESLA_BLUE = rl.Color(52, 132, 255, 255)
PATH_HIGHLIGHT = rl.Color(116, 190, 255, 255)
ROAD_MARKING = rl.Color(224, 232, 238, 255)
ROAD_EDGE = rl.Color(142, 153, 162, 255)
SCENE_HORIZON = rl.Color(47, 54, 62, 255)
SCENE_FOREGROUND = rl.Color(12, 16, 21, 255)
ROAD_SURFACE = rl.Color(55, 61, 68, 255)
UI_BLACK = rl.Color(0, 0, 0, 255)
UI_WHITE = rl.Color(255, 255, 255, 255)
BLOCKED_RED = rl.Color(255, 75, 85, 255)
GEOMETRY_SMOOTHING_RC = 0.10
SCENE_DEPTH_METERS = 30.0
SCENE_NEAR_LATERAL_SCALE = 0.11
SCENE_FAR_LATERAL_SCALE = 0.012
VEHICLE_SPRITE_VISIBLE_BOTTOM = 641.0 / 768.0
VEHICLE_SPRITE_VISIBLE_WIDTH = (606.0 - 161.0) / 768.0
LEAD_VEHICLE_WIDTH_METERS = 1.55


def draw_solid_ribbon(points: list[tuple[float, float]], color: rl.Color) -> None:
  if rl.is_window_ready():
    if len(points) >= 4:
      p0, p1, p2 = points[:3]
      winding = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
      if winding > 0.0 and len(points) % 2 == 0:
        points = [point for i in range(0, len(points), 2) for point in (points[i + 1], points[i])]
    vertices = [rl.Vector2(*point) for point in points]
    rl.draw_triangle_strip(vertices, len(vertices), color)


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
class LaneChangeIntent:
  direction: str
  phase: LaneChangePhase
  boundary_indices: tuple[int, int]
  blocked: bool


@dataclass(frozen=True)
class IntentState:
  stop: StopIntent | None = None
  lane_change: LaneChangeIntent | None = None
  controlling_lead_index: int | None = None


def _enum_name(value) -> str:
  return str(value).rsplit('.', maxsplit=1)[-1]


def _control_stop_distance(plan) -> float | None:
  if _enum_name(plan.longitudinalPlanSource) == 'e2e':
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
    lane_change=_lane_change_intent(model, car_state),
    controlling_lead_index=controlling_lead_index,
  )


class IntentOverlay(Widget):
  """Render a virtual chase-view scene from existing model and control messages."""

  def __init__(self, compact: bool):
    super().__init__()
    self.set_enabled(False)
    self._compact = compact
    self._alpha_filter = FirstOrderFilter(0.0, 0.15, 1 / gui_app.target_fps)
    self._geometry_alpha = (1 / gui_app.target_fps) / (GEOMETRY_SMOOTHING_RC + 1 / gui_app.target_fps)
    self._smoothed_path: np.ndarray | None = None
    self._smoothed_lanes: np.ndarray | None = None
    self._smoothed_edges: np.ndarray | None = None
    self._smoothed_leads: list[np.ndarray | None] = [None, None]
    self._sm = None
    self._intent_enabled = False
    self._longitudinal_control = False
    self._path_offset_z = 0.0
    self._render_state: IntentState | None = None
    self._ego_texture: rl.Texture | None = None

  def set_transform(self, _transform: np.ndarray) -> None:
    # Kept for the ModelRenderer interface. The dedicated world scene deliberately
    # does not inherit the physical road-camera calibration.
    pass

  @staticmethod
  def _message_valid(sm, service: str) -> bool:
    services = getattr(sm, 'services', sm)
    return bool(service in services and sm.alive.get(service, False) and sm.valid.get(service, False))

  @staticmethod
  def _color(color: rl.Color, alpha: float) -> rl.Color:
    return rl.Color(color.r, color.g, color.b, int(np.clip(alpha, 0.0, 255.0)))

  @staticmethod
  def _lateral_scale(rect: rl.Rectangle, forward: float) -> float:
    progress = 1.0 - math.exp(-forward / SCENE_DEPTH_METERS)
    return rect.width * (
      SCENE_FAR_LATERAL_SCALE +
      (SCENE_NEAR_LATERAL_SCALE - SCENE_FAR_LATERAL_SCALE) * (1.0 - progress)
    )

  def _project(self, rect: rl.Rectangle, point: tuple[float, float, float]) -> tuple[float, float] | None:
    forward, lateral, _height = point
    if not np.all(np.isfinite(point)) or not 0.0 <= forward <= MAX_DRAW_DISTANCE:
      return None

    progress = 1.0 - math.exp(-forward / SCENE_DEPTH_METERS)
    near_y = rect.y + rect.height * 0.98
    horizon_y = rect.y + rect.height * (0.18 if self._compact else 0.15)
    y = near_y + (horizon_y - near_y) * progress

    lateral_scale = self._lateral_scale(rect, forward)
    # openpilot vehicle coordinates use positive lateral to the left.
    x = rect.x + rect.width * 0.5 - lateral * lateral_scale
    return float(x), float(y)

  @staticmethod
  def _model_path(model) -> np.ndarray | None:
    path = np.asarray([model.position.x, model.position.y, model.position.z], dtype=np.float64).T
    if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != 3:
      return None
    if not np.all(np.isfinite(path)) or np.any(np.diff(path[:, 0]) <= 0.0):
      return None
    return path

  @staticmethod
  def _model_lanes(model) -> np.ndarray | None:
    if len(model.laneLines) != 4:
      return None
    lanes = []
    for line in model.laneLines:
      lane = np.asarray([line.x, line.y, line.z], dtype=np.float64).T
      if lane.ndim != 2 or lane.shape[0] < 2 or lane.shape[1] != 3 or not np.all(np.isfinite(lane)):
        return None
      lanes.append(lane)
    if len({lane.shape for lane in lanes}) != 1:
      return None
    return np.asarray(lanes)

  @staticmethod
  def _model_edges(model) -> np.ndarray | None:
    if len(model.roadEdges) != 2:
      return None
    edges = []
    for line in model.roadEdges:
      edge = np.asarray([line.x, line.y, line.z], dtype=np.float64).T
      if edge.ndim != 2 or edge.shape[0] < 2 or edge.shape[1] != 3 or not np.all(np.isfinite(edge)):
        return None
      edges.append(edge)
    if edges[0].shape != edges[1].shape:
      return None
    return np.asarray(edges)

  def _smooth_geometry(self, current: np.ndarray | None, target: np.ndarray | None) -> np.ndarray | None:
    if target is None:
      return None
    if current is None or current.shape != target.shape:
      return target.copy()
    return current + self._geometry_alpha * (target - current)

  def _update_geometry(self, model) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    self._smoothed_path = self._smooth_geometry(self._smoothed_path, self._model_path(model))
    self._smoothed_lanes = self._smooth_geometry(self._smoothed_lanes, self._model_lanes(model))
    self._smoothed_edges = self._smooth_geometry(self._smoothed_edges, self._model_edges(model))
    return self._smoothed_path, self._smoothed_lanes, self._smoothed_edges

  @staticmethod
  def _sample_path(path: np.ndarray, distance: float) -> tuple[float, float, float, float] | None:
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc_distance = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if np.any(segment_lengths <= 1e-6) or not 0.0 <= distance <= arc_distance[-1]:
      return None

    idx = int(np.clip(np.searchsorted(arc_distance, distance), 1, len(path) - 1))
    yaw = math.atan2(path[idx, 1] - path[idx - 1, 1], path[idx, 0] - path[idx - 1, 0])
    return (
      float(np.interp(distance, arc_distance, path[:, 0])),
      float(np.interp(distance, arc_distance, path[:, 1])),
      float(np.interp(distance, arc_distance, path[:, 2])),
      yaw,
    )

  @classmethod
  def _path_sample(cls, model, distance: float) -> tuple[float, float, float, float] | None:
    path = cls._model_path(model)
    return None if path is None else cls._sample_path(path, distance)

  @staticmethod
  def _path_height(model, forward_distance: float) -> float:
    x = np.asarray(model.position.x, dtype=np.float64)
    z = np.asarray(model.position.z, dtype=np.float64)
    if len(x) < 2 or len(x) != len(z) or not all(np.all(np.isfinite(a)) for a in (x, z)) or np.any(np.diff(x) <= 0.0):
      return 0.0
    return float(np.interp(forward_distance, x, z))

  def _project_ribbon(self, rect: rl.Rectangle, path: np.ndarray, half_width: float,
                      path_offset_z: float) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    draw_path = path[(path[:, 0] >= 1.0) & (path[:, 0] <= MAX_DRAW_DISTANCE)]
    if len(draw_path) < 2:
      return [], []

    yaw = np.arctan2(np.gradient(draw_path[:, 1]), np.gradient(draw_path[:, 0]))
    normals = np.column_stack((-np.sin(yaw), np.cos(yaw))) * half_width
    left_car = draw_path.copy()
    right_car = draw_path.copy()
    left_car[:, :2] += normals
    right_car[:, :2] -= normals
    left_car[:, 2] += path_offset_z
    right_car[:, 2] += path_offset_z

    left: list[tuple[float, float]] = []
    right: list[tuple[float, float]] = []
    for left_point, right_point in zip(left_car, right_car, strict=True):
      left_projected = self._project(rect, tuple(left_point))
      right_projected = self._project(rect, tuple(right_point))
      if left_projected is not None and right_projected is not None:
        left.append(left_projected)
        right.append(right_projected)
    return left, right

  def _project_line(self, rect: rl.Rectangle, line: np.ndarray,
                    path_offset_z: float = 0.0) -> list[tuple[float, float]]:
    projected: list[tuple[float, float]] = []
    visible = line[(line[:, 0] >= 1.0) & (line[:, 0] <= MAX_DRAW_DISTANCE)]
    if len(visible) > 3:
      visible = np.concatenate((visible[::2], visible[-1:])) if len(visible) % 2 == 0 else visible[::2]
    for point in visible:
      screen_point = self._project(rect, (float(point[0]), float(point[1]), float(point[2] + path_offset_z)))
      if screen_point is not None:
        projected.append(screen_point)
    return projected

  def _draw_road_context(self, rect: rl.Rectangle, lanes: np.ndarray | None, lane_probs,
                         edges: np.ndarray | None, edge_stds, alpha: float) -> None:
    if edges is not None:
      for index, edge in enumerate(edges):
        if index >= len(edge_stds):
          continue
        confidence = float(np.clip(1.0 - edge_stds[index], 0.0, 1.0))
        if confidence < 0.2:
          continue
        points = self._project_line(rect, edge)
        width = 2.8 if self._compact else 7.0
        for start, end in zip(points[:-1], points[1:], strict=True):
          rl.draw_line_ex(rl.Vector2(*start), rl.Vector2(*end), width,
                          self._color(ROAD_EDGE, 110 * confidence * alpha))

    if lanes is not None:
      for index in range(len(lanes)):
        if index >= len(lanes) or index >= len(lane_probs):
          continue
        confidence = float(np.clip(lane_probs[index], 0.0, 1.0))
        if confidence < 0.45:
          continue
        points = self._project_line(rect, lanes[index])
        primary = index in (1, 2)
        width = (4.0 if primary else 3.0) if self._compact else (14.0 if primary else 9.0)
        line_alpha = 178 if primary else 112
        for start, end in zip(points[:-1], points[1:], strict=True):
          rl.draw_line_ex(rl.Vector2(*start), rl.Vector2(*end), width,
                          self._color(ROAD_MARKING, line_alpha * confidence * alpha))

  def _draw_blindspot_guard(self, rect: rl.Rectangle, car_state, alpha: float) -> None:
    """Blind-spot messages have no object position, so only mark the affected side."""
    for blocked, side in ((car_state.leftBlindspot, -1), (car_state.rightBlindspot, 1)):
      if not blocked:
        continue
      x = rect.x + (rect.width * (0.035 if side < 0 else 0.965))
      y0 = rect.y + rect.height * 0.56
      y1 = rect.y + rect.height * 0.82
      glow = 8.0 if self._compact else 20.0
      core = 2.0 if self._compact else 5.0
      rl.draw_line_ex(rl.Vector2(x, y0), rl.Vector2(x, y1), glow, self._color(BLOCKED_RED, 34 * alpha))
      rl.draw_line_ex(rl.Vector2(x, y0), rl.Vector2(x, y1), core, self._color(BLOCKED_RED, 220 * alpha))

  def _draw_scene_background(self, rect: rl.Rectangle, path: np.ndarray | None,
                             path_offset_z: float, alpha: float) -> None:
    """Replace the camera with a restrained pseudo-3D road scene."""
    rl.draw_rectangle_gradient_v(
      int(rect.x), int(rect.y), int(rect.width), int(rect.height),
      self._color(SCENE_HORIZON, 255 * alpha),
      self._color(SCENE_FOREGROUND, 255 * alpha),
    )

    if path is not None:
      shoulder_left, shoulder_right = self._project_ribbon(rect, path, 7.2, path_offset_z)
      if len(shoulder_left) >= 2:
        shoulder = np.asarray(shoulder_left + list(reversed(shoulder_right)), dtype=np.float32)
        draw_polygon(rect, shoulder, self._color(rl.Color(36, 41, 47, 255), 246 * alpha))
      left, right = self._project_ribbon(rect, path, 5.4, path_offset_z)
      if len(left) >= 2:
        road = np.asarray(left + list(reversed(right)), dtype=np.float32)
        draw_polygon(rect, road, self._color(ROAD_SURFACE, 244 * alpha))
        return

    center_x = rect.x + rect.width * 0.5
    horizon_y = rect.y + rect.height * (0.13 if self._compact else 0.10)
    road = (
      (center_x - rect.width * 0.105, horizon_y),
      (center_x + rect.width * 0.105, horizon_y),
      (rect.x + rect.width * 1.06, rect.y + rect.height),
      (rect.x - rect.width * 0.06, rect.y + rect.height),
    )
    rl.draw_triangle_fan(road, len(road), self._color(ROAD_SURFACE, 238 * alpha))

  def _ensure_vehicle_texture(self) -> bool:
    if self._ego_texture is None and rl.is_window_ready():
      self._ego_texture = gui_app.texture("images/intent_ego_vehicle.png", alpha_premultiply=True)
    return self._ego_texture is not None

  def _draw_vehicle_sprite(self, center_x: float, center_y: float, size: float,
                           rotation: float, color: rl.Color) -> bool:
    if not self._ensure_vehicle_texture():
      return False
    assert self._ego_texture is not None
    source = rl.Rectangle(0, 0, self._ego_texture.width, self._ego_texture.height)
    dest = rl.Rectangle(center_x, center_y, size, size)
    origin = rl.Vector2(size * 0.5, size * 0.5)
    rl.draw_texture_pro(self._ego_texture, source, dest, origin, rotation, color)
    return True

  @staticmethod
  def _rotated_offset(center_x: float, center_y: float, offset_x: float,
                      offset_y: float, rotation: float) -> tuple[float, float]:
    angle = math.radians(rotation)
    return (
      center_x + offset_x * math.cos(angle) - offset_y * math.sin(angle),
      center_y + offset_x * math.sin(angle) + offset_y * math.cos(angle),
    )

  def _draw_vehicle_lights(self, center_x: float, center_y: float, size: float, rotation: float,
                           braking: bool, alpha: float) -> None:
    left = self._rotated_offset(center_x, center_y, -size * 0.21, size * 0.18, rotation)
    right = self._rotated_offset(center_x, center_y, size * 0.21, size * 0.18, rotation)
    if braking:
      for point in (left, right):
        rl.draw_circle(int(point[0]), int(point[1]), size * 0.055, self._color(BLOCKED_RED, 48 * alpha))
        rl.draw_circle(int(point[0]), int(point[1]), size * 0.025, self._color(BLOCKED_RED, 235 * alpha))

  def _draw_ego_vehicle(self, rect: rl.Rectangle, braking: bool, alpha: float) -> None:
    if not self._ensure_vehicle_texture():
      return

    size = min(
      rect.width * (0.34 if self._compact else 0.27),
      rect.height * (0.68 if self._compact else 0.48),
    )
    center_x = rect.x + rect.width * 0.5
    bottom = rect.y + rect.height
    center_y = bottom - size * 0.42
    rotation = 0.0

    rl.draw_ellipse(
      int(center_x), int(bottom - size * 0.075),
      size * 0.31, size * 0.075,
      self._color(UI_BLACK, 112 * alpha),
    )
    self._draw_vehicle_sprite(center_x, center_y, size, rotation, self._color(UI_WHITE, 255 * alpha))
    self._draw_vehicle_lights(center_x, center_y, size, rotation, braking, alpha)

  def _draw_trajectory(self, rect: rl.Rectangle, path: np.ndarray, path_offset_z: float, alpha: float) -> None:
    # A solid vehicle-width ribbon reads as one committed path, instead of a
    # second pair of lane boundaries.
    half_width = 0.82 if self._compact else 1.0
    left, right = self._project_ribbon(rect, path, half_width, path_offset_z)
    if len(left) < 2:
      return

    strip = [point for pair in zip(right, left, strict=True) for point in pair]
    draw_solid_ribbon(strip, self._color(TESLA_BLUE, 225 * alpha))

    rail_color = self._color(PATH_HIGHLIGHT, 232 * alpha)
    rail_width = 1.5 if self._compact else 4.0
    for boundary in (left, right):
      for start, end in zip(boundary[:-1], boundary[1:], strict=True):
        rl.draw_line_ex(rl.Vector2(*start), rl.Vector2(*end), rail_width, rail_color)

  @staticmethod
  def _lane_change_branch(path: np.ndarray, lanes: np.ndarray,
                          lane_change: LaneChangeIntent) -> np.ndarray | None:
    first_idx, second_idx = lane_change.boundary_indices
    target = (lanes[first_idx] + lanes[second_idx]) * 0.5
    if len(target) < 2 or np.any(np.diff(target[:, 0]) <= 0.0):
      return None

    branch = path.copy()
    target_y = np.interp(path[:, 0], target[:, 0], target[:, 1])
    target_z = np.interp(path[:, 0], target[:, 0], target[:, 2])
    split_start, merge_distance = {
      LaneChangePhase.CANDIDATE: (7.0, 55.0),
      LaneChangePhase.ACTIVE: (3.5, 42.0),
      LaneChangePhase.FINISHING: (1.0, 28.0),
    }[lane_change.phase]
    blend = np.clip((path[:, 0] - split_start) / (merge_distance - split_start), 0.0, 1.0)
    blend = blend * blend * blend * (blend * (blend * 6.0 - 15.0) + 10.0)
    branch[:, 1] += blend * (target_y - path[:, 1])
    branch[:, 2] += blend * (target_z - path[:, 2])
    return branch

  def _draw_lane_change_indicator(self, rect: rl.Rectangle, path: np.ndarray,
                                  branch: np.ndarray, path_offset_z: float,
                                  alpha: float) -> None:
    marker_distance = 16.0 + (rl.get_time() * 7.0) % 20.0
    marker = self._sample_path(branch, marker_distance)
    origin = self._sample_path(path, marker_distance)
    if marker is None or origin is None:
      return

    marker_point = self._project(rect, (marker[0], marker[1], marker[2] + path_offset_z))
    origin_point = self._project(rect, (origin[0], origin[1], origin[2] + path_offset_z))
    if marker_point is None or origin_point is None:
      return

    direction = 1.0 if marker_point[0] >= origin_point[0] else -1.0
    size = 8.0 if self._compact else 22.0
    width = 2.5 if self._compact else 7.0
    tip = rl.Vector2(marker_point[0] + direction * size, marker_point[1])
    upper = rl.Vector2(marker_point[0] - direction * size * 0.55, marker_point[1] - size * 0.7)
    lower = rl.Vector2(marker_point[0] - direction * size * 0.55, marker_point[1] + size * 0.7)
    rl.draw_circle(int(marker_point[0]), int(marker_point[1]), size * 1.15,
                   self._color(TESLA_BLUE, 35 * alpha))
    color = self._color(PATH_HIGHLIGHT, 245 * alpha)
    rl.draw_line_ex(upper, tip, width, color)
    rl.draw_line_ex(lower, tip, width, color)

  def _draw_lane_target(self, rect: rl.Rectangle, path: np.ndarray, lanes: np.ndarray, lane_probs,
                        lane_change: LaneChangeIntent, path_offset_z: float, alpha: float) -> None:
    first_idx, second_idx = lane_change.boundary_indices
    if len(lanes) <= second_idx or len(lane_probs) <= second_idx:
      return
    if min(lane_probs[first_idx], lane_probs[second_idx]) < LANE_PROB_THRESHOLD:
      return

    boundary_data = []
    for line_idx in lane_change.boundary_indices:
      x, y, z = lanes[line_idx].T
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
    color = BLOCKED_RED if lane_change.blocked else TESLA_BLUE
    phase_alpha = {
      LaneChangePhase.CANDIDATE: 7,
      LaneChangePhase.ACTIVE: 13,
      LaneChangePhase.FINISHING: 6,
    }[lane_change.phase]
    if lane_change.blocked:
      phase_alpha = 48
    gradient = Gradient(
      start=(0.0, 1.0),
      end=(0.0, 0.0),
      colors=[self._color(color, phase_alpha * alpha), self._color(color, phase_alpha * 0.55 * alpha), self._color(color, 0)],
      stops=[0.0, 0.62, 1.0],
    )
    draw_polygon(rect, polygon, gradient=gradient)

    branch = self._lane_change_branch(path, lanes, lane_change)
    if branch is not None:
      branch_left, branch_right = self._project_ribbon(
        rect, branch, 0.40 if self._compact else 0.52, path_offset_z,
      )
      if len(branch_left) >= 2:
        branch_strip = [point for pair in zip(branch_right, branch_left, strict=True) for point in pair]
        branch_alpha = {
          LaneChangePhase.CANDIDATE: 105,
          LaneChangePhase.ACTIVE: 178,
          LaneChangePhase.FINISHING: 138,
        }[lane_change.phase]
        if lane_change.blocked:
          branch_alpha = 82
        pulse = 0.92 + 0.08 * math.sin(rl.get_time() * math.tau * 0.7)
        draw_solid_ribbon(branch_strip, self._color(color, branch_alpha * alpha * pulse))
        if not lane_change.blocked:
          self._draw_lane_change_indicator(rect, path, branch, path_offset_z, alpha)

    outer_boundary = first if lane_change.direction == 'left' else second
    glow_width = 4.0 if self._compact else 12.0
    line_width = 1.5 if self._compact else 4.0
    for start, end in zip(outer_boundary[:-1], outer_boundary[1:], strict=True):
      rl.draw_line_ex(rl.Vector2(*start), rl.Vector2(*end), glow_width, self._color(color, 28 * alpha))
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

  def _draw_stop(self, rect: rl.Rectangle, path: np.ndarray, stop: StopIntent,
                 path_offset_z: float, alpha: float) -> None:
    sample = self._sample_path(path, stop.distance)
    if sample is None:
      return
    x, y, z, yaw = sample
    lateral_x = -math.sin(yaw) * 1.1
    lateral_y = math.cos(yaw) * 1.1
    forward_x = math.cos(yaw) * 0.3
    forward_y = math.sin(yaw) * 0.3
    color = TESLA_BLUE
    pulse = 0.88 + 0.12 * math.sin(rl.get_time() * math.tau * 0.7)
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
      rl.draw_line_ex(rl.Vector2(*left), rl.Vector2(*right), glow_width, self._color(color, 45 * alpha * pulse))
      rl.draw_line_ex(rl.Vector2(*left), rl.Vector2(*right), core_width, self._color(color, core_alpha * alpha * pulse))

  def _draw_lead(self, rect: rl.Rectangle, model, radar_state, lead_index: int,
                 path_offset_z: float, alpha: float, controlling: bool) -> None:
    leads = (radar_state.leadOne, radar_state.leadTwo)
    if lead_index >= len(leads) or not leads[lead_index].present:
      if lead_index < len(self._smoothed_leads):
        self._smoothed_leads[lead_index] = None
      return
    lead = leads[lead_index]
    z = self._path_height(model, float(lead.dRel))
    # RadarState.yRel uses the opposite lateral sign from model coordinates.
    point = self._project(rect, (float(lead.dRel), -float(lead.yRel), z + path_offset_z))
    if point is None:
      return

    target = np.asarray(point)
    smoothed = self._smoothed_leads[lead_index]
    if smoothed is None or np.linalg.norm(target - smoothed) > rect.width * 0.35:
      smoothed = target.copy()
    else:
      smoothed += self._geometry_alpha * (target - smoothed)
    self._smoothed_leads[lead_index] = smoothed

    # Match the lead's visible width to the same perspective scale used by the
    # road and lane geometry. Account for transparent padding in the sprite.
    perspective_width = LEAD_VEHICLE_WIDTH_METERS * self._lateral_scale(rect, max(float(lead.dRel), 0.0))
    ego_size = min(
      rect.width * (0.34 if self._compact else 0.27),
      rect.height * (0.68 if self._compact else 0.48),
    )
    sprite_size = float(np.clip(perspective_width / VEHICLE_SPRITE_VISIBLE_WIDTH,
                                7.0 if self._compact else 18.0, ego_size * 0.92))
    size = sprite_size / 1.55
    x, ground_y = smoothed
    visible_bottom_offset = (VEHICLE_SPRITE_VISIBLE_BOTTOM - 0.5) * sprite_size
    sprite_center_y = ground_y - visible_bottom_offset
    top, middle, bottom = ground_y - size * 1.18, ground_y - size * 0.55, ground_y + size * 0.04
    body = (
      (x - size * 0.52, bottom),
      (x - size * 0.68, middle),
      (x - size * 0.38, top),
      (x + size * 0.38, top),
      (x + size * 0.68, middle),
      (x + size * 0.52, bottom),
    )
    window = (
      (x - size * 0.31, middle - size * 0.06),
      (x - size * 0.25, top + size * 0.18),
      (x + size * 0.25, top + size * 0.18),
      (x + size * 0.31, middle - size * 0.06),
    )
    if self._draw_vehicle_sprite(float(x), float(sprite_center_y), sprite_size, 0.0,
                                 self._color(UI_WHITE, (235 if controlling else 165) * alpha)):
      return

    body_alpha = 218 if controlling else 150
    rl.draw_triangle_fan(body, len(body), self._color(rl.Color(198, 207, 214, 255), body_alpha * alpha))
    rl.draw_triangle_fan(window, len(window), self._color(rl.Color(48, 62, 74, 255), 224 * alpha))

    outline_width = 1.5 if self._compact else 3.5
    outline_color = TESLA_BLUE if controlling else ROAD_MARKING
    for start, end in zip(body, (*body[1:], body[0]), strict=True):
      rl.draw_line_ex(rl.Vector2(*start), rl.Vector2(*end), outline_width * 2.8,
                      self._color(outline_color, 30 * alpha))
      rl.draw_line_ex(rl.Vector2(*start), rl.Vector2(*end), outline_width,
                      self._color(outline_color, (205 if controlling else 112) * alpha))

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
      self._smoothed_path = None
      self._smoothed_lanes = None
      self._smoothed_edges = None
      self._smoothed_leads = [None, None]
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
    path, lanes, edges = self._update_geometry(model)

    self._draw_scene_background(rect, path, self._path_offset_z, alpha)
    if car_control.latActive:
      self._draw_road_context(rect, lanes, model.laneLineProbs, edges, model.roadEdgeStds, alpha)
    if car_control.latActive and path is not None:
      self._draw_trajectory(rect, path, self._path_offset_z, alpha)
    if car_control.latActive and path is not None and lanes is not None and state.lane_change is not None:
      self._draw_lane_target(rect, path, lanes, model.laneLineProbs, state.lane_change,
                             self._path_offset_z, alpha)
    if self._longitudinal_control and car_control.longActive and state.stop is not None and path is not None:
      self._draw_stop(rect, path, state.stop, self._path_offset_z, alpha)
    if self._message_valid(sm, 'radarState'):
      controlling_index = state.controlling_lead_index if self._longitudinal_control and car_control.longActive else None
      for lead_index, lead in enumerate((sm['radarState'].leadOne, sm['radarState'].leadTwo)):
        if lead.present:
          self._draw_lead(rect, model, sm['radarState'], lead_index, self._path_offset_z, alpha,
                          controlling=lead_index == controlling_index)
    self._draw_blindspot_guard(rect, sm['carState'], alpha)
    commanded_accel = getattr(getattr(car_control, 'actuators', None), 'accel', 0.0)
    braking = bool(self._longitudinal_control and car_control.longActive and commanded_accel < -0.12)
    self._draw_ego_vehicle(rect, braking, alpha)
    self._render_state = state
