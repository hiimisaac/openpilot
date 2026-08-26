from bisect import bisect_left
from collections import deque
from dataclasses import dataclass
import math

from opendbc.car.ford.direct_lateral_path import PATH_LIMITS


MODEL_STALE_NS = 300_000_000
MOTION_HISTORY_NS = 1_000_000_000
MAX_MOTION_GAP_S = 0.2
MAX_MODEL_TIME_EXTRAPOLATION_NS = 50_000_000
MAX_REFERENCE_ERROR = 5.0
MAX_FRENET_HEADING_ERROR = min(abs(PATH_LIMITS[1][0]), PATH_LIMITS[1][1])
MIN_LONGITUDINAL_PATH_SCALE = math.cos(max(abs(PATH_LIMITS[1][0]), abs(PATH_LIMITS[1][1])))
SPATIAL_CURVATURE_RATE_LIMITS = (
  PATH_LIMITS[3][0] * MIN_LONGITUDINAL_PATH_SCALE,
  PATH_LIMITS[3][1] * MIN_LONGITUDINAL_PATH_SCALE,
)
PROJECTION_BACKTRACK_DISTANCE = 0.5
SEGMENT_TIME = 0.10
SEGMENT_DISTANCE = (0.5, 2.0)
OBSERVER_OFFSET_PROCESS_VARIANCE = 0.02 ** 2
OBSERVER_ANGLE_PROCESS_VARIANCE = math.radians(0.25) ** 2
OBSERVER_OFFSET_MEASUREMENT_VARIANCE = 0.05 ** 2
OBSERVER_ANGLE_MEASUREMENT_VARIANCE = math.radians(1.0) ** 2
GAUSS_LEGENDRE_NODES = (0.0, -0.5384693101056831, 0.5384693101056831, -0.9061798459386640, 0.9061798459386640)
GAUSS_LEGENDRE_WEIGHTS = (0.5688888888888889, 0.4786286704993665, 0.4786286704993665, 0.2369268850561891, 0.2369268850561891)


@dataclass(frozen=True)
class LateralPathTarget:
  valid: bool = False
  path_offset: float = 0.0
  path_angle: float = 0.0
  curvature: float = 0.0
  curvature_rate: float = 0.0


@dataclass(frozen=True)
class _Pose:
  mono_time_ns: int
  x: float
  y: float
  yaw: float
  odometer: float
  speed: float
  yaw_rate: float


@dataclass(frozen=True)
class _PathSample:
  x: float
  y: float
  heading: float


@dataclass(frozen=True)
class _Projection:
  station: float
  x: float
  y: float
  heading: float
  distance: float


def _clip(value: float, limits: tuple[float, float]) -> float:
  return min(max(value, limits[0]), limits[1])


def _wrap(angle: float) -> float:
  return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _derivative_weights(offsets: list[float], order: int) -> list[float]:
  """Finite-difference weights at zero for an arbitrary local spatial grid."""
  size = len(offsets)
  matrix = [[offset ** power for offset in offsets] for power in range(size)]
  result = [float(math.factorial(order) if power == order else 0.0) for power in range(size)]
  for column in range(size):
    pivot = max(range(column, size), key=lambda row: abs(matrix[row][column]))
    if abs(matrix[pivot][column]) < 1e-12:
      raise ValueError("reference derivative grid is singular")
    matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
    result[column], result[pivot] = result[pivot], result[column]
    scale = matrix[column][column]
    matrix[column] = [value / scale for value in matrix[column]]
    result[column] /= scale
    for row in range(size):
      if row == column:
        continue
      scale = matrix[row][column]
      matrix[row] = [value - scale * pivot_value for value, pivot_value in zip(matrix[row], matrix[column], strict=True)]
      result[row] -= scale * result[column]
  return result


def _advance_pose(pose: _Pose, dt: float, speed: float, yaw_rate: float, mono_time_ns: int) -> _Pose:
  average_speed = 0.5 * (pose.speed + speed)
  average_yaw_rate = 0.5 * (pose.yaw_rate + yaw_rate)
  yaw_delta = average_yaw_rate * dt
  if abs(average_yaw_rate) > 1e-6:
    radius = average_speed / average_yaw_rate
    local_x = radius * math.sin(yaw_delta)
    local_y = radius * (1.0 - math.cos(yaw_delta))
  else:
    local_x = average_speed * dt
    local_y = 0.0
  cosine = math.cos(pose.yaw)
  sine = math.sin(pose.yaw)
  return _Pose(
    mono_time_ns,
    pose.x + cosine * local_x - sine * local_y,
    pose.y + sine * local_x + cosine * local_y,
    pose.yaw + yaw_delta,
    pose.odometer + average_speed * dt,
    speed,
    yaw_rate,
  )


class _ReferencePath:
  def __init__(self, points: list[_PathSample]):
    if len(points) < 3:
      raise ValueError("reference path needs at least three points")

    filtered_positions = [(points[0].x, points[0].y)]
    for point in points[1:]:
      if math.hypot(point.x - filtered_positions[-1][0], point.y - filtered_positions[-1][1]) > 1e-3:
        filtered_positions.append((point.x, point.y))
    if len(filtered_positions) < 3:
      raise ValueError("reference path has no spatial extent")

    stations = [0.0]
    for previous, point in zip(filtered_positions, filtered_positions[1:], strict=False):
      stations.append(stations[-1] + math.hypot(point[0] - previous[0], point[1] - previous[1]))
    if stations[-1] < 1.0:
      raise ValueError("reference path is too short")

    # One Ford LMC2 tuple must be one geometric jet. Every query evaluates one
    # local parametric position polynomial; heading, curvature, and curvature
    # rate are derivatives of that same polynomial. The five-point window is
    # local, so a far-field model replan cannot alter the committed near field.
    self._positions = tuple(filtered_positions)
    self.stations = tuple(stations)
    self._window_size = min(5, len(self._positions))
    self._windows: dict[int, tuple[float, tuple[float, ...], tuple[float, ...]]] = {}
    # Stored headings are intentionally not a second geometry source. All
    # consumers obtain tangent from _evaluate() on the position curve.
    self.points = tuple(_PathSample(x, y, 0.0) for x, y in self._positions)

  @property
  def length(self) -> float:
    return self.stations[-1]

  def _window(self, start: int) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    if start not in self._windows:
      origin = self.stations[start]
      indices = tuple(range(start, start + self._window_size))
      offsets = [self.stations[selected] - origin for selected in indices]
      weights = [_derivative_weights(offsets, order) for order in range(self._window_size)]
      coefficients = [
        tuple(
          sum(weight * self._positions[selected][coordinate]
              for weight, selected in zip(order_weights, indices, strict=True)) / math.factorial(order)
          for order, order_weights in enumerate(weights)
        )
        for coordinate in range(2)
      ]
      self._windows[start] = origin, coefficients[0], coefficients[1]
    return self._windows[start]

  def _evaluate(self, station: float) -> tuple[float, float, float, float, float, float]:
    station = _clip(station, (0.0, self.length))
    index = bisect_left(self.stations, station)
    start = min(max(index - 2, 0), len(self._positions) - self._window_size)
    origin, x_coefficients, y_coefficients = self._window(start)
    offset = station - origin
    coordinate_derivatives = []
    for coefficients in (x_coefficients, y_coefficients):
      derivatives = []
      for order in range(3):
        derivatives.append(sum(
          math.factorial(power) / math.factorial(power - order) * coefficient * offset ** (power - order)
          for power, coefficient in enumerate(coefficients) if power >= order
        ))
      coordinate_derivatives.append(derivatives)
    (x, dx, ddx), (y, dy, ddy) = coordinate_derivatives
    return x, y, dx, dy, ddx, ddy

  def sample(self, station: float) -> _PathSample:
    x, y, dx, dy, _, _ = self._evaluate(station)
    return _PathSample(x, y, math.atan2(dy, dx))

  def project(self, x: float, y: float, *, heading: float | None = None,
              station_hint: float | None = None, min_station: float | None = None,
              max_station: float | None = None) -> _Projection:
    best: _Projection | None = None
    best_score: tuple[float, float, float] | None = None
    for index, (first, second) in enumerate(zip(self.points, self.points[1:], strict=False)):
      chord_x = second.x - first.x
      chord_y = second.y - first.y
      length_squared = chord_x * chord_x + chord_y * chord_y
      segment_length = math.sqrt(length_squared)
      station_start = self.stations[index]
      station_end = self.stations[index + 1]
      allowed_start = station_start if min_station is None else max(station_start, min_station)
      allowed_end = station_end if max_station is None else min(station_end, max_station)
      if allowed_start > allowed_end:
        continue
      alpha_limits = (
        0.0 if segment_length < 1e-12 else (allowed_start - station_start) / segment_length,
        1.0 if segment_length < 1e-12 else (allowed_end - station_start) / segment_length,
      )
      alpha = 0.0 if length_squared < 1e-12 else _clip(
        ((x - first.x) * chord_x + (y - first.y) * chord_y) / length_squared,
        alpha_limits,
      )
      station = station_start + alpha * segment_length
      for _ in range(5):
        projected_x, projected_y, dx, dy, ddx, ddy = self._evaluate(station)
        gradient = (projected_x - x) * dx + (projected_y - y) * dy
        hessian = dx * dx + dy * dy + (projected_x - x) * ddx + (projected_y - y) * ddy
        if abs(hessian) < 1e-9:
          break
        updated_station = _clip(station - gradient / hessian, (allowed_start, allowed_end))
        if abs(updated_station - station) < 1e-6:
          station = updated_station
          break
        station = updated_station
      projected_x, projected_y, dx, dy, _, _ = self._evaluate(station)
      distance = math.hypot(x - projected_x, y - projected_y)
      projected_heading = math.atan2(dy, dx)
      heading_error = abs(_wrap(projected_heading - heading)) if heading is not None else 0.0
      if heading_error > MAX_FRENET_HEADING_ERROR:
        continue
      station_error = abs(station - station_hint) if station_hint is not None else 0.0
      # Ford defines C0 at the nearest point on the active reference line.
      # Heading and station continuity select/reject a topological branch; they
      # must never move the projection away from the Euclidean nearest point.
      score = distance, heading_error, station_error
      if best_score is None or score < best_score:
        best_score = score
        best = _Projection(
          station,
          projected_x,
          projected_y,
          projected_heading,
          distance,
        )
    if best is None:
      raise ValueError("reference path cannot be projected")
    return best


@dataclass(frozen=True)
class _PathSegment:
  x: float
  y: float
  heading: float
  # Desired reference geometry remains unbounded so C0/C1 measure the real
  # Frenet error to the requested path. The wire jet is the same feedforward
  # geometry represented inside LMC2's C2/C3 envelope.
  curvature: float
  curvature_rate: float
  wire_curvature: float
  wire_curvature_rate: float
  length: float
  start_odometer: float
  generation: int
  source_ns: int
  provisional: bool = False

  @property
  def end_odometer(self) -> float:
    return self.start_odometer + self.length

  def jet(self, station: float) -> tuple[float, float]:
    station = _clip(station, (0.0, self.length))
    return self.curvature + self.curvature_rate * station, self.curvature_rate

  def wire_jet(self, station: float) -> tuple[float, float]:
    station = _clip(station, (0.0, self.length))
    return self.wire_curvature + self.wire_curvature_rate * station, self.wire_curvature_rate

  def sample(self, station: float) -> _PathSample:
    station = _clip(station, (0.0, self.length))
    if station <= 0.0:
      return _PathSample(self.x, self.y, self.heading)

    # Five-point Gauss-Legendre quadrature integrates the short clothoid to
    # sub-millimetre accuracy without adding path samples or spline seams.
    midpoint = 0.5 * station
    integrated_x = integrated_y = 0.0
    for node, weight in zip(GAUSS_LEGENDRE_NODES, GAUSS_LEGENDRE_WEIGHTS, strict=True):
      distance = midpoint * (node + 1.0)
      heading = self.heading + self.curvature * distance + 0.5 * self.curvature_rate * distance ** 2
      integrated_x += weight * math.cos(heading)
      integrated_y += weight * math.sin(heading)
    heading = self.heading + self.curvature * station + 0.5 * self.curvature_rate * station ** 2
    return _PathSample(
      self.x + midpoint * integrated_x,
      self.y + midpoint * integrated_y,
      heading,
    )

  def project(self, x: float, y: float, expected_station: float) -> _Projection:
    station = _clip(expected_station, (0.0, self.length))
    for _ in range(8):
      point = self.sample(station)
      curvature, _ = self.jet(station)
      tangent_x = math.cos(point.heading)
      tangent_y = math.sin(point.heading)
      normal_x = -tangent_y
      normal_y = tangent_x
      relative_x = point.x - x
      relative_y = point.y - y
      gradient = relative_x * tangent_x + relative_y * tangent_y
      hessian = 1.0 + curvature * (relative_x * normal_x + relative_y * normal_y)
      if abs(hessian) < 1e-6:
        break
      updated = _clip(station - gradient / hessian, (0.0, self.length))
      if abs(updated - station) < 1e-6:
        station = updated
        break
      station = updated
    point = self.sample(station)
    return _Projection(station, point.x, point.y, point.heading, math.hypot(x - point.x, y - point.y))


@dataclass(frozen=True)
class _PendingSegment:
  segment: _PathSegment
  for_generation: int


def _fit_segment_jet(path: _ReferencePath, station: float, length: float,
                     start_curvature: float | None = None) -> tuple[float, float]:
  """Fit one local Ford cubic, optionally constraining curvature continuity."""
  length = min(length, path.length - station)
  if length < 0.25:
    return (start_curvature if start_curvature is not None else 0.0), 0.0
  origin_heading = path.sample(station).heading
  distances = [length * index / 4.0 for index in range(1, 5)]
  heading_deltas = [_wrap(path.sample(station + distance).heading - origin_heading) for distance in distances]
  normal_00 = normal_01 = normal_11 = rhs_0 = rhs_1 = 0.0
  for distance, heading_delta in zip(distances, heading_deltas, strict=True):
    linear = distance
    quadratic = 0.5 * distance ** 2
    normal_00 += linear ** 2
    normal_01 += linear * quadratic
    normal_11 += quadratic ** 2
    rhs_0 += linear * heading_delta
    rhs_1 += quadratic * heading_delta
  determinant = normal_00 * normal_11 - normal_01 ** 2
  if abs(determinant) < 1e-12:
    return 0.0, 0.0
  curvature, curvature_rate = (
    (rhs_0 * normal_11 - rhs_1 * normal_01) / determinant,
    (normal_00 * rhs_1 - normal_01 * rhs_0) / determinant,
  )
  if start_curvature is not None:
    target_end_curvature = curvature + curvature_rate * length
    return start_curvature, (target_end_curvature - start_curvature) / length
  return curvature, curvature_rate


def _model_reference(model, capture_pose: _Pose) -> _ReferencePath | None:
  try:
    xs = [float(value) for value in model.position.x]
    ys = [float(value) for value in model.position.y]
    headings = [float(value) for value in model.orientation.z]
  except (AttributeError, TypeError, ValueError):
    return None
  if len(xs) < 3 or len(xs) != len(ys) or len(xs) != len(headings):
    return None
  if not all(math.isfinite(value) for values in (xs, ys, headings) for value in values):
    return None

  cosine = math.cos(capture_pose.yaw)
  sine = math.sin(capture_pose.yaw)
  points = [
    _PathSample(
      capture_pose.x + cosine * x - sine * y,
      capture_pose.y + sine * x + cosine * y,
      capture_pose.yaw + heading,
    )
    for x, y, heading in zip(xs, ys, headings, strict=True)
  ]
  try:
    return _ReferencePath(points)
  except ValueError:
    return None


def _segment_length(speed: float) -> float:
  return _clip(SEGMENT_TIME * max(speed, 0.0), SEGMENT_DISTANCE)


def _bounded_wire_jet(curvature: float, curvature_rate: float, length: float,
                      start_curvature: float | None = None) -> tuple[float, float]:
  """Represent the desired segment with one spatially coherent bounded C2/C3 jet."""
  wire_start = _clip(curvature if start_curvature is None else start_curvature, PATH_LIMITS[2])
  desired_end = _clip(curvature + curvature_rate * length, PATH_LIMITS[2])
  # C3 is d-kappa/dx. Conservatively bound the underlying d-kappa/ds so the
  # local conversion remains representable throughout the complete C1 range.
  wire_rate = _clip((desired_end - wire_start) / length, SPATIAL_CURVATURE_RATE_LIMITS) if length > 1e-6 else 0.0
  return wire_start, wire_rate


class FrenetErrorObserver:
  """Two-state observer using the Frenet kinematics described by US20200047752."""

  def __init__(self) -> None:
    self._state: tuple[float, float] | None = None
    self._covariance = (0.0, 0.0, 0.0, 0.0)
    self._last_update_ns = 0

  @property
  def state(self) -> tuple[float, float] | None:
    return self._state

  def reset(self) -> None:
    self._state = None
    self._covariance = (0.0, 0.0, 0.0, 0.0)
    self._last_update_ns = 0

  def update(self, mono_time_ns: int, *, speed: float, reference_curvature: float, yaw_rate: float,
             measurement: tuple[float, float] | None = None) -> tuple[float, float] | None:
    values = (speed, reference_curvature, yaw_rate)
    if mono_time_ns <= 0 or not all(math.isfinite(value) for value in values):
      self.reset()
      return None
    if measurement is not None and not all(math.isfinite(value) for value in measurement):
      self.reset()
      return None

    if self._state is None:
      if measurement is None:
        return None
      self._state = measurement[0], _wrap(measurement[1])
      self._covariance = (
        OBSERVER_OFFSET_MEASUREMENT_VARIANCE, 0.0,
        0.0, OBSERVER_ANGLE_MEASUREMENT_VARIANCE,
      )
      self._last_update_ns = mono_time_ns
      return self._state

    dt = (mono_time_ns - self._last_update_ns) * 1e-9
    if dt < 0.0 or dt > MAX_MOTION_GAP_S:
      self.reset()
      return self.update(
        mono_time_ns,
        speed=speed,
        reference_curvature=reference_curvature,
        yaw_rate=yaw_rate,
        measurement=measurement,
      )

    path_offset, path_angle = self._state
    p00, p01, p10, p11 = self._covariance
    if dt > 0.0:
      # Ford's small-angle Frenet model: e_y_dot=Vx*e_psi,
      # e_psi_dot=Vx*kappa_ref-yawRate.
      heading_rate = speed * reference_curvature - yaw_rate
      path_offset += speed * path_angle * dt
      path_angle = _wrap(path_angle + heading_rate * dt)

      transition = speed * dt
      p00, p01, p10, p11 = (
        p00 + transition * (p01 + p10) + transition ** 2 * p11 + OBSERVER_OFFSET_PROCESS_VARIANCE * dt,
        p01 + transition * p11,
        p10 + transition * p11,
        p11 + OBSERVER_ANGLE_PROCESS_VARIANCE * dt,
      )
    self._last_update_ns = mono_time_ns

    if measurement is not None:
      innovation_offset = measurement[0] - path_offset
      innovation_angle = _wrap(measurement[1] - path_angle)
      s00 = p00 + OBSERVER_OFFSET_MEASUREMENT_VARIANCE
      s01 = p01
      s10 = p10
      s11 = p11 + OBSERVER_ANGLE_MEASUREMENT_VARIANCE
      determinant = s00 * s11 - s01 * s10
      if determinant <= 1e-15:
        self.reset()
        return None

      gain00 = (p00 * s11 - p01 * s10) / determinant
      gain01 = (-p00 * s01 + p01 * s00) / determinant
      gain10 = (p10 * s11 - p11 * s10) / determinant
      gain11 = (-p10 * s01 + p11 * s00) / determinant
      path_offset += gain00 * innovation_offset + gain01 * innovation_angle
      path_angle = _wrap(path_angle + gain10 * innovation_offset + gain11 * innovation_angle)

      new_p00 = (1.0 - gain00) * p00 - gain01 * p10
      new_p01 = (1.0 - gain00) * p01 - gain01 * p11
      new_p10 = -gain10 * p00 + (1.0 - gain11) * p10
      new_p11 = -gain10 * p01 + (1.0 - gain11) * p11
      cross = 0.5 * (new_p01 + new_p10)
      p00, p01, p10, p11 = max(new_p00, 0.0), cross, cross, max(new_p11, 0.0)

    self._state = path_offset, path_angle
    self._covariance = p00, p01, p10, p11
    return self._state


class PersistentLateralPath:
  """Turn model trajectories into Ford's persistent local Frenet path state."""

  def __init__(self) -> None:
    self._poses: deque[_Pose] = deque()
    self._active_segment: _PathSegment | None = None
    self._pending_segment: _PendingSegment | None = None
    self._observer = FrenetErrorObserver()
    self._last_model_key: tuple[int, int] | None = None
    self._last_model_source_ns = 0
    self._generation = 0

  def reset(self) -> None:
    self._poses.clear()
    self._clear_reference()

  def _clear_reference(self) -> None:
    self._active_segment = None
    self._pending_segment = None
    self._observer.reset()
    self._last_model_key = None
    self._last_model_source_ns = 0
    self._generation = 0

  def _update_motion(self, mono_time_ns: int, speed: float, yaw_rate: float) -> _Pose | None:
    if mono_time_ns <= 0 or not all(math.isfinite(value) for value in (speed, yaw_rate)):
      self.reset()
      return None
    speed = max(speed, 0.0)
    if not self._poses:
      pose = _Pose(mono_time_ns, 0.0, 0.0, 0.0, 0.0, speed, yaw_rate)
      self._poses.append(pose)
      return pose

    previous = self._poses[-1]
    dt = (mono_time_ns - previous.mono_time_ns) * 1e-9
    if dt < 0.0 or dt > MAX_MOTION_GAP_S:
      self.reset()
      pose = _Pose(mono_time_ns, 0.0, 0.0, 0.0, 0.0, speed, yaw_rate)
      self._poses.append(pose)
      return pose
    if dt == 0.0:
      pose = _Pose(mono_time_ns, previous.x, previous.y, previous.yaw, previous.odometer, speed, yaw_rate)
      self._poses[-1] = pose
      return pose

    pose = _advance_pose(previous, dt, speed, yaw_rate, mono_time_ns)
    self._poses.append(pose)
    while len(self._poses) > 1 and self._poses[1].mono_time_ns < mono_time_ns - MOTION_HISTORY_NS:
      self._poses.popleft()
    return pose

  def _pose_at(self, mono_time_ns: int) -> _Pose | None:
    if not self._poses:
      return None
    times = [pose.mono_time_ns for pose in self._poses]
    index = bisect_left(times, mono_time_ns)
    if index == 0:
      first = self._poses[0]
      delta_ns = mono_time_ns - first.mono_time_ns
      if abs(delta_ns) > MAX_MODEL_TIME_EXTRAPOLATION_NS:
        return None
      return _advance_pose(first, delta_ns * 1e-9, first.speed, first.yaw_rate, mono_time_ns)
    if index >= len(self._poses):
      last = self._poses[-1]
      delta_ns = mono_time_ns - last.mono_time_ns
      if delta_ns > MAX_MODEL_TIME_EXTRAPOLATION_NS:
        return None
      return _advance_pose(last, delta_ns * 1e-9, last.speed, last.yaw_rate, mono_time_ns)

    first = self._poses[index - 1]
    second = self._poses[index]
    progress = (mono_time_ns - first.mono_time_ns) / (second.mono_time_ns - first.mono_time_ns)
    return _Pose(
      mono_time_ns,
      first.x + progress * (second.x - first.x),
      first.y + progress * (second.y - first.y),
      first.yaw + progress * (second.yaw - first.yaw),
      first.odometer + progress * (second.odometer - first.odometer),
      first.speed + progress * (second.speed - first.speed),
      first.yaw_rate + progress * (second.yaw_rate - first.yaw_rate),
    )

  def _promote_crossed_segment(self, pose: _Pose, speed: float) -> bool:
    promoted = False
    while self._active_segment is not None and pose.odometer >= self._active_segment.end_odometer - 1e-3:
      active = self._active_segment
      pending = self._pending_segment
      if (pending is not None and pending.for_generation == active.generation and
          pose.mono_time_ns - pending.segment.source_ns <= MODEL_STALE_NS):
        self._active_segment = pending.segment
      else:
        # Continue the same committed cubic through a brief model dropout.
        endpoint = active.sample(active.length)
        endpoint_curvature, _ = active.jet(active.length)
        endpoint_wire_curvature, _ = active.wire_jet(active.length)
        length = _segment_length(speed)
        wire_curvature, wire_curvature_rate = _bounded_wire_jet(
          endpoint_curvature, active.curvature_rate, length, endpoint_wire_curvature,
        )
        self._active_segment = _PathSegment(
          endpoint.x, endpoint.y, endpoint.heading,
          endpoint_curvature, active.curvature_rate,
          wire_curvature, wire_curvature_rate,
          length, active.end_odometer, active.generation + 1, active.source_ns,
          provisional=True,
        )
      self._pending_segment = None
      self._generation += 1
      promoted = True
    return promoted

  def _bootstrap_reference(self, fresh: _ReferencePath, capture_pose: _Pose, current_pose: _Pose,
                           fresh_origin: _Projection, speed: float) -> bool:
    expected_station = min(fresh_origin.station + max(current_pose.odometer - capture_pose.odometer, 0.0), fresh.length)
    try:
      current_projection = fresh.project(
        current_pose.x, current_pose.y,
        heading=current_pose.yaw,
        station_hint=expected_station,
        min_station=max(expected_station - PROJECTION_BACKTRACK_DISTANCE, 0.0),
        max_station=min(expected_station + 1.0, fresh.length),
      )
    except ValueError:
      return False
    if current_projection.distance > MAX_REFERENCE_ERROR:
      return False
    length = min(_segment_length(speed), fresh.length - current_projection.station)
    if length < 0.25:
      return False
    curvature, curvature_rate = _fit_segment_jet(fresh, current_projection.station, length)
    wire_curvature, wire_curvature_rate = _bounded_wire_jet(curvature, curvature_rate, length)
    start = fresh.sample(current_projection.station)
    self._generation += 1
    self._active_segment = _PathSegment(
      start.x, start.y, start.heading,
      curvature, curvature_rate,
      wire_curvature, wire_curvature_rate,
      length, current_pose.odometer, self._generation, capture_pose.mono_time_ns,
    )
    self._pending_segment = None
    return True

  def _build_pending(self, fresh: _ReferencePath, fresh_origin: _Projection,
                     capture_pose: _Pose, speed: float, source_ns: int) -> bool:
    active = self._active_segment
    if active is None:
      return False
    remaining_to_boundary = active.end_odometer - capture_pose.odometer
    if remaining_to_boundary < -0.5:
      return False
    fresh_station = fresh_origin.station + max(remaining_to_boundary, 0.0)
    length = min(_segment_length(speed), fresh.length - fresh_station)
    if length < 0.25:
      return False
    endpoint = active.sample(active.length)
    endpoint_curvature, _ = active.jet(active.length)
    endpoint_wire_curvature, _ = active.wire_jet(active.length)
    curvature, curvature_rate = _fit_segment_jet(fresh, fresh_station, length, start_curvature=endpoint_curvature)
    wire_curvature, wire_curvature_rate = _bounded_wire_jet(
      curvature, curvature_rate, length, endpoint_wire_curvature,
    )
    segment = _PathSegment(
      endpoint.x, endpoint.y, endpoint.heading,
      curvature, curvature_rate,
      wire_curvature, wire_curvature_rate,
      length, active.end_odometer, active.generation + 1, source_ns,
    )
    self._pending_segment = _PendingSegment(segment, active.generation)
    return True

  def _replace_unconsumed_standstill_segment(self, fresh: _ReferencePath, fresh_origin: _Projection,
                                              current_pose: _Pose, speed: float, source_ns: int) -> bool:
    active = self._active_segment
    if active is None or speed >= 0.3 or current_pose.odometer - active.start_odometer >= 0.05:
      return False
    length = min(_segment_length(speed), fresh.length - fresh_origin.station)
    if length < 0.25:
      return False
    active_projection = active.project(
      current_pose.x,
      current_pose.y,
      max(current_pose.odometer - active.start_odometer, 0.0),
    )
    if active_projection.distance > MAX_REFERENCE_ERROR:
      return False
    curvature, curvature_rate = _fit_segment_jet(fresh, fresh_origin.station, length)
    wire_curvature, wire_curvature_rate = _bounded_wire_jet(curvature, curvature_rate, length)
    self._generation += 1
    self._active_segment = _PathSegment(
      active_projection.x, active_projection.y, active_projection.heading,
      curvature, curvature_rate,
      wire_curvature, wire_curvature_rate,
      length, current_pose.odometer, self._generation, source_ns,
    )
    self._pending_segment = None
    return True

  def _replace_unconsumed_fallback_segment(self, fresh: _ReferencePath, fresh_origin: _Projection,
                                            capture_pose: _Pose, current_pose: _Pose,
                                            speed: float, source_ns: int) -> bool:
    """Let a just-arrived plan replace only the uncommitted tail of a dropout fallback."""
    active = self._active_segment
    if active is None or not active.provisional:
      return False
    consumed = max(current_pose.odometer - active.start_odometer, 0.0)
    if consumed > PROJECTION_BACKTRACK_DISTANCE:
      return False
    fresh_station = fresh_origin.station + max(current_pose.odometer - capture_pose.odometer, 0.0)
    length = min(_segment_length(speed), fresh.length - fresh_station)
    if length < 0.25:
      return False

    start = active.sample(consumed)
    start_curvature, _ = active.jet(consumed)
    start_wire_curvature, _ = active.wire_jet(consumed)
    curvature, curvature_rate = _fit_segment_jet(fresh, fresh_station, length, start_curvature=start_curvature)
    wire_curvature, wire_curvature_rate = _bounded_wire_jet(
      curvature, curvature_rate, length, start_wire_curvature,
    )
    self._generation += 1
    self._active_segment = _PathSegment(
      start.x, start.y, start.heading,
      curvature, curvature_rate,
      wire_curvature, wire_curvature_rate,
      length, current_pose.odometer, self._generation, source_ns,
    )
    self._pending_segment = None
    return True

  def update(self, model, *, active: bool,
             mono_time_ns: int, v_ego: float, yaw_rate: float) -> LateralPathTarget:
    current_pose = self._update_motion(mono_time_ns, v_ego, yaw_rate)
    if current_pose is None:
      return LateralPathTarget()
    if not active:
      self._clear_reference()
      return LateralPathTarget()

    promoted = False
    accepted_model = False
    if model is not None:
      try:
        timestamp_eof = int(model.timestampEof)
        frame_id = int(model.frameId)
      except (AttributeError, TypeError, ValueError):
        timestamp_eof = frame_id = 0

      model_key = frame_id, timestamp_eof
      if timestamp_eof > 0 and model_key != self._last_model_key:
        capture_pose = self._pose_at(timestamp_eof)
        fresh = _model_reference(model, capture_pose) if capture_pose is not None else None
        if capture_pose is not None and fresh is not None:
          try:
            fresh_origin = fresh.project(
              capture_pose.x, capture_pose.y,
              heading=capture_pose.yaw,
              station_hint=0.0,
              min_station=0.0,
              max_station=min(2.0, fresh.length),
            )
          except ValueError:
            fresh_origin = None
        else:
          fresh_origin = None
        if capture_pose is not None and fresh is not None and fresh_origin is not None:
          if self._active_segment is None:
            accepted_model = self._bootstrap_reference(
              fresh, capture_pose, current_pose, fresh_origin, max(v_ego, 0.0),
            )
          elif self._replace_unconsumed_standstill_segment(
            fresh, fresh_origin, current_pose, max(v_ego, 0.0), timestamp_eof,
          ):
            accepted_model = True
          elif self._replace_unconsumed_fallback_segment(
            fresh, fresh_origin, capture_pose, current_pose, max(v_ego, 0.0), timestamp_eof,
          ):
            accepted_model = True
          else:
            accepted_model = self._build_pending(
              fresh, fresh_origin, capture_pose, max(v_ego, 0.0), timestamp_eof,
            )
          if accepted_model:
            self._last_model_key = model_key
            self._last_model_source_ns = timestamp_eof

    # Ingest a frame captured at or just beyond this boundary before choosing
    # the fallback segment, so fresh maneuver geometry is never delayed by a
    # complete additional segment.
    promoted |= self._promote_crossed_segment(current_pose, max(v_ego, 0.0))

    active_segment = self._active_segment
    if active_segment is None or mono_time_ns - self._last_model_source_ns > MODEL_STALE_NS:
      self._clear_reference()
      return LateralPathTarget()

    expected_station = max(current_pose.odometer - active_segment.start_odometer, 0.0)
    projection = active_segment.project(current_pose.x, current_pose.y, expected_station)
    if projection.distance > MAX_REFERENCE_ERROR:
      self._clear_reference()
      return LateralPathTarget()

    path_angle = _wrap(projection.heading - current_pose.yaw)
    if abs(path_angle) > MAX_FRENET_HEADING_ERROR:
      self._clear_reference()
      return LateralPathTarget()
    normal_x = -math.sin(projection.heading)
    normal_y = math.cos(projection.heading)
    path_offset = (projection.x - current_pose.x) * normal_x + (projection.y - current_pose.y) * normal_y
    reference_curvature, _ = active_segment.jet(projection.station)
    curvature, curvature_rate_ds = active_segment.wire_jet(projection.station)
    estimate = self._observer.update(
      mono_time_ns,
      speed=max(v_ego, 0.0),
      reference_curvature=reference_curvature,
      yaw_rate=yaw_rate,
      measurement=(path_offset, path_angle) if accepted_model or promoted else None,
    )
    if estimate is None:
      return LateralPathTarget()
    wire_path_angle = _clip(estimate[1], PATH_LIMITS[1])
    curvature_rate = curvature_rate_ds / max(math.cos(wire_path_angle), MIN_LONGITUDINAL_PATH_SCALE)
    coefficients = tuple(
      _clip(value, limits)
      for value, limits in zip((*estimate, curvature, curvature_rate), PATH_LIMITS, strict=True)
    )
    return LateralPathTarget(True, *coefficients)
