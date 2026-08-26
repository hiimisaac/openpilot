from dataclasses import dataclass
from itertools import product
import math

from opendbc.car.ford.direct_lateral_path import PATH_LIMITS as PATH_COEFFICIENT_LIMITS


PATH_PREVIEW_TIME = 1.0
PATH_MIN_PREVIEW_DISTANCE = 7.0
PATH_MAX_PREVIEW_DISTANCE = 30.0
PATH_FIT_SAMPLES = 9
PATH_C2_FULL_CURVATURE = 0.008
PATH_C2_ZERO_CURVATURE = 0.018
PATH_C2_FULL_ANGLE_DEG = 35.0
PATH_C2_ZERO_ANGLE_DEG = 80.0
@dataclass(frozen=True)
class LateralPathTarget:
  valid: bool = False
  path_offset: float = 0.0
  path_angle: float = 0.0
  curvature: float = 0.0
  curvature_rate: float = 0.0


def _finite(value: float) -> float:
  return float(value) if math.isfinite(value) else 0.0


def _clip(value: float, limits: tuple[float, float]) -> float:
  return min(max(value, limits[0]), limits[1])


def _sample(distance: float, distances: list[float], values: list[float]) -> float:
  if distance <= distances[0]:
    return values[0]
  if distance >= distances[-1]:
    return values[-1]
  for i in range(1, len(distances)):
    if distance <= distances[i]:
      span = distances[i] - distances[i - 1]
      alpha = 0.0 if span == 0.0 else (distance - distances[i - 1]) / span
      return values[i - 1] + alpha * (values[i] - values[i - 1])
  return values[-1]


def _model_path(model) -> tuple[list[float], list[float], list[float]] | None:
  if model is None:
    return None
  try:
    xs = [float(value) for value in model.position.x]
    ys = [float(value) for value in model.position.y]
    headings = [float(value) for value in model.orientation.z]
  except (AttributeError, TypeError, ValueError):
    return None
  if len(xs) < 2 or len(xs) != len(ys) or len(xs) != len(headings):
    return None
  if not all(math.isfinite(value) for values in (xs, ys, headings) for value in values):
    return None

  valid_length = 1
  while valid_length < len(xs) and xs[valid_length] > xs[valid_length - 1]:
    valid_length += 1
  xs = xs[:valid_length]
  ys = ys[:valid_length]
  headings = headings[:valid_length]
  if len(xs) < 2:
    return None

  # LMC2's cubic is lateral offset as a function of forward distance. A tight
  # turn can later flatten/recede in vehicle X, so fit its valid prefix only.
  distances = [value - xs[0] for value in xs]
  if distances[-1] <= 0.0:
    return None

  unwrapped_headings = [headings[0]]
  for heading in headings[1:]:
    delta = (heading - unwrapped_headings[-1] + math.pi) % (2.0 * math.pi) - math.pi
    unwrapped_headings.append(unwrapped_headings[-1] + delta)
  return distances, ys, unwrapped_headings


def _smoothstep(progress: float) -> float:
  progress = min(max(progress, 0.0), 1.0)
  return progress * progress * (3.0 - 2.0 * progress)


def _range_progress(value: float, full: float, zero: float) -> float:
  return (abs(value) - full) / (zero - full)


def _solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
  size = len(vector)
  augmented = [row[:] + [value] for row, value in zip(matrix, vector, strict=True)]
  for column in range(size):
    pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
    if abs(augmented[pivot][column]) < 1e-12:
      return None
    augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
    scale = augmented[column][column]
    augmented[column] = [value / scale for value in augmented[column]]
    for row in range(size):
      if row == column:
        continue
      factor = augmented[row][column]
      augmented[row] = [
        value - factor * pivot_value
        for value, pivot_value in zip(augmented[row], augmented[column], strict=True)
      ]
  return [augmented[row][size] for row in range(size)]


def _solve_bounded(normal: list[list[float]], rhs: list[float],
                   bounds: tuple[tuple[float, float], ...]) -> tuple[float, float, float] | None:
  unconstrained = _solve_linear(normal, rhs)
  if unconstrained is not None and all(lower <= value <= upper
                                       for value, (lower, upper) in zip(unconstrained, bounds, strict=True)):
    return unconstrained[0], unconstrained[1], unconstrained[2]

  best: tuple[float, tuple[float, float, float]] | None = None
  for states in product((-1, 0, 1), repeat=3):
    candidate = [0.0, 0.0, 0.0]
    fixed = [index for index, state in enumerate(states) if state]
    free = [index for index, state in enumerate(states) if not state]
    for index in fixed:
      candidate[index] = bounds[index][0 if states[index] < 0 else 1]

    if free:
      free_matrix = [[normal[row][column] for column in free] for row in free]
      free_rhs = [rhs[row] - sum(normal[row][column] * candidate[column] for column in fixed) for row in free]
      solved = _solve_linear(free_matrix, free_rhs)
      if solved is None:
        continue
      for index, value in zip(free, solved, strict=True):
        candidate[index] = value

    if any(value < lower - 1e-10 or value > upper + 1e-10
           for value, (lower, upper) in zip(candidate, bounds, strict=True)):
      continue
    objective = sum(candidate[row] * normal[row][column] * candidate[column]
                    for row in range(3) for column in range(3)) - 2.0 * sum(
                      rhs[index] * candidate[index] for index in range(3)
                    )
    coefficients = candidate[0], candidate[1], candidate[2]
    if best is None or objective < best[0]:
      best = objective, coefficients
  return None if best is None else best[1]


def _fit_fast_coefficients(path: tuple[list[float], list[float], list[float]],
                           preview_distance: float, allocated_c2: float) -> tuple[float, float, float]:
  distances, offsets, headings = path
  normal = [[0.0] * 3 for _ in range(3)]
  rhs = [0.0] * 3

  def add_equation(row: tuple[float, float, float], target: float) -> None:
    for i in range(3):
      rhs[i] += row[i] * target
      for j in range(3):
        normal[i][j] += row[i] * row[j]

  for index in range(PATH_FIT_SAMPLES):
    progress = index / (PATH_FIT_SAMPLES - 1)
    distance = progress * preview_distance
    offset = _sample(distance, distances, offsets)
    heading = _sample(distance, distances, headings)

    # Solve in distance-normalized coefficients to keep the least-squares
    # system well conditioned: a1=C1*H and a3=C3*H^3/6.
    add_equation((1.0, progress, progress ** 3), offset - 0.5 * allocated_c2 * distance ** 2)
    add_equation((0.0, 1.0, 3.0 * progress ** 2),
                 (heading - allocated_c2 * distance) * preview_distance)

  scaled_bounds = (
    PATH_COEFFICIENT_LIMITS[0],
    (PATH_COEFFICIENT_LIMITS[1][0] * preview_distance,
     PATH_COEFFICIENT_LIMITS[1][1] * preview_distance),
    (PATH_COEFFICIENT_LIMITS[3][0] * preview_distance ** 3 / 6.0,
     PATH_COEFFICIENT_LIMITS[3][1] * preview_distance ** 3 / 6.0),
  )
  solved = _solve_bounded(normal, rhs, scaled_bounds)
  if solved is None:
    return _sample(0.0, distances, offsets), _sample(0.0, distances, headings), 0.0
  path_offset, scaled_angle, scaled_cubic = solved
  return (
    _clip(path_offset, PATH_COEFFICIENT_LIMITS[0]),
    _clip(scaled_angle / preview_distance, PATH_COEFFICIENT_LIMITS[1]),
    _clip(6.0 * scaled_cubic / preview_distance ** 3, PATH_COEFFICIENT_LIMITS[3]),
  )


def model_lateral_path(model, desired_curvature: float, v_ego: float,
                       desired_angle_deg: float = 0.0) -> LateralPathTarget:
  """Fit one current-frame cubic over the PSCM's speed-scaled preview horizon."""
  desired_curvature = _finite(desired_curvature)
  v_ego = max(_finite(v_ego), 0.0)
  desired_angle_deg = _finite(desired_angle_deg)
  path = _model_path(model)
  if path is None:
    return LateralPathTarget(curvature=desired_curvature)

  distances, _, headings = path
  preview_distance = min(max(v_ego * PATH_PREVIEW_TIME, PATH_MIN_PREVIEW_DISTANCE),
                         PATH_MAX_PREVIEW_DISTANCE, distances[-1])
  average_model_curvature = (
    _sample(preview_distance, distances, headings) - _sample(0.0, distances, headings)
  ) / preview_distance
  curvature_progress = _range_progress(
    max(abs(desired_curvature), abs(average_model_curvature)),
    PATH_C2_FULL_CURVATURE,
    PATH_C2_ZERO_CURVATURE,
  )
  angle_progress = _range_progress(desired_angle_deg, PATH_C2_FULL_ANGLE_DEG, PATH_C2_ZERO_ANGLE_DEG)
  c2_ownership = 1.0 - _smoothstep(max(curvature_progress, angle_progress))
  allocated_c2 = _clip(desired_curvature * c2_ownership, PATH_COEFFICIENT_LIMITS[2])
  path_offset, path_angle, curvature_rate = _fit_fast_coefficients(path, preview_distance, allocated_c2)

  return LateralPathTarget(
    valid=True,
    path_offset=path_offset,
    path_angle=path_angle,
    curvature=allocated_c2,
    curvature_rate=curvature_rate,
  )
