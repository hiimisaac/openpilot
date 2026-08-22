from dataclasses import dataclass
import math
from statistics import median


PATH_CURVATURE_RATE_HORIZONS = (3.5, 5.0, 7.0)


@dataclass(frozen=True)
class LateralPathTarget:
  valid: bool = False
  path_offset: float = 0.0
  path_angle: float = 0.0
  curvature: float = 0.0
  curvature_rate: float = 0.0


def _finite(value: float) -> float:
  return float(value) if math.isfinite(value) else 0.0


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

  distances = [0.0]
  for i in range(1, len(xs)):
    distances.append(distances[-1] + math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]))
  if distances[-1] <= 0.0:
    return None

  unwrapped_headings = [headings[0]]
  for heading in headings[1:]:
    delta = (heading - unwrapped_headings[-1] + math.pi) % (2.0 * math.pi) - math.pi
    unwrapped_headings.append(unwrapped_headings[-1] + delta)
  return distances, ys, unwrapped_headings


def _curvature_rate(path: tuple[list[float], list[float], list[float]],
                    path_offset: float, path_angle: float,
                    desired_curvature: float) -> float:
  """Fit the one C3 that joins the current action to the future model path."""
  distances, offsets, headings = path
  rates = []
  for requested_horizon in PATH_CURVATURE_RATE_HORIZONS:
    horizon = min(requested_horizon, distances[-1])
    if horizon <= 0.0:
      continue
    offset = _sample(horizon, distances, offsets)
    heading = _sample(horizon, distances, headings)
    rates.extend((
      6.0 * (offset - path_offset - path_angle * horizon -
             0.5 * desired_curvature * horizon ** 2) / horizon ** 3,
      2.0 * (heading - path_angle - desired_curvature * horizon) / horizon ** 2,
    ))
  return median(rates) if rates else 0.0


def model_lateral_path(model, desired_curvature: float, v_ego: float) -> LateralPathTarget:
  """Fit one current-origin cubic to the model trajectory and action."""
  desired_curvature = _finite(desired_curvature)
  del v_ego  # A spatial path is independent of how quickly the vehicle traverses it.
  path = _model_path(model)
  if path is None:
    return LateralPathTarget(curvature=desired_curvature)

  distances, offsets, headings = path
  path_offset = _sample(0.0, distances, offsets)
  path_angle = _sample(0.0, distances, headings)
  return LateralPathTarget(
    valid=True,
    path_offset=path_offset,
    path_angle=path_angle,
    curvature=desired_curvature,
    curvature_rate=_curvature_rate(
      path, path_offset, path_angle, desired_curvature,
    ),
  )
