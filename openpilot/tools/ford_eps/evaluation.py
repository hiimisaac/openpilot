from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import numpy as np

from openpilot.tools.ford_eps.controller import FordEpsCommandPlanner, FordEpsPlanRequest, FordEpsPlannerConfig
from openpilot.tools.ford_eps.dataset import FordEpsDataset, input_from_sample
from openpilot.tools.ford_eps.model import FordEpsModel


@dataclass(frozen=True)
class FordEpsRoutePlannerMetrics:
  route_id: str
  supported_window_count: int
  predicted_baseline_angle_mae_deg: float
  predicted_planned_angle_mae_deg: float


@dataclass(frozen=True)
class FordEpsPlannerEvaluation:
  evidence_scope: str
  route_count: int
  routes: tuple[str, ...]
  requested_window_count: int
  supported_window_count: int
  rejected_window_count: int
  predicted_baseline_angle_mae_deg: float
  predicted_planned_angle_mae_deg: float
  predicted_baseline_first_angle_mae_deg: float
  predicted_planned_first_angle_mae_deg: float
  predicted_improved_fraction: float
  predicted_baseline_limit_fraction: float
  predicted_planned_limit_fraction: float
  minimum_confidence: float
  mean_abs_coefficient_delta: tuple[float, float, float, float]
  route_baseline_angle_mae_p90_deg: float
  route_planned_angle_mae_p90_deg: float
  route_metrics: tuple[FordEpsRoutePlannerMetrics, ...]

  def to_dict(self) -> dict:
    return asdict(self)

  def write_json(self, path: str | Path) -> None:
    Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


def _window_starts(dataset: FordEpsDataset, routes: tuple[str, ...], horizon_steps: int,
                   stride: int) -> dict[str, list[int]]:
  samples = dataset.samples
  starts = {route: [] for route in routes}
  for start in range(6, len(samples) - horizon_steps, max(stride, 1)):
    end = start + horizon_steps
    route = str(samples[start]["route_id"])
    if route not in starts or samples[start - 6]["segment_id"] != samples[end]["segment_id"]:
      continue
    window = samples[start - 6:end + 1]
    if not np.all(window["lat_active"]) or np.any(window["steering_pressed"]):
      continue
    starts[route].append(start)
  return starts


def _sample_evenly(starts: dict[str, list[int]], max_windows: int | None) -> list[int]:
  selected = [index for route_starts in starts.values() for index in route_starts]
  if max_windows is None or len(selected) <= max_windows:
    return selected
  populated = [route_starts for route_starts in starts.values() if route_starts]
  sampled: list[int] = []
  remaining = max_windows
  for route_number, route_starts in enumerate(populated):
    route_budget = max(1, remaining // (len(populated) - route_number))
    count = min(route_budget, len(route_starts))
    positions = np.linspace(0, len(route_starts) - 1, count).round().astype(np.int64)
    sampled.extend(route_starts[int(position)] for position in positions)
    remaining = max_windows - len(sampled)
    if remaining == 0:
      break
  return sorted(sampled)


def evaluate_planner(dataset: FordEpsDataset, model: FordEpsModel, routes: tuple[str, ...], *,
                     config: FordEpsPlannerConfig | None = None, horizon_steps: int = 5,
                     stride: int = 20, max_windows: int | None = None) -> FordEpsPlannerEvaluation:
  """Measure model-internal planner consistency on complete held-out routes."""
  if horizon_steps < 1:
    raise ValueError("planner evaluation horizon must be positive")
  if max_windows is not None and max_windows < 1:
    raise ValueError("maximum planner windows must be positive")
  available_routes = set(dataset.samples["route_id"])
  missing = set(routes) - available_routes
  if missing:
    raise ValueError(f"planner evaluation routes are absent from the dataset: {sorted(missing)}")
  planner = FordEpsCommandPlanner(model, config)
  starts_by_route = _window_starts(dataset, routes, horizon_steps, stride)
  starts = _sample_evenly(starts_by_route, max_windows)
  plans = []
  route_plans: dict[str, list] = {route: [] for route in routes}
  samples = dataset.samples
  for start in starts:
    sample = samples[start]
    history = tuple(input_from_sample(samples[index]) for index in range(start - 6, start + 1))
    current_environment = history[-1]
    baseline = tuple(
      replace(
        current_environment,
        c0=float(samples[index]["c0"]), c1=float(samples[index]["c1"]),
        c2=float(samples[index]["c2"]), c3=float(samples[index]["c3"]),
      )
      for index in range(start + 1, start + horizon_steps + 1)
    )
    request = FordEpsPlanRequest(
      pinion_angle_deg=float(sample["pinion_angle_deg"]),
      steering_rate_deg_s=float(sample["steering_rate_deg_s"]),
      eps_current_a=float(sample["eps_current_a"]),
      command_history=history,
      baseline_commands=baseline,
      desired_angles_deg=tuple(
        float(samples[index]["desired_angle_deg"])
        for index in range(start + 1, start + horizon_steps + 1)
      ),
    )
    try:
      plan = planner.plan(request)
    except ValueError:
      continue
    plans.append((plan, baseline))
    route_plans[str(sample["route_id"])].append(plan)

  route_metrics = tuple(
    FordEpsRoutePlannerMetrics(
      route_id=route,
      supported_window_count=len(route_values),
      predicted_baseline_angle_mae_deg=_mean([plan.baseline_angle_mae_deg for plan in route_values]),
      predicted_planned_angle_mae_deg=_mean([plan.predicted_angle_mae_deg for plan in route_values]),
    )
    for route, route_values in route_plans.items()
  )
  coefficient_deltas = [
    [abs(getattr(command, field) - getattr(original, field)) for field in ("c0", "c1", "c2", "c3")]
    for plan, baseline in plans for command, original in zip(plan.commands, baseline, strict=True)
  ]
  baseline_route_maes = [metric.predicted_baseline_angle_mae_deg for metric in route_metrics if metric.supported_window_count]
  planned_route_maes = [metric.predicted_planned_angle_mae_deg for metric in route_metrics if metric.supported_window_count]
  mean_delta = np.mean(coefficient_deltas, axis=0) if coefficient_deltas else np.full(4, np.nan)
  return FordEpsPlannerEvaluation(
    evidence_scope="virtual_eps_internal_objective_only",
    route_count=len(routes),
    routes=routes,
    requested_window_count=len(starts),
    supported_window_count=len(plans),
    rejected_window_count=len(starts) - len(plans),
    predicted_baseline_angle_mae_deg=_mean([plan.baseline_angle_mae_deg for plan, _ in plans]),
    predicted_planned_angle_mae_deg=_mean([plan.predicted_angle_mae_deg for plan, _ in plans]),
    predicted_baseline_first_angle_mae_deg=_mean([abs(plan.baseline_first_angle_error_deg) for plan, _ in plans]),
    predicted_planned_first_angle_mae_deg=_mean([abs(plan.predicted_first_angle_error_deg) for plan, _ in plans]),
    predicted_improved_fraction=_mean([
      plan.predicted_angle_mae_deg < plan.baseline_angle_mae_deg for plan, _ in plans
    ]),
    predicted_baseline_limit_fraction=_mean([plan.baseline_limit_fraction for plan, _ in plans]),
    predicted_planned_limit_fraction=_mean([plan.predicted_limit_fraction for plan, _ in plans]),
    minimum_confidence=min((plan.minimum_confidence for plan, _ in plans), default=float("nan")),
    mean_abs_coefficient_delta=(float(mean_delta[0]), float(mean_delta[1]), float(mean_delta[2]), float(mean_delta[3])),
    route_baseline_angle_mae_p90_deg=_p90(baseline_route_maes),
    route_planned_angle_mae_p90_deg=_p90(planned_route_maes),
    route_metrics=route_metrics,
  )


def _mean(values: list) -> float:
  return float(np.mean(values)) if values else float("nan")


def _p90(values: list[float]) -> float:
  return float(np.quantile(values, 0.9)) if values else float("nan")
