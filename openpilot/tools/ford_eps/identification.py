from collections.abc import Iterable
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from openpilot.tools.ford_eps.dataset import FORD_LMC2_COEFFICIENT_SCHEMA, FordEpsDataset, device_id_from_route
from openpilot.tools.ford_eps.model import HORIZON_STEPS, MODEL_TIMESTEP_S, FEATURE_NAMES, FordEpsModel, feature_vector


MIN_VALIDATION_ROUTES = 3
MIN_HORIZON_SAMPLES = 1_000
MAX_BASELINE_ERROR_RATIO = 0.85
MAX_RATE_MAE_DEG_S = 12.0
MAX_CURRENT_MAE_A = 1.0
MIN_LIMIT_PRECISION = 0.60
MIN_LIMIT_RECALL = 0.60
MIN_LIMIT_F1 = 0.65

@dataclass(frozen=True)
class AnalysisConfig:
  horizons_s: tuple[float, ...] = (0.25, 0.5, 1.0)
  ridge: float = 1e-2
  validation_fraction: float = 0.2
  validation_route: str | None = None
  device_id: str | None = None
  require_active: bool = True
  rollout_stride: int = 2


@dataclass(frozen=True)
class HorizonMetrics:
  sample_count: int
  model_angle_mae_deg: float
  constant_angle_mae_deg: float
  constant_rate_angle_mae_deg: float
  model_rate_mae_deg_s: float
  model_current_mae_a: float
  route_model_angle_mae_std_deg: float
  route_model_angle_mae_p90_deg: float


@dataclass(frozen=True)
class LimitMetrics:
  sample_count: int
  prevalence: float
  precision: float
  recall: float
  f1: float
  positive_count: int
  negative_count: int
  identifiable: bool


@dataclass(frozen=True)
class LargeTurnMetrics:
  sample_count: int
  desired_large_turn_count: int
  active_maneuver_expert_count: int
  model_angle_mae_deg: float
  constant_angle_mae_deg: float
  constant_rate_angle_mae_deg: float
  model_rate_mae_deg_s: float
  model_current_mae_a: float
  route_model_angle_mae_p90_deg: float
  in_distribution_fraction: float


@dataclass(frozen=True)
class ExcitationMetrics:
  minimum: float
  maximum: float
  standard_deviation: float
  nonzero_fraction: float
  lower_limit_fraction: float
  upper_limit_fraction: float


@dataclass(frozen=True)
class IdentificationReport:
  device_id: str
  sample_count: int
  excluded_device_sample_count: int
  transition_count: int
  train_route_count: int
  validation_route_count: int
  train_routes: tuple[str, ...]
  validation_routes: tuple[str, ...]
  horizons: dict[str, HorizonMetrics]
  direct_horizons: dict[str, HorizonMetrics]
  limit_metrics: LimitMetrics
  large_turn_metrics: LargeTurnMetrics
  feature_condition_number: float
  feature_rank: int
  feature_count: int
  response_blend: float
  normalized_angle_increment_coefficients: dict[str, float]
  coefficient_excitation: dict[str, ExcitationMetrics]
  signal_ranges: dict[str, tuple[float, float]]
  literal_motor_kt_identifiable: bool
  effective_response_identifiable: bool
  screening_ready: bool
  screening_failures: tuple[str, ...]
  large_turn_ready: bool
  large_turn_failures: tuple[str, ...]
  notes: tuple[str, ...]

  def to_dict(self) -> dict:
    return asdict(self)

  def write_json(self, path: str | Path) -> None:
    Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class IdentificationResult:
  model: "FordEpsModel"
  report: IdentificationReport


def _transition_indices(samples: np.ndarray, route_ids: set[str], require_active: bool) -> np.ndarray:
  if len(samples) < 2:
    return np.empty(0, dtype=np.int64)
  dt = (samples["mono_time_ns"][1:] - samples["mono_time_ns"][:-1]) * 1e-9
  valid = (
    (samples["segment_id"][1:] == samples["segment_id"][:-1]) &
    np.isclose(dt, 0.05, atol=1e-6) &
    np.isin(samples["route_id"][:-1], tuple(route_ids))
  )
  if require_active:
    valid &= (
      samples["lat_active"][:-1] & samples["lat_active"][1:] &
      ~samples["steering_pressed"][:-1] & ~samples["steering_pressed"][1:]
    )
  return np.flatnonzero(valid)


def _select_routes(samples: np.ndarray, config: AnalysisConfig) -> tuple[set[str], set[str]]:
  routes = sorted(set(samples["route_id"]))
  if len(routes) < 2:
    raise ValueError("at least two complete routes are required for held-out validation")
  if config.validation_route is not None:
    if config.validation_route not in routes:
      raise ValueError(f"validation route {config.validation_route!r} is not in the dataset")
    validation = {config.validation_route}
  else:
    validation_count = max(1, round(len(routes) * config.validation_fraction))
    validation = set(routes[-validation_count:])
  return set(routes) - validation, validation


def _mae(values: list[float]) -> float:
  return float(np.mean(np.abs(values))) if values else float("nan")


def _evaluate_horizon(model: FordEpsModel, samples: np.ndarray, route_ids: set[str], horizon_s: float,
                      stride: int, require_active: bool) -> HorizonMetrics:
  steps = max(round(horizon_s / 0.05), 1)
  model_errors: list[float] = []
  constant_errors: list[float] = []
  constant_rate_errors: list[float] = []
  rate_errors: list[float] = []
  current_errors: list[float] = []
  route_model_errors: dict[str, list[float]] = {}
  for start in range(0, len(samples) - steps, max(stride, 1)):
    end = start + steps
    first = samples[start]
    final = samples[end]
    if first["route_id"] not in route_ids or first["segment_id"] != final["segment_id"]:
      continue
    if require_active:
      window = samples[start:end + 1]
      if not np.all(window["lat_active"]) or np.any(window["steering_pressed"]):
        continue
    elapsed = (final["mono_time_ns"] - first["mono_time_ns"]) * 1e-9
    if not 0.7 * horizon_s <= elapsed <= 1.3 * horizon_s:
      continue
    state = np.asarray([first["pinion_angle_deg"], first["steering_rate_deg_s"], first["eps_current_a"]])
    for offset in range(steps):
      state = model.step(samples, start + offset, state)
    actual_angle = final["pinion_angle_deg"]
    model_errors.append(state[0] - actual_angle)
    route_model_errors.setdefault(str(first["route_id"]), []).append(state[0] - actual_angle)
    constant_errors.append(first["pinion_angle_deg"] - actual_angle)
    constant_rate_errors.append(first["pinion_angle_deg"] + first["steering_rate_deg_s"] * elapsed - actual_angle)
    rate_errors.append(state[1] - final["steering_rate_deg_s"])
    current_errors.append(state[2] - final["eps_current_a"])
  route_maes = np.asarray([_mae(errors) for errors in route_model_errors.values()])
  return HorizonMetrics(
    sample_count=len(model_errors),
    model_angle_mae_deg=_mae(model_errors),
    constant_angle_mae_deg=_mae(constant_errors),
    constant_rate_angle_mae_deg=_mae(constant_rate_errors),
    model_rate_mae_deg_s=_mae(rate_errors),
    model_current_mae_a=_mae(current_errors),
    route_model_angle_mae_std_deg=float(np.std(route_maes)) if len(route_maes) else float("nan"),
    route_model_angle_mae_p90_deg=float(np.quantile(route_maes, 0.9)) if len(route_maes) else float("nan"),
  )


def _evaluate_limits(model: FordEpsModel, samples: np.ndarray, route_ids: set[str], require_active: bool) -> LimitMetrics:
  selected_indices = np.flatnonzero(np.isin(samples["route_id"], tuple(route_ids)))
  if require_active:
    selected_indices = selected_indices[samples["lat_active"][selected_indices] & ~samples["steering_pressed"][selected_indices]]
  selected = samples[selected_indices]
  actual = selected["lat_limit"] > 0
  predicted = np.asarray(
    [model.limit_score(samples, int(index)) >= model.limit_threshold for index in selected_indices], dtype=np.bool_,
  )
  tp = int(np.sum(predicted & actual))
  fp = int(np.sum(predicted & ~actual))
  fn = int(np.sum(~predicted & actual))
  positive_count = int(np.sum(actual))
  negative_count = len(actual) - positive_count
  precision = tp / (tp + fp) if tp + fp else 0.0
  recall = tp / (tp + fn) if tp + fn else 0.0
  f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
  return LimitMetrics(
    sample_count=len(selected),
    prevalence=float(np.mean(actual)) if len(actual) else 0.0,
    precision=precision,
    recall=recall,
    f1=f1,
    positive_count=positive_count,
    negative_count=negative_count,
    identifiable=positive_count >= 20 and negative_count >= 20,
  )


def _evaluate_large_turns(model: FordEpsModel, samples: np.ndarray, route_ids: set[str],
                          stride: int, require_active: bool) -> LargeTurnMetrics:
  model_errors = []
  constant_errors = []
  constant_rate_errors = []
  support = []
  rate_errors = []
  current_errors = []
  desired_large_turn_count = 0
  active_maneuver_expert_count = 0
  route_errors: dict[str, list[float]] = {}
  elapsed = HORIZON_STEPS * MODEL_TIMESTEP_S
  for index in range(6, len(samples) - HORIZON_STEPS, max(stride, 1)):
    final_index = index + HORIZON_STEPS
    if samples[index]["route_id"] not in route_ids or samples[index - 6]["segment_id"] != samples[final_index]["segment_id"]:
      continue
    window = samples[index - 6:final_index + 1]
    if require_active and (not np.all(window["lat_active"]) or np.any(window["steering_pressed"])):
      continue
    state = np.asarray([
      samples[index]["pinion_angle_deg"], samples[index]["steering_rate_deg_s"], samples[index]["eps_current_a"],
    ])
    predicted_states, _, in_distribution, maneuver_share = model.predict_recorded_horizon(samples, index, state)
    desired_large_turn = np.max(np.abs(samples[index + 1:final_index + 1]["desired_angle_deg"])) >= 80.0
    active_maneuver_expert = maneuver_share >= 0.5
    if not desired_large_turn and not active_maneuver_expert:
      continue
    desired_large_turn_count += int(desired_large_turn)
    active_maneuver_expert_count += int(active_maneuver_expert)
    predicted = predicted_states[-1]
    actual_angle = float(samples[final_index]["pinion_angle_deg"])
    model_error = float(predicted[0] - actual_angle)
    model_errors.append(model_error)
    constant_errors.append(float(samples[index]["pinion_angle_deg"] - actual_angle))
    constant_rate_errors.append(float(
      samples[index]["pinion_angle_deg"] + samples[index]["steering_rate_deg_s"] * elapsed - actual_angle,
    ))
    rate_errors.append(float(predicted[1] - samples[final_index]["steering_rate_deg_s"]))
    current_errors.append(float(predicted[2] - samples[final_index]["eps_current_a"]))
    support.append(in_distribution)
    route_errors.setdefault(str(samples[index]["route_id"]), []).append(model_error)

  route_maes = [_mae(errors) for errors in route_errors.values()]
  return LargeTurnMetrics(
    sample_count=len(model_errors),
    desired_large_turn_count=desired_large_turn_count,
    active_maneuver_expert_count=active_maneuver_expert_count,
    model_angle_mae_deg=_mae(model_errors),
    constant_angle_mae_deg=_mae(constant_errors),
    constant_rate_angle_mae_deg=_mae(constant_rate_errors),
    model_rate_mae_deg_s=_mae(rate_errors),
    model_current_mae_a=_mae(current_errors),
    route_model_angle_mae_p90_deg=float(np.quantile(route_maes, 0.9)) if route_maes else float("nan"),
    in_distribution_fraction=float(np.mean(support)) if support else 0.0,
  )


def _evaluate_direct_horizons(model: FordEpsModel, samples: np.ndarray, route_ids: set[str],
                              stride: int, require_active: bool) -> dict[str, HorizonMetrics]:
  model_errors = [[] for _ in range(HORIZON_STEPS)]
  constant_errors = [[] for _ in range(HORIZON_STEPS)]
  constant_rate_errors = [[] for _ in range(HORIZON_STEPS)]
  rate_errors = [[] for _ in range(HORIZON_STEPS)]
  current_errors = [[] for _ in range(HORIZON_STEPS)]
  route_errors: list[dict[str, list[float]]] = [{} for _ in range(HORIZON_STEPS)]

  for index in range(6, len(samples) - HORIZON_STEPS, max(stride, 1)):
    final_index = index + HORIZON_STEPS
    if samples[index]["route_id"] not in route_ids or samples[index - 6]["segment_id"] != samples[final_index]["segment_id"]:
      continue
    window = samples[index - 6:final_index + 1]
    if require_active and (not np.all(window["lat_active"]) or np.any(window["steering_pressed"])):
      continue
    state = np.asarray([
      samples[index]["pinion_angle_deg"], samples[index]["steering_rate_deg_s"], samples[index]["eps_current_a"],
    ])
    predicted_states, _, _, _ = model.predict_recorded_horizon(samples, index, state)
    route_id = str(samples[index]["route_id"])
    for horizon_index, predicted in enumerate(predicted_states):
      actual = samples[index + horizon_index + 1]
      elapsed = (horizon_index + 1) * MODEL_TIMESTEP_S
      model_error = float(predicted[0] - actual["pinion_angle_deg"])
      model_errors[horizon_index].append(model_error)
      constant_errors[horizon_index].append(float(samples[index]["pinion_angle_deg"] - actual["pinion_angle_deg"]))
      constant_rate_errors[horizon_index].append(float(
        samples[index]["pinion_angle_deg"] + samples[index]["steering_rate_deg_s"] * elapsed - actual["pinion_angle_deg"],
      ))
      rate_errors[horizon_index].append(float(predicted[1] - actual["steering_rate_deg_s"]))
      current_errors[horizon_index].append(float(predicted[2] - actual["eps_current_a"]))
      route_errors[horizon_index].setdefault(route_id, []).append(model_error)

  metrics = {}
  for horizon_index in range(HORIZON_STEPS):
    route_maes = np.asarray([_mae(errors) for errors in route_errors[horizon_index].values()])
    metrics[f"{(horizon_index + 1) * MODEL_TIMESTEP_S:.2f}"] = HorizonMetrics(
      sample_count=len(model_errors[horizon_index]),
      model_angle_mae_deg=_mae(model_errors[horizon_index]),
      constant_angle_mae_deg=_mae(constant_errors[horizon_index]),
      constant_rate_angle_mae_deg=_mae(constant_rate_errors[horizon_index]),
      model_rate_mae_deg_s=_mae(rate_errors[horizon_index]),
      model_current_mae_a=_mae(current_errors[horizon_index]),
      route_model_angle_mae_std_deg=float(np.std(route_maes)) if len(route_maes) else float("nan"),
      route_model_angle_mae_p90_deg=float(np.quantile(route_maes, 0.9)) if len(route_maes) else float("nan"),
    )
  return metrics


def _large_turn_failures(metrics: LargeTurnMetrics) -> tuple[str, ...]:
  failures = []
  if metrics.sample_count < 250:
    failures.append("requires at least 250 held-out large-turn windows")
  if metrics.active_maneuver_expert_count < 100:
    failures.append("requires at least 100 held-out active maneuver-expert windows")
  if metrics.model_angle_mae_deg > 0.95 * min(metrics.constant_angle_mae_deg, metrics.constant_rate_angle_mae_deg):
    failures.append("large-turn angle MAE lacks a 5% margin over both baselines")
  if metrics.model_angle_mae_deg > 8.0:
    failures.append("large-turn angle MAE exceeds 8.0 deg at 250 ms")
  if metrics.route_model_angle_mae_p90_deg > 10.0:
    failures.append("large-turn route-p90 angle MAE exceeds 10.0 deg")
  if metrics.model_rate_mae_deg_s > 60.0:
    failures.append("large-turn rate MAE exceeds 60.0 deg/s")
  if metrics.model_current_mae_a > 1.5:
    failures.append("large-turn current MAE exceeds 1.5 A")
  if metrics.in_distribution_fraction < 0.5:
    failures.append("fewer than half of held-out large turns are inside identified support")
  return tuple(failures)


def _direct_screening_failures(horizons: dict[str, HorizonMetrics]) -> tuple[str, ...]:
  """Require the predictor used by the inverse planner to preserve ordinary driving."""
  failures = []
  for horizon_text, metrics in horizons.items():
    horizon = float(horizon_text)
    baseline = min(metrics.constant_angle_mae_deg, metrics.constant_rate_angle_mae_deg)
    max_angle_mae = max(0.75, 6.0 * horizon)
    max_route_p90 = max(1.0, 8.0 * horizon)
    if metrics.sample_count < MIN_HORIZON_SAMPLES:
      failures.append(f"direct {horizon_text}s has fewer than {MIN_HORIZON_SAMPLES} validation windows")
    if metrics.model_angle_mae_deg > 1.05 * baseline:
      failures.append(f"direct {horizon_text}s angle MAE is more than 5% worse than baseline")
    if metrics.model_angle_mae_deg > max_angle_mae:
      failures.append(f"direct {horizon_text}s angle MAE exceeds {max_angle_mae:.2f} deg")
    if metrics.route_model_angle_mae_p90_deg > max_route_p90:
      failures.append(f"direct {horizon_text}s route-p90 angle MAE exceeds {max_route_p90:.2f} deg")
    if metrics.model_rate_mae_deg_s > 25.0:
      failures.append(f"direct {horizon_text}s rate MAE exceeds 25.0 deg/s")
    if metrics.model_current_mae_a > MAX_CURRENT_MAE_A:
      failures.append(f"direct {horizon_text}s current MAE exceeds {MAX_CURRENT_MAE_A:.1f} A")
  return tuple(failures)


def _calibrate_response_blend(samples: np.ndarray, train_routes: set[str], config: AnalysisConfig) -> float:
  """Tune how strongly the learned PSCM response corrects an inertial steering baseline."""
  routes = sorted(train_routes)
  if len(routes) < 3:
    calibration_routes = train_routes
    fit_routes = train_routes
  else:
    calibration_count = max(1, round(len(routes) * 0.2))
    calibration_routes = set(routes[-calibration_count:])
    fit_routes = set(routes[:-calibration_count])
  fit_indices = _transition_indices(samples, fit_routes, config.require_active)
  if len(fit_indices) < len(FEATURE_NAMES) * 2:
    return 1.0
  probe = FordEpsModel(config.ridge)
  probe.fit(samples, fit_indices)
  best_blend = 1.0
  best_score = float("inf")
  for blend in np.linspace(0.0, 1.0, 11):
    probe.response_blend = float(blend)
    metrics = [
      _evaluate_horizon(probe, samples, calibration_routes, horizon, max(config.rollout_stride, 8), config.require_active)
      for horizon in (0.25, 0.5)
    ]
    score = float(np.mean([
      metric.model_angle_mae_deg / max(metric.constant_rate_angle_mae_deg, 1e-3)
      for metric in metrics if metric.sample_count
    ]))
    if score < best_score:
      best_score = score
      best_blend = float(blend)
  return best_blend


def _screening_failures(horizons: dict[str, HorizonMetrics], limit: LimitMetrics,
                        validation_route_count: int) -> tuple[str, ...]:
  failures = []
  if validation_route_count < MIN_VALIDATION_ROUTES:
    failures.append(f"requires at least {MIN_VALIDATION_ROUTES} held-out routes")
  for horizon_text, metrics in horizons.items():
    horizon = float(horizon_text)
    baseline = min(metrics.constant_angle_mae_deg, metrics.constant_rate_angle_mae_deg)
    max_angle_mae = max(1.5, 6.0 * horizon)
    max_route_p90 = max(2.0, 8.0 * horizon)
    if metrics.sample_count < MIN_HORIZON_SAMPLES:
      failures.append(f"{horizon_text}s has fewer than {MIN_HORIZON_SAMPLES} validation windows")
    if metrics.model_angle_mae_deg > MAX_BASELINE_ERROR_RATIO * baseline:
      failures.append(f"{horizon_text}s angle MAE lacks a 15% baseline margin")
    if metrics.model_angle_mae_deg > max_angle_mae:
      failures.append(f"{horizon_text}s angle MAE exceeds {max_angle_mae:.1f} deg")
    if metrics.route_model_angle_mae_p90_deg > max_route_p90:
      failures.append(f"{horizon_text}s route-p90 angle MAE exceeds {max_route_p90:.1f} deg")
    if metrics.model_rate_mae_deg_s > MAX_RATE_MAE_DEG_S:
      failures.append(f"{horizon_text}s rate MAE exceeds {MAX_RATE_MAE_DEG_S:.1f} deg/s")
    if metrics.model_current_mae_a > MAX_CURRENT_MAE_A:
      failures.append(f"{horizon_text}s current MAE exceeds {MAX_CURRENT_MAE_A:.1f} A")
  if not limit.identifiable:
    failures.append("PSCM limit response lacks positive/negative validation coverage")
  if limit.precision < MIN_LIMIT_PRECISION or limit.recall < MIN_LIMIT_RECALL or limit.f1 < MIN_LIMIT_F1:
    failures.append("PSCM limit classifier misses precision/recall/F1 thresholds")
  return tuple(failures)


def fit(source: Iterable[str | Path] | FordEpsDataset, config: AnalysisConfig | None = None) -> IdentificationResult:
  """Fit a replayable equivalent Ford EPS and evaluate it on complete held-out routes."""
  config = AnalysisConfig() if config is None else config
  dataset = source if isinstance(source, FordEpsDataset) else FordEpsDataset.from_rlogs(source)
  source_sample_count = len(dataset.samples)
  device_ids = sorted({device_id_from_route(str(route_id)) for route_id in dataset.samples["route_id"]})
  if config.device_id is None:
    if len(device_ids) != 1:
      raise ValueError(f"dataset spans multiple devices {device_ids}; select one with AnalysisConfig.device_id")
    device_id = device_ids[0]
  else:
    device_id = config.device_id
    if device_id not in device_ids:
      raise ValueError(f"device {device_id!r} is not in the dataset; available devices: {device_ids}")
  device_mask = np.asarray(
    [device_id_from_route(str(route_id)) == device_id for route_id in dataset.samples["route_id"]], dtype=np.bool_,
  )
  samples = dataset.samples[device_mask]
  train_routes, validation_routes = _select_routes(samples, config)
  train_indices = _transition_indices(samples, train_routes, config.require_active)
  if len(train_indices) < len(FEATURE_NAMES) * 2:
    raise ValueError(f"not enough training transitions: {len(train_indices)}")

  response_blend = _calibrate_response_blend(samples, train_routes, config)
  model = FordEpsModel(config.ridge)
  model.fit(samples, train_indices)
  model.response_blend = response_blend
  raw_features = np.stack([feature_vector(samples, int(index)) for index in train_indices])
  normalized_features = (raw_features - model.dynamics.mean) / model.dynamics.scale
  singular_values = np.linalg.svd(normalized_features, compute_uv=False)
  tolerance = singular_values[0] * max(normalized_features.shape) * np.finfo(np.float64).eps
  identifiable_singular_values = singular_values[singular_values > tolerance]
  condition = float(identifiable_singular_values[0] / identifiable_singular_values[-1])
  rank = len(identifiable_singular_values)
  angle_increment_coefficients = {
    name: float(coefficient)
    for name, coefficient in zip(FEATURE_NAMES, model.dynamics.coefficients[:, 0], strict=True)
  }
  horizons = {
    f"{horizon:.2f}": _evaluate_horizon(
      model, samples, validation_routes, horizon, config.rollout_stride, config.require_active,
    )
    for horizon in config.horizons_s
  }
  direct_horizons = _evaluate_direct_horizons(
    model, samples, validation_routes, config.rollout_stride, config.require_active,
  )
  short_horizons = [metrics for horizon, metrics in horizons.items() if float(horizon) <= 0.5]
  effective_response_identifiable = bool(short_horizons) and all(
    metrics.model_angle_mae_deg < min(metrics.constant_angle_mae_deg, metrics.constant_rate_angle_mae_deg)
    for metrics in short_horizons
  )
  range_fields = (
    "c0", "c1", "c2", "c3", "pinion_angle_deg", "steering_rate_deg_s", "speed_mps",
    "eps_current_a", "eps_voltage_v", "column_torque_nm", "yaw_rate_rad_s", "lateral_accel_mps2",
  )
  signal_ranges = {
    field: (float(np.min(samples[field])), float(np.max(samples[field])))
    for field in range_fields
  }
  excitation_samples = samples[samples["lat_active"] & ~samples["steering_pressed"]]
  coefficient_excitation = {}
  for field, (lower, upper, resolution) in FORD_LMC2_COEFFICIENT_SCHEMA.items():
    values = excitation_samples[field]
    coefficient_excitation[field] = ExcitationMetrics(
      minimum=float(np.min(values)),
      maximum=float(np.max(values)),
      standard_deviation=float(np.std(values)),
      nonzero_fraction=float(np.mean(np.abs(values) >= resolution)),
      lower_limit_fraction=float(np.mean(values <= lower + resolution)),
      upper_limit_fraction=float(np.mean(values >= upper - resolution)),
    )
  limit_metrics = _evaluate_limits(model, samples, validation_routes, config.require_active)
  large_turn_metrics = _evaluate_large_turns(
    model, samples, validation_routes, config.rollout_stride, config.require_active,
  )
  large_turn_failures = _large_turn_failures(large_turn_metrics)
  large_turn_ready = not large_turn_failures
  screening_failures = _screening_failures(horizons, limit_metrics, len(validation_routes))
  screening_failures = (*screening_failures, *_direct_screening_failures(direct_horizons))
  if not effective_response_identifiable:
    screening_failures = (*screening_failures, "effective command response is not identifiable")
  screening_ready = not screening_failures
  model.screening_ready = screening_ready
  model.large_turn_ready = large_turn_ready
  report = IdentificationReport(
    device_id=device_id,
    sample_count=len(samples),
    excluded_device_sample_count=source_sample_count - len(samples),
    transition_count=len(train_indices),
    train_route_count=len(train_routes),
    validation_route_count=len(validation_routes),
    train_routes=tuple(sorted(train_routes)),
    validation_routes=tuple(sorted(validation_routes)),
    horizons=horizons,
    direct_horizons=direct_horizons,
    limit_metrics=limit_metrics,
    large_turn_metrics=large_turn_metrics,
    feature_condition_number=condition,
    feature_rank=rank,
    feature_count=len(FEATURE_NAMES),
    response_blend=response_blend,
    normalized_angle_increment_coefficients=angle_increment_coefficients,
    coefficient_excitation=coefficient_excitation,
    signal_ranges=signal_ranges,
    literal_motor_kt_identifiable=False,
    effective_response_identifiable=effective_response_identifiable,
    screening_ready=screening_ready,
    screening_failures=screening_failures,
    large_turn_ready=large_turn_ready,
    large_turn_failures=large_turn_failures,
    notes=(
      "SteMdule_I_Est is an estimated module-current signal, not a measured motor phase-current/torque pair.",
      "The fitted gain is a closed-loop PSCM/EPS equivalent and cannot uniquely separate motor kT, gearing, efficiency, and road load.",
      "Validation holds out complete routes from the selected device; other devices are excluded rather than averaged into one plant.",
    ),
  )
  return IdentificationResult(model=model, report=report)


def identify(source: Iterable[str | Path] | FordEpsDataset, config: AnalysisConfig | None = None) -> IdentificationReport:
  """Convenience wrapper returning only the held-out identification report."""
  return fit(source, config).report
