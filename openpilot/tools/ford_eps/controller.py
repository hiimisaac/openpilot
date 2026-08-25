from dataclasses import dataclass, replace

import numpy as np

from openpilot.tools.ford_eps.dataset import FORD_LMC2_COEFFICIENT_SCHEMA, FordEpsInput, FordEpsOutput
from openpilot.tools.ford_eps.model import COMMAND_HISTORY_LENGTH, FEATURE_NAMES, MODEL_TIMESTEP_S, FordEpsModel


COEFFICIENT_NAMES = tuple(FORD_LMC2_COEFFICIENT_SCHEMA)
COEFFICIENT_LIMITS = np.asarray([
  FORD_LMC2_COEFFICIENT_SCHEMA[name][:2] for name in COEFFICIENT_NAMES
])
COEFFICIENT_RESOLUTIONS = np.asarray([
  FORD_LMC2_COEFFICIENT_SCHEMA[name][2] for name in COEFFICIENT_NAMES
])
COEFFICIENT_FEATURE_INDICES = np.asarray([FEATURE_NAMES.index(name) for name in COEFFICIENT_NAMES], dtype=np.int64)
@dataclass(frozen=True)
class FordEpsPlanRequest:
  pinion_angle_deg: float
  steering_rate_deg_s: float
  eps_current_a: float
  command_history: tuple[FordEpsInput, ...]
  baseline_commands: tuple[FordEpsInput, ...]
  desired_angles_deg: tuple[float, ...]


@dataclass(frozen=True)
class FordEpsPlan:
  commands: tuple[FordEpsInput, ...]
  baseline_outputs: tuple[FordEpsOutput, ...]
  predicted_outputs: tuple[FordEpsOutput, ...]
  baseline_angle_mae_deg: float
  predicted_angle_mae_deg: float
  baseline_first_angle_error_deg: float
  predicted_first_angle_error_deg: float
  baseline_limit_fraction: float
  predicted_limit_fraction: float
  minimum_confidence: float
  in_distribution: bool


@dataclass(frozen=True)
class FordEpsPlannerConfig:
  optimization_iterations: int = 5
  allow_c2_adjustment: bool = False
  angle_rate_weight: float = 0.002
  command_weight: float = 0.002
  c2_change_weight: float = 0.04
  limit_weight: float = 100.0


class FordEpsCommandPlanner:
  """Physics-constrained inverse controller around a validated virtual Ford EPS."""

  def __init__(self, model: FordEpsModel, config: FordEpsPlannerConfig | None = None):
    if not model.screening_ready:
      raise ValueError("virtual EPS did not pass held-out validation")
    self.model = model
    self.config = FordEpsPlannerConfig() if config is None else config

  def plan(self, request: FordEpsPlanRequest) -> FordEpsPlan:
    self._validate_request(request)
    if max(abs(angle) for angle in request.desired_angles_deg) >= 80.0 and not self.model.large_turn_ready:
      raise ValueError("virtual EPS did not pass held-out large-turn validation")
    active = np.asarray((0, 1, 2, 3) if self.config.allow_c2_adjustment else (0, 1, 3), dtype=np.int64)
    correction = np.zeros((2, 4), dtype=np.float64)
    baseline_outputs, baseline_cost = self._evaluate(request, correction)
    best_outputs = baseline_outputs
    best_cost = baseline_cost
    modes = self._coupled_modes(active)
    temporal_profiles = np.asarray(((1.0, 1.0), (1.0, 0.0), (0.0, 1.0)))
    scale = 0.75

    for _ in range(self.config.optimization_iterations):
      for mode in modes:
        for profile in temporal_profiles:
          for direction in (-1.0, 1.0):
            candidate = correction + direction * scale * profile[:, None] * mode[None, :]
            try:
              candidate_outputs, candidate_cost = self._evaluate(request, candidate)
            except ValueError:
              continue
            if any(
              (candidate_output.limit_predicted and not baseline_output.limit_predicted) or
              (baseline_output.limit_predicted and candidate_output.limit_score > baseline_output.limit_score)
              for baseline_output, candidate_output in zip(baseline_outputs, candidate_outputs, strict=True)
            ):
              continue
            if candidate_cost < best_cost:
              correction = candidate
              best_outputs = candidate_outputs
              best_cost = candidate_cost
      scale *= 0.5

    commands = self._commands(request.baseline_commands, correction)
    baseline_errors = self._angle_errors(baseline_outputs, request.desired_angles_deg)
    predicted_errors = self._angle_errors(best_outputs, request.desired_angles_deg)
    return FordEpsPlan(
      commands=commands,
      baseline_outputs=baseline_outputs,
      predicted_outputs=best_outputs,
      baseline_angle_mae_deg=float(np.mean(np.abs(baseline_errors))),
      predicted_angle_mae_deg=float(np.mean(np.abs(predicted_errors))),
      baseline_first_angle_error_deg=float(baseline_errors[0]),
      predicted_first_angle_error_deg=float(predicted_errors[0]),
      baseline_limit_fraction=float(np.mean([output.limit_predicted for output in baseline_outputs])),
      predicted_limit_fraction=float(np.mean([output.limit_predicted for output in best_outputs])),
      minimum_confidence=min(output.confidence for output in best_outputs),
      in_distribution=all(output.in_distribution for output in best_outputs),
    )

  def _coupled_modes(self, active: np.ndarray) -> np.ndarray:
    """Observed joint-command axes, expressed in physical coefficient units."""
    feature_indices = COEFFICIENT_FEATURE_INDICES[active]
    covariance = np.linalg.inv(self.model.feature_precision)
    coefficient_covariance = covariance[np.ix_(feature_indices, feature_indices)]
    _, eigenvectors = np.linalg.eigh(coefficient_covariance)
    physical_scale = self.model.dynamics.scale[feature_indices]
    modes = []
    for vector in eigenvectors.T:
      mode = np.zeros(4, dtype=np.float64)
      physical = vector * physical_scale
      normalizer = np.sqrt(np.sum((physical / np.maximum(physical_scale, COEFFICIENT_RESOLUTIONS[active])) ** 2))
      mode[active] = physical / max(normalizer, 1e-12)
      modes.append(mode)
    return np.asarray(modes)

  @staticmethod
  def _validate_request(request: FordEpsPlanRequest) -> None:
    if len(request.command_history) != COMMAND_HISTORY_LENGTH:
      raise ValueError("exactly seven oldest-to-newest 50 ms command-history samples are required")
    if not request.baseline_commands:
      raise ValueError("at least one future baseline command is required")
    if len(request.baseline_commands) != len(request.desired_angles_deg):
      raise ValueError("baseline command and desired-angle horizons must match")

  @staticmethod
  def _quantize(coefficients: np.ndarray) -> np.ndarray:
    clipped = np.clip(coefficients, COEFFICIENT_LIMITS[:, 0], COEFFICIENT_LIMITS[:, 1])
    steps = np.rint((clipped - COEFFICIENT_LIMITS[:, 0]) / COEFFICIENT_RESOLUTIONS)
    return np.clip(
      COEFFICIENT_LIMITS[:, 0] + steps * COEFFICIENT_RESOLUTIONS,
      COEFFICIENT_LIMITS[:, 0], COEFFICIENT_LIMITS[:, 1],
    )

  @classmethod
  def _commands(cls, baseline: tuple[FordEpsInput, ...], correction: np.ndarray) -> tuple[FordEpsInput, ...]:
    horizon = np.linspace(0.0, 1.0, len(baseline))
    deltas = correction[0][None, :] + horizon[:, None] * (correction[1] - correction[0])[None, :]
    commands = []
    for command, delta in zip(baseline, deltas, strict=True):
      coefficients = cls._quantize(np.asarray([command.c0, command.c1, command.c2, command.c3]) + delta)
      commands.append(replace(
        command, c0=float(coefficients[0]), c1=float(coefficients[1]),
        c2=float(coefficients[2]), c3=float(coefficients[3]),
      ))
    return tuple(commands)

  @staticmethod
  def _angle_errors(outputs: tuple[FordEpsOutput, ...], desired_angles: tuple[float, ...]) -> np.ndarray:
    return np.asarray([output.pinion_angle_deg for output in outputs]) - np.asarray(desired_angles)

  def _evaluate(self, request: FordEpsPlanRequest,
                correction: np.ndarray) -> tuple[tuple[FordEpsOutput, ...], float]:
    commands = self._commands(request.baseline_commands, correction)
    outputs = self.model.predict_horizon(
      request.pinion_angle_deg,
      request.steering_rate_deg_s,
      request.eps_current_a,
      request.command_history,
      commands,
    )
    predicted_angles = np.asarray([output.pinion_angle_deg for output in outputs])
    predicted_rates = np.asarray([output.steering_rate_deg_s for output in outputs])
    desired_angles = np.asarray(request.desired_angles_deg)
    desired_rates = np.diff(np.concatenate(([request.pinion_angle_deg], desired_angles))) / MODEL_TIMESTEP_S
    horizon_weights = np.linspace(1.0, 2.0, len(outputs))
    angle_cost = float(np.mean(horizon_weights * (predicted_angles - desired_angles) ** 2))
    rate_cost = self.config.angle_rate_weight * float(np.mean((predicted_rates - desired_rates) ** 2))
    normalized_correction = correction / self.model.dynamics.scale[COEFFICIENT_FEATURE_INDICES][None, :]
    command_cost = self.config.command_weight * float(np.mean(normalized_correction ** 2))
    c2_cost = self.config.c2_change_weight * float(np.mean(normalized_correction[:, 2] ** 2))
    limit_excess = np.maximum(
      np.asarray([output.limit_score for output in outputs]) - self.model.limit_threshold,
      0.0,
    )
    limit_cost = self.config.limit_weight * float(limit_excess @ limit_excess)
    return outputs, angle_cost + rate_cost + command_cost + c2_cost + limit_cost
