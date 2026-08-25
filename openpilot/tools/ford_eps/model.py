from collections import deque
from pathlib import Path

import numpy as np

from openpilot.tools.ford_eps.dataset import (
  SAMPLE_DTYPE,
  FordEpsDataset,
  FordEpsInput,
  FordEpsOutput,
  sample_from_input,
)


VIRTUAL_EPS_VERSION = 6
MODEL_TIMESTEP_S = 0.05
SPEED_EDGES_MPS = np.asarray([5.0, 10.0, 18.0, 27.0])

FEATURE_NAMES = (
  "angle",
  "angle_rate",
  "eps_current",
  "column_torque",
  "driver_torque",
  "speed",
  "yaw_rate",
  "lateral_accel",
  "longitudinal_accel",
  "voltage_delta",
  "c0",
  "c1",
  "c2",
  "c3",
  "path_curvature_3m",
  "path_curvature_5m",
  "path_curvature_7m",
  "path_curvature_10m",
  "path_curvature_7m_x_speed",
  "path_curvature_7m_x_speed_sq",
  "rate_sign",
  "abs_angle",
  "abs_angle_rate",
  "abs_eps_current",
  "abs_path_curvature_7m",
  "abs_c0",
  "abs_c1",
  "abs_c2",
  "abs_c3",
  "c0_lag_50ms",
  "c1_lag_50ms",
  "c2_lag_50ms",
  "c3_lag_50ms",
  "path_curvature_7m_lag_50ms",
  "c0_lag_100ms",
  "c1_lag_100ms",
  "c2_lag_100ms",
  "c3_lag_100ms",
  "path_curvature_7m_lag_100ms",
  "c0_lag_200ms",
  "c1_lag_200ms",
  "c2_lag_200ms",
  "c3_lag_200ms",
  "path_curvature_7m_lag_200ms",
  "c0_lag_300ms",
  "c1_lag_300ms",
  "c2_lag_300ms",
  "c3_lag_300ms",
  "path_curvature_7m_lag_300ms",
  "lat_active",
)


def _path_equivalent_curvature(sample, distance: float) -> float:
  offset = sample["c0"] + sample["c1"] * distance + 0.5 * sample["c2"] * distance ** 2 + sample["c3"] * distance ** 3 / 6.0
  return 2.0 * offset / distance ** 2


def _command_features(sample) -> tuple[float, float, float, float, float]:
  return sample["c0"], sample["c1"], sample["c2"], sample["c3"], _path_equivalent_curvature(sample, 7.0)


def feature_vector(samples: np.ndarray, index: int, state: np.ndarray | None = None) -> np.ndarray:
  sample = samples[index]
  angle, rate, current = state if state is not None else (
    sample["pinion_angle_deg"],
    sample["steering_rate_deg_s"],
    sample["eps_current_a"],
  )
  k3 = _path_equivalent_curvature(sample, 3.0)
  k5 = _path_equivalent_curvature(sample, 5.0)
  k7 = _path_equivalent_curvature(sample, 7.0)
  k10 = _path_equivalent_curvature(sample, 10.0)
  speed = sample["speed_mps"]
  lagged_commands = []
  for lag in (1, 2, 4, 6):
    lag_index = max(index - lag, 0)
    if samples[lag_index]["segment_id"] != sample["segment_id"]:
      lag_index = index
    lagged_commands.extend(_command_features(samples[lag_index]))
  return np.asarray([
    angle,
    rate,
    current,
    sample["column_torque_nm"],
    sample["driver_torque_nm"],
    speed,
    sample["yaw_rate_rad_s"],
    sample["lateral_accel_mps2"],
    sample["longitudinal_accel_mps2"],
    sample["eps_voltage_v"] - 14.0,
    sample["c0"],
    sample["c1"],
    sample["c2"],
    sample["c3"],
    k3,
    k5,
    k7,
    k10,
    k7 * speed,
    k7 * speed ** 2,
    np.sign(rate),
    abs(angle),
    abs(rate),
    abs(current),
    abs(k7),
    abs(sample["c0"]),
    abs(sample["c1"]),
    abs(sample["c2"]),
    abs(sample["c3"]),
    *lagged_commands,
    float(sample["lat_active"]),
  ], dtype=np.float64)


class _Ridge:
  def __init__(self, ridge: float):
    self.ridge = ridge
    self.mean = np.empty(0)
    self.scale = np.empty(0)
    self.target_mean = np.empty(0)
    self.coefficients = np.empty((0, 0))

  def fit(self, features: np.ndarray, targets: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
    self.mean = np.mean(features, axis=0)
    self.scale = np.std(features, axis=0)
    self.scale[self.scale < 1e-9] = 1.0
    normalized = (features - self.mean) / self.scale
    self.target_mean = np.mean(targets, axis=0)
    centered_targets = targets - self.target_mean
    if sample_weight is not None:
      root_weight = np.sqrt(sample_weight)[:, None]
      normalized = normalized * root_weight
      centered_targets = centered_targets * root_weight
    gram = normalized.T @ normalized
    gram.flat[::len(gram) + 1] += self.ridge
    try:
      self.coefficients = np.linalg.solve(gram, normalized.T @ centered_targets)
    except np.linalg.LinAlgError:
      self.coefficients = np.linalg.lstsq(gram, normalized.T @ centered_targets, rcond=None)[0]

  def predict(self, features: np.ndarray) -> np.ndarray:
    return (features - self.mean) / self.scale @ self.coefficients + self.target_mean

  def copy(self) -> "_Ridge":
    copied = _Ridge(self.ridge)
    copied.mean = self.mean.copy()
    copied.scale = self.scale.copy()
    copied.target_mean = self.target_mean.copy()
    copied.coefficients = self.coefficients.copy()
    return copied


class FordEpsModel:
  def __init__(self, ridge: float):
    self.dynamics = _Ridge(ridge)
    self.speed_dynamics: list[_Ridge] = []
    self.limit = _Ridge(ridge)
    self.limit_threshold = 0.5
    self.response_blend = 1.0
    self.feature_low = np.full(len(FEATURE_NAMES), -np.inf)
    self.feature_high = np.full(len(FEATURE_NAMES), np.inf)
    self.feature_precision = np.eye(len(FEATURE_NAMES))
    self.joint_support_threshold = np.inf
    self.screening_ready = False
    self.state_min = np.asarray([-1600.0, -1000.0, -64.0])
    self.state_max = np.asarray([1676.7, 1000.0, 140.7])

  def fit(self, samples: np.ndarray, transition_indices: np.ndarray) -> None:
    current = samples[transition_indices]
    following = samples[transition_indices + 1]
    features = np.stack([feature_vector(samples, int(index)) for index in transition_indices])
    transition_dt = (following["mono_time_ns"] - current["mono_time_ns"]) * 1e-9
    normalized_angle_increment = (
      following["pinion_angle_deg"] - current["pinion_angle_deg"]
    ) * MODEL_TIMESTEP_S / transition_dt
    targets = np.column_stack([
      normalized_angle_increment,
      following["eps_current_a"],
    ])
    self.dynamics.fit(features, targets)
    self.speed_dynamics = []
    speed = current["speed_mps"]
    speed_bins = np.searchsorted(SPEED_EDGES_MPS, speed, side="right")
    for speed_bin in range(len(SPEED_EDGES_MPS) + 1):
      selected = speed_bins == speed_bin
      if np.sum(selected) >= max(len(FEATURE_NAMES) * 10, 1_000):
        expert = _Ridge(self.dynamics.ridge)
        expert.fit(features[selected], targets[selected])
      else:
        expert = self.dynamics.copy()
      self.speed_dynamics.append(expert)
    self.feature_low = np.quantile(features, 0.005, axis=0)
    self.feature_high = np.quantile(features, 0.995, axis=0)
    normalized_features = (features - self.dynamics.mean) / self.dynamics.scale
    covariance = normalized_features.T @ normalized_features / len(normalized_features)
    covariance.flat[::len(covariance) + 1] += 0.05
    self.feature_precision = np.linalg.inv(covariance)
    distances = np.sqrt(np.einsum(
      "ij,jk,ik->i", normalized_features, self.feature_precision, normalized_features,
    ))
    self.joint_support_threshold = float(np.quantile(distances, 0.995))

    limit_target = (current["lat_limit"] > 0).astype(np.float64)[:, None]
    positives = max(float(np.sum(limit_target)), 1.0)
    negatives = max(float(len(limit_target) - positives), 1.0)
    weights = np.where(limit_target[:, 0] > 0.0, len(limit_target) / (2.0 * positives), len(limit_target) / (2.0 * negatives))
    self.limit.fit(features, limit_target, weights)
    limit_scores = self.limit.predict(features)[:, 0]
    candidates = np.unique(np.quantile(limit_scores, np.linspace(0.0, 1.0, 201)))
    best_f1 = -1.0
    for threshold in candidates:
      predicted = limit_scores >= threshold
      actual = limit_target[:, 0] > 0.0
      tp = np.sum(predicted & actual)
      fp = np.sum(predicted & ~actual)
      fn = np.sum(~predicted & actual)
      precision = tp / (tp + fp) if tp + fp else 0.0
      recall = tp / (tp + fn) if tp + fn else 0.0
      f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
      if f1 > best_f1:
        best_f1 = f1
        self.limit_threshold = float(threshold)

    states = np.column_stack([
      current["pinion_angle_deg"],
      current["steering_rate_deg_s"],
      current["eps_current_a"],
    ])
    self.state_min = np.maximum(self.state_min, np.quantile(states, 0.001, axis=0))
    self.state_max = np.minimum(self.state_max, np.quantile(states, 0.999, axis=0))

  def step(self, samples: np.ndarray, index: int, state: np.ndarray, dt: float = MODEL_TIMESTEP_S) -> np.ndarray:
    if not np.isclose(dt, MODEL_TIMESTEP_S):
      raise ValueError("virtual EPS timestep must be exactly 50 ms")
    features = feature_vector(samples, index, state)
    speed = float(samples[index]["speed_mps"])
    speed_bin = int(np.searchsorted(SPEED_EDGES_MPS, speed, side="right"))
    learned = self.speed_dynamics[speed_bin].predict(features)
    transition_width = 1.0
    if speed_bin > 0 and speed - SPEED_EDGES_MPS[speed_bin - 1] < transition_width:
      blend = max((speed - SPEED_EDGES_MPS[speed_bin - 1]) / transition_width, 0.0)
      learned = (1.0 - blend) * self.speed_dynamics[speed_bin - 1].predict(features) + blend * learned
    elif speed_bin < len(SPEED_EDGES_MPS) and SPEED_EDGES_MPS[speed_bin] - speed < transition_width:
      blend = max((SPEED_EDGES_MPS[speed_bin] - speed) / transition_width, 0.0)
      learned = blend * learned + (1.0 - blend) * self.speed_dynamics[speed_bin + 1].predict(features)
    learned_increment, learned_current = learned
    learned_state = np.asarray([
      state[0] + learned_increment,
      learned_increment / MODEL_TIMESTEP_S,
      learned_current,
    ])
    inertial_state = np.asarray([
      state[0] + state[1] * MODEL_TIMESTEP_S,
      state[1],
      state[2],
    ])
    next_state = inertial_state + self.response_blend * (learned_state - inertial_state)
    return np.clip(next_state, self.state_min, self.state_max)

  def limit_score(self, samples: np.ndarray, index: int, state: np.ndarray | None = None) -> float:
    return float(self.limit.predict(feature_vector(samples, index, state))[0])

  def confidence(self, samples: np.ndarray, index: int, state: np.ndarray | None = None) -> tuple[float, bool]:
    features = feature_vector(samples, index, state)
    outside = np.maximum(np.maximum(self.feature_low - features, features - self.feature_high), 0.0)
    marginal_score = float(np.max(outside / self.dynamics.scale))
    normalized = (features - self.dynamics.mean) / self.dynamics.scale
    joint_distance = float(np.sqrt(normalized @ self.feature_precision @ normalized))
    joint_ratio = joint_distance / max(self.joint_support_threshold, 1e-9)
    confidence = float(np.exp(-0.5 * min(joint_ratio, 10.0) ** 2 - min(marginal_score, 50.0)))
    return confidence, marginal_score == 0.0 and joint_ratio <= 1.0

  def simulator(self, pinion_angle_deg: float, steering_rate_deg_s: float, eps_current_a: float,
                initial_input: FordEpsInput, *, command_history: tuple[FordEpsInput, ...] = (),
                allow_unvalidated: bool = False) -> "FordEpsSimulator":
    if not self.screening_ready and not allow_unvalidated:
      raise ValueError("virtual EPS did not pass held-out validation")
    return FordEpsSimulator(
      self, np.asarray([pinion_angle_deg, steering_rate_deg_s, eps_current_a]), initial_input, command_history,
    )

  def rollout(self, dataset: FordEpsDataset, start: int, steps: int) -> np.ndarray:
    """Replay recorded LMC2/environment inputs from one sample and return the predicted EPS state."""
    samples = dataset.samples
    if start < 0 or steps < 0 or start + steps >= len(samples):
      raise IndexError("rollout lies outside the dataset")
    segment_id = samples[start]["segment_id"]
    if samples[start + steps]["segment_id"] != segment_id:
      raise ValueError("rollout cannot cross a segment boundary")
    state = np.asarray([
      samples[start]["pinion_angle_deg"],
      samples[start]["steering_rate_deg_s"],
      samples[start]["eps_current_a"],
    ])
    for index in range(start, start + steps):
      state = self.step(samples, index, state)
    return state

  def save(self, path: str | Path, *, allow_unvalidated: bool = False) -> None:
    if not self.screening_ready and not allow_unvalidated:
      raise ValueError("virtual EPS did not pass held-out validation")
    np.savez_compressed(
      path,
      version=np.asarray(VIRTUAL_EPS_VERSION),
      timestep_s=np.asarray(MODEL_TIMESTEP_S),
      dynamics_mean=self.dynamics.mean,
      dynamics_scale=self.dynamics.scale,
      dynamics_target_mean=self.dynamics.target_mean,
      dynamics_coefficients=self.dynamics.coefficients,
      speed_dynamics_mean=np.stack([expert.mean for expert in self.speed_dynamics]),
      speed_dynamics_scale=np.stack([expert.scale for expert in self.speed_dynamics]),
      speed_dynamics_target_mean=np.stack([expert.target_mean for expert in self.speed_dynamics]),
      speed_dynamics_coefficients=np.stack([expert.coefficients for expert in self.speed_dynamics]),
      limit_mean=self.limit.mean,
      limit_scale=self.limit.scale,
      limit_target_mean=self.limit.target_mean,
      limit_coefficients=self.limit.coefficients,
      limit_threshold=np.asarray(self.limit_threshold),
      response_blend=np.asarray(self.response_blend),
      feature_low=self.feature_low,
      feature_high=self.feature_high,
      feature_precision=self.feature_precision,
      joint_support_threshold=np.asarray(self.joint_support_threshold),
      screening_ready=np.asarray(self.screening_ready),
      state_min=self.state_min,
      state_max=self.state_max,
    )

  @classmethod
  def load(cls, path: str | Path) -> "FordEpsModel":
    with np.load(path, allow_pickle=False) as archive:
      version = int(archive["version"])
      if version != VIRTUAL_EPS_VERSION:
        raise ValueError(f"unsupported Ford virtual EPS version {version}")
      if not np.isclose(float(archive["timestep_s"]), MODEL_TIMESTEP_S):
        raise ValueError("Ford virtual EPS artifact has an unsupported timestep")
      model = cls(ridge=0.0)
      model.dynamics.mean = archive["dynamics_mean"]
      model.dynamics.scale = archive["dynamics_scale"]
      model.dynamics.target_mean = archive["dynamics_target_mean"]
      model.dynamics.coefficients = archive["dynamics_coefficients"]
      model.speed_dynamics = []
      for mean, scale, target_mean, coefficients in zip(
        archive["speed_dynamics_mean"], archive["speed_dynamics_scale"],
        archive["speed_dynamics_target_mean"], archive["speed_dynamics_coefficients"], strict=True,
      ):
        expert = _Ridge(ridge=0.0)
        expert.mean = mean
        expert.scale = scale
        expert.target_mean = target_mean
        expert.coefficients = coefficients
        model.speed_dynamics.append(expert)
      model.limit.mean = archive["limit_mean"]
      model.limit.scale = archive["limit_scale"]
      model.limit.target_mean = archive["limit_target_mean"]
      model.limit.coefficients = archive["limit_coefficients"]
      model.limit_threshold = float(archive["limit_threshold"])
      model.response_blend = float(archive["response_blend"])
      model.feature_low = archive["feature_low"]
      model.feature_high = archive["feature_high"]
      model.feature_precision = archive["feature_precision"]
      model.joint_support_threshold = float(archive["joint_support_threshold"])
      model.screening_ready = bool(archive["screening_ready"])
      model.state_min = archive["state_min"]
      model.state_max = archive["state_max"]
      return model


class FordEpsSimulator:
  """Stateful input-oriented interface for replaying new LMC2 command sequences."""

  def __init__(self, model: FordEpsModel, initial_state: np.ndarray, initial_input: FordEpsInput,
               command_history: tuple[FordEpsInput, ...] = ()):
    self.model = model
    self._state = initial_state.astype(np.float64)
    history = command_history or (initial_input,)
    history = (history[0],) * max(7 - len(history), 0) + history[-7:]
    self._history: deque[np.void] = deque(
      (sample_from_input(command, (index - 6) * round(MODEL_TIMESTEP_S * 1e9)) for index, command in enumerate(history)),
      maxlen=7,
    )
    self._step = 0

  @property
  def state(self) -> np.ndarray:
    return self._state.copy()

  def step(self, inputs: FordEpsInput, dt: float = MODEL_TIMESTEP_S, *, allow_ood: bool = False) -> FordEpsOutput:
    if not np.isclose(dt, MODEL_TIMESTEP_S):
      raise ValueError("virtual EPS timestep must be exactly 50 ms")
    candidate_step = self._step + 1
    candidate = sample_from_input(inputs, round(candidate_step * MODEL_TIMESTEP_S * 1e9))
    samples = np.asarray([*list(self._history)[1:], candidate], dtype=SAMPLE_DTYPE)
    index = len(samples) - 1
    limit_score = self.model.limit_score(samples, index, self._state)
    confidence, in_distribution = self.model.confidence(samples, index, self._state)
    if not in_distribution and not allow_ood:
      raise ValueError("counterfactual input is outside the identified joint support")
    self._step = candidate_step
    self._history.append(candidate)
    self._state = self.model.step(samples, index, self._state)
    return FordEpsOutput(
      pinion_angle_deg=float(self._state[0]),
      steering_rate_deg_s=float(self._state[1]),
      eps_current_a=float(self._state[2]),
      limit_score=limit_score,
      limit_predicted=limit_score >= self.model.limit_threshold,
      confidence=confidence,
      in_distribution=in_distribution,
    )
