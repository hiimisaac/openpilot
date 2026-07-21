#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

from openpilot.selfdrive.debug.ford_lmc2_shadow import Lmc2Command, polynomial_state


COEFFICIENTS = ("pathOffset", "pathAngle", "curvature", "curvatureRate")
COMMAND_ATTRS = {
  "pathOffset": "path_offset",
  "pathAngle": "path_angle",
  "curvature": "curvature",
  "curvatureRate": "curvature_rate",
}
COEFFICIENT_SCALES = {
  "pathOffset": 4.61,
  "pathAngle": 0.497,
  "curvature": 0.02,
  "curvatureRate": 0.001024,
}
HORIZONS = (3.5, 7.0, 12.0)
MAX_MODEL_TO_CONTROL_AGE_S = 0.1
MAX_CONTROL_TO_CAN_AGE_S = 0.03


def timing_is_coherent(model_to_control_age_s: float, control_to_can_age_s: float) -> bool:
  return 0.0 <= model_to_control_age_s <= MAX_MODEL_TO_CONTROL_AGE_S and \
         0.0 <= control_to_can_age_s <= MAX_CONTROL_TO_CAN_AGE_S


def _distribution(values: list[float]) -> dict[str, float]:
  if not values:
    return {"mean": 0.0, "mae": 0.0, "p95": 0.0, "max": 0.0, "rmse": 0.0}
  absolute = sorted(abs(value) for value in values)
  p95_index = math.ceil(0.95 * len(absolute)) - 1
  mean_absolute = sum(absolute) / len(absolute)
  return {
    "mean": sum(values) / len(values),
    "mae": mean_absolute,
    "p95": absolute[p95_index],
    "max": absolute[-1],
    "rmse": math.sqrt(sum(value * value for value in values) / len(values)),
  }


ShadowPair = tuple[Lmc2Command, Lmc2Command, float] | tuple[Lmc2Command, Lmc2Command, float, float, float]


def summarize_pairs(pairs: list[ShadowPair]) -> dict:
  coefficient_errors = {name: [] for name in COEFFICIENTS}
  polynomial_errors = {
    distance: {name: [] for name in ("pathOffset", "pathAngle", "curvature")}
    for distance in HORIZONS
  }
  normalized_max_errors = []
  computation_times = []
  model_to_control_ages = []
  control_to_can_ages = []

  for pair in pairs:
    live, shadow, computation_time_s, *ages = pair
    sample_normalized_errors = []
    for name in COEFFICIENTS:
      attr = COMMAND_ATTRS[name]
      error = getattr(shadow, attr) - getattr(live, attr)
      coefficient_errors[name].append(error)
      sample_normalized_errors.append(abs(error) / COEFFICIENT_SCALES[name])
    normalized_max_errors.append(max(sample_normalized_errors))

    for distance in HORIZONS:
      live_state = polynomial_state(live, distance)
      shadow_state = polynomial_state(shadow, distance)
      for name, live_value, shadow_value in zip(
        ("pathOffset", "pathAngle", "curvature"), live_state, shadow_state, strict=True,
      ):
        polynomial_errors[distance][name].append(shadow_value - live_value)
    computation_times.append(computation_time_s)
    if ages:
      model_to_control_ages.append(ages[0])
      control_to_can_ages.append(ages[1])

  sample_count = len(pairs)
  return {
    "sampleCount": sample_count,
    "coefficients": {name: _distribution(errors) for name, errors in coefficient_errors.items()},
    "polynomial": {
      f"{distance:g}m": {name: _distribution(errors) for name, errors in states.items()}
      for distance, states in polynomial_errors.items()
    },
    "maxNormalizedCoefficientError": _distribution(normalized_max_errors),
    "samplesWithinCoefficientRange": {
      "1Percent": sum(error <= 0.01 for error in normalized_max_errors) / sample_count if sample_count else 0.0,
      "5Percent": sum(error <= 0.05 for error in normalized_max_errors) / sample_count if sample_count else 0.0,
      "10Percent": sum(error <= 0.10 for error in normalized_max_errors) / sample_count if sample_count else 0.0,
    },
    "computationTimeS": _distribution(computation_times),
    "modelToControlAgeS": _distribution(model_to_control_ages),
    "controlToCanAgeS": _distribution(control_to_can_ages),
  }


def _command_from_capnp(command) -> Lmc2Command:
  return Lmc2Command(
    path_offset=command.pathOffset,
    path_angle=command.pathAngle,
    curvature=command.curvature,
    curvature_rate=command.curvatureRate,
  )


def load_pairs(paths: list[str], include_inactive: bool = False,
               include_driver_override: bool = False,
               include_high_skew: bool = False) -> list[ShadowPair]:
  from openpilot.tools.lib.logreader import LogReader

  pairs = []
  for path in paths:
    for msg in LogReader(path):
      if msg.which() != "fordLmc2Shadow" or not msg.valid:
        continue
      sample = msg.fordLmc2Shadow
      model_shadow = sample.carControlMonoTime != 0
      if model_shadow and not sample.shadowValid:
        continue
      if not include_inactive and not sample.active:
        continue
      if not include_driver_override and sample.driverOverride:
        continue
      pair: ShadowPair = (
        _command_from_capnp(sample.liveCommand),
        _command_from_capnp(sample.shadowCommand),
        sample.computationTimeS,
      )
      if model_shadow:
        model_to_control_age_s = (sample.carControlMonoTime - sample.modelMonoTime) * 1e-9
        control_to_can_age_s = (sample.sendcanMonoTime - sample.carControlMonoTime) * 1e-9
        if not include_high_skew and not timing_is_coherent(model_to_control_age_s, control_to_can_age_s):
          continue
        pair = (*pair, model_to_control_age_s, control_to_can_age_s)
      pairs.append(pair)
  return pairs


def main() -> None:
  parser = argparse.ArgumentParser(description="Compare live and simple-shadow Ford LMC2 polynomials")
  parser.add_argument("rlogs", nargs="+", type=Path)
  parser.add_argument("--include-inactive", action="store_true")
  parser.add_argument("--include-driver-override", action="store_true")
  parser.add_argument("--include-high-skew", action="store_true")
  args = parser.parse_args()

  pairs = load_pairs(
    [str(path) for path in args.rlogs],
    include_inactive=args.include_inactive,
    include_driver_override=args.include_driver_override,
    include_high_skew=args.include_high_skew,
  )
  print(json.dumps(summarize_pairs(pairs), indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
