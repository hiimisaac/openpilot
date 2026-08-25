#!/usr/bin/env python3
"""Identify and validate an equivalent Ford PSCM/EPS model from local rlogs."""

import argparse
import glob
from pathlib import Path
import random

from tqdm import tqdm

from openpilot.tools.ford_eps import AnalysisConfig, FordEpsDataset, FordEpsPlannerConfig, evaluate_planner, fit
from openpilot.tools.ford_eps.dataset import device_id_from_route, route_and_segment


def find_rlogs(sources: list[str]) -> list[Path]:
  paths: set[Path] = set()
  for source in sources:
    path = Path(source).expanduser()
    if path.is_dir():
      paths.update(path.rglob("*rlog.zst"))
      paths.update(path.rglob("*rlog.bz2"))
    elif path.is_file():
      paths.add(path)
    else:
      paths.update(Path(match) for match in glob.glob(str(path), recursive=True))
  return sorted(paths)


def select_complete_routes(paths: list[Path], max_segments: int | None, seed: int) -> list[Path]:
  if max_segments is None or len(paths) <= max_segments:
    return paths
  routes: dict[str, list[Path]] = {}
  for path in paths:
    route, _ = route_and_segment(path)
    routes.setdefault(route, []).append(path)
  route_ids = sorted(routes)
  random.Random(seed).shuffle(route_ids)
  selected: list[Path] = []
  for route_id in route_ids:
    route_paths = routes[route_id]
    if selected and len(selected) + len(route_paths) > max_segments:
      continue
    selected.extend(route_paths)
    if len(selected) >= max_segments:
      break
  return sorted(selected)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("sources", nargs="*", help="rlog files, directories, or glob patterns")
  parser.add_argument("--cache", type=Path, help="load an existing .npz dataset cache")
  parser.add_argument("--dataset-output", type=Path, help="write the decoded dataset to this .npz cache")
  parser.add_argument("--model-output", type=Path, help="write the fitted replayable virtual EPS to this .npz artifact")
  parser.add_argument("--planner-output", type=Path, help="write held-out inverse-planner screening metrics to this JSON file")
  parser.add_argument("--planner-max-windows", type=int, help="evenly sample at most this many held-out planner windows")
  parser.add_argument("--planner-stride", type=int, default=20, help="planner evaluation stride in 50 ms samples")
  parser.add_argument("--planner-horizon", type=int, default=5, help="planner evaluation horizon in 50 ms samples")
  parser.add_argument("--planner-allow-c2", action="store_true", help="allow the inverse planner to alter persistent C2")
  parser.add_argument(
    "--allow-unvalidated-model", action="store_true",
    help="write an artifact that failed held-out screening checks (unsafe for controller comparison)",
  )
  parser.add_argument("--output", type=Path, default=Path("ford_eps_report.json"), help="identification report path")
  parser.add_argument("--max-segments", type=int, help="randomly select at most this many segments")
  parser.add_argument("--seed", type=int, default=0, help="segment selection seed")
  parser.add_argument("--validation-route", help="route id to reserve in full for validation")
  parser.add_argument("--device-id", help="dongle/device id to identify; required when the cache contains multiple vehicles")
  parser.add_argument("--validation-fraction", type=float, default=0.2)
  parser.add_argument("--horizons", type=float, nargs="+", default=(0.25, 0.5, 1.0), metavar="SECONDS")
  parser.add_argument("--include-inactive", action="store_true", help="include inactive and driver-override transitions")
  args = parser.parse_args()

  if args.cache is not None:
    if args.sources:
      parser.error("sources cannot be combined with --cache")
    dataset = FordEpsDataset.load(args.cache)
  else:
    if not args.sources:
      parser.error("provide at least one rlog source or --cache")
    paths = find_rlogs(args.sources)
    if args.device_id is not None:
      paths = [path for path in paths if device_id_from_route(route_and_segment(path)[0]) == args.device_id]
    paths = select_complete_routes(paths, args.max_segments, args.seed)
    if not paths:
      parser.error("no rlogs found")
    print(f"Decoding {len(paths)} Ford rlog segments...")
    dataset = FordEpsDataset.from_rlogs(tqdm(paths, unit="segment"))
    if args.dataset_output is not None:
      args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
      dataset.save(args.dataset_output)
      print(f"Wrote {len(dataset):,} synchronized samples to {args.dataset_output}")

  config = AnalysisConfig(
    horizons_s=tuple(args.horizons),
    validation_fraction=args.validation_fraction,
    validation_route=args.validation_route,
    device_id=args.device_id,
    require_active=not args.include_inactive,
  )
  result = fit(dataset, config)
  report = result.report
  args.output.parent.mkdir(parents=True, exist_ok=True)
  report.write_json(args.output)
  print(f"Wrote identification report to {args.output}")
  if args.planner_output is not None:
    if not report.screening_ready:
      parser.error("model failed held-out screening checks; planner evaluation is unavailable")
    planner_evaluation = evaluate_planner(
      dataset, result.model, report.validation_routes,
      config=FordEpsPlannerConfig(allow_c2_adjustment=args.planner_allow_c2),
      horizon_steps=args.planner_horizon, stride=args.planner_stride, max_windows=args.planner_max_windows,
    )
    args.planner_output.parent.mkdir(parents=True, exist_ok=True)
    planner_evaluation.write_json(args.planner_output)
    print(
      f"Wrote inverse-planner screening report to {args.planner_output}: " +
      f"model-internal predicted MAE {planner_evaluation.predicted_baseline_angle_mae_deg:.2f} -> " +
      f"{planner_evaluation.predicted_planned_angle_mae_deg:.2f} deg; " +
      f"OOD rejected {planner_evaluation.rejected_window_count}/{planner_evaluation.requested_window_count}",
    )
  if args.model_output is not None:
    if not report.screening_ready and not args.allow_unvalidated_model:
      parser.error("model failed held-out screening checks; omit --model-output or pass --allow-unvalidated-model")
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    result.model.save(args.model_output, allow_unvalidated=args.allow_unvalidated_model)
    print(f"Wrote replayable virtual EPS to {args.model_output}")
  print(f"Device: {report.device_id}; samples: {report.sample_count:,}; training transitions: {report.transition_count:,}")
  for horizon, metrics in report.horizons.items():
    print(
      f"{horizon}s: model angle MAE {metrics.model_angle_mae_deg:.2f} deg; " +
      f"hold {metrics.constant_angle_mae_deg:.2f}; constant-rate {metrics.constant_rate_angle_mae_deg:.2f}",
    )
  print(
    f"PSCM limit: prevalence {report.limit_metrics.prevalence:.1%}; " +
    f"precision {report.limit_metrics.precision:.1%}; recall {report.limit_metrics.recall:.1%}",
  )


if __name__ == "__main__":
  main()
