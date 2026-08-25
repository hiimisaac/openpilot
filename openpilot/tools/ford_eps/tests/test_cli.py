from pathlib import Path

from tools.car_porting.ford_eps_identification import select_complete_routes


def test_segment_budget_never_splits_a_route():
  paths = [
    Path(f"device_route-a--{segment}--rlog.zst") for segment in range(3)
  ] + [
    Path(f"device_route-b--{segment}--rlog.zst") for segment in range(3)
  ]

  selected = select_complete_routes(paths, max_segments=4, seed=0)

  selected_names = {path.name for path in selected}
  route_a = {path.name for path in paths if "route-a" in path.name}
  route_b = {path.name for path in paths if "route-b" in path.name}
  assert selected_names in (route_a, route_b)


def test_segment_budget_groups_native_logger_directories():
  paths = [
    Path(f"/data/media/0/realdata/device|route-a--{segment}/rlog.zst") for segment in range(3)
  ] + [
    Path(f"/data/media/0/realdata/device|route-b--{segment}/rlog.zst") for segment in range(3)
  ]

  selected = select_complete_routes(paths, max_segments=4, seed=0)

  selected_parents = {path.parent.name for path in selected}
  route_a = {path.parent.name for path in paths if "route-a" in path.parent.name}
  route_b = {path.parent.name for path in paths if "route-b" in path.parent.name}
  assert selected_parents in (route_a, route_b)
