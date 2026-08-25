from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from opendbc.can import CANParser
from openpilot.tools.lib.logreader import LogReader


FORD_DBC = "ford_lincoln_base_pt"
LMC2 = "LateralMotionControl2"
FORD_LMC2_COEFFICIENT_SCHEMA = {
  "c0": (-5.12, 5.11, 0.01),
  "c1": (-0.5, 0.5235, 0.0005),
  # The DBC encodes C2 through 0.02094, but Panda caps active Ford steering at 0.02000.
  "c2": (-0.02, 0.02, 0.00002),
  "c3": (-0.001024, 0.001023, 0.000001),
}
DATASET_VERSION = 3
MAX_SAMPLE_AGE_NS = 100_000_000
SAMPLE_PERIOD_NS = 50_000_000
RESAMPLE_TOLERANCE_NS = 15_000_000
REQUIRED_TELEMETRY = {
  "EPAS_INFO",
  "SteeringPinion_Data",
  "Lane_Assist_Data3_FD1",
  "Yaw_Data_FD1",
  "Accel_Data_FD1",
}

SAMPLE_DTYPE = np.dtype([
  ("route_id", "U96"),
  ("segment_id", "U112"),
  ("mono_time_ns", "i8"),
  ("c0", "f8"),
  ("c1", "f8"),
  ("c2", "f8"),
  ("c3", "f8"),
  ("pinion_angle_deg", "f8"),
  ("steering_rate_deg_s", "f8"),
  ("speed_mps", "f8"),
  ("yaw_rate_rad_s", "f8"),
  ("lateral_accel_mps2", "f8"),
  ("longitudinal_accel_mps2", "f8"),
  ("eps_current_a", "f8"),
  ("eps_voltage_v", "f8"),
  ("column_torque_nm", "f8"),
  ("driver_torque_nm", "f8"),
  ("desired_angle_deg", "f8"),
  ("model_c0", "f8"),
  ("model_c1", "f8"),
  ("model_c2", "f8"),
  ("model_c3", "f8"),
  ("model_path_valid", "?"),
  ("lat_limit", "i1"),
  ("lat_status", "i1"),
  ("lat_mode", "i1"),
  ("lat_active", "?"),
  ("steering_pressed", "?"),
])


def _frames(event) -> list[tuple[int, bytes, int]]:
  return [(frame.address, bytes(frame.dat), frame.src) for frame in getattr(event, event.which())]


def route_and_segment(path: Path) -> tuple[str, str]:
  """Return canonical route and segment ids for flattened downloads or native logger directories."""
  source_name = path.parent.name if re.fullmatch(r"rlog(?:\.zst|\.bz2)?", path.name) else path.name
  match = re.fullmatch(r"(.+)--(\d+)(?:--rlog(?:\.zst|\.bz2)?)?", source_name)
  if match is None:
    return path.stem, path.stem
  return match.group(1), f"{match.group(1)}--{int(match.group(2))}"


def device_id_from_route(route_id: str) -> str:
  """Extract the dongle id from canonical (`|`) and downloaded (`_`) route names."""
  return re.split(r"[|_]", route_id, maxsplit=1)[0]


class FordEpsDataset:
  """A synchronized 20 Hz view of LMC2 commands and Ford PSCM telemetry."""

  def __init__(self, samples: np.ndarray):
    if samples.dtype != SAMPLE_DTYPE:
      samples = np.asarray(samples, dtype=SAMPLE_DTYPE)
    self.samples = _normalize_to_20hz(samples)
    self._derive_causal_steering_rates()

  def __len__(self) -> int:
    return len(self.samples)

  @classmethod
  def from_rlogs(cls, paths: Iterable[str | Path]) -> "FordEpsDataset":
    segments = []
    for path in paths:
      samples = np.fromiter(_extract_segment(Path(path)), dtype=SAMPLE_DTYPE)
      if len(samples):
        segments.append(samples)
    combined = np.concatenate(segments) if segments else np.empty(0, dtype=SAMPLE_DTYPE)
    return cls(combined)

  @classmethod
  def load(cls, path: str | Path) -> "FordEpsDataset":
    with np.load(path, allow_pickle=False) as archive:
      version = int(archive["version"])
      if version != DATASET_VERSION:
        raise ValueError(f"unsupported Ford EPS dataset version {version}")
      return cls(archive["samples"])

  def save(self, path: str | Path) -> None:
    np.savez_compressed(path, version=np.asarray(DATASET_VERSION), samples=self.samples)

  def _derive_causal_steering_rates(self) -> None:
    for segment_id in np.unique(self.samples["segment_id"]):
      indices = np.flatnonzero(self.samples["segment_id"] == segment_id)
      if len(indices) < 2:
        continue
      times = self.samples["mono_time_ns"][indices].astype(np.float64) * 1e-9
      angles = self.samples["pinion_angle_deg"][indices]
      if times[-1] <= times[0]:
        continue
      rates = np.empty(len(indices), dtype=np.float64)
      rates[0] = self.samples["steering_rate_deg_s"][indices[0]]
      rates[1:] = np.diff(angles) / np.diff(times)
      self.samples["steering_rate_deg_s"][indices] = rates


@dataclass(frozen=True)
class FordEpsInput:
  """Exogenous inputs accepted by the virtual EPS for counterfactual command replay."""

  c0: float
  c1: float
  c2: float
  c3: float
  speed_mps: float
  yaw_rate_rad_s: float = 0.0
  lateral_accel_mps2: float = 0.0
  longitudinal_accel_mps2: float = 0.0
  eps_voltage_v: float = 14.0
  column_torque_nm: float = 0.0
  driver_torque_nm: float = 0.0
  lat_active: bool = True
  lat_limit: int = 0
  lat_status: int = 0
  lat_mode: int = 0


@dataclass(frozen=True)
class FordEpsOutput:
  pinion_angle_deg: float
  steering_rate_deg_s: float
  eps_current_a: float
  limit_score: float
  limit_predicted: bool
  confidence: float
  in_distribution: bool


def sample_from_input(inputs: FordEpsInput, mono_time_ns: int = 0) -> np.void:
  sample = np.zeros(1, dtype=SAMPLE_DTYPE)[0]
  sample["route_id"] = "virtual"
  sample["segment_id"] = "virtual--0"
  sample["mono_time_ns"] = mono_time_ns
  for field in (
    "c0", "c1", "c2", "c3", "speed_mps", "yaw_rate_rad_s", "lateral_accel_mps2",
    "longitudinal_accel_mps2", "eps_voltage_v", "column_torque_nm", "driver_torque_nm", "lat_active",
    "lat_limit", "lat_status", "lat_mode",
  ):
    sample[field] = getattr(inputs, field)
  return sample


def input_from_sample(sample: np.void) -> FordEpsInput:
  """Build a counterfactual input while retaining the sample's observed environment."""
  return FordEpsInput(
    c0=float(sample["c0"]),
    c1=float(sample["c1"]),
    c2=float(sample["c2"]),
    c3=float(sample["c3"]),
    speed_mps=float(sample["speed_mps"]),
    yaw_rate_rad_s=float(sample["yaw_rate_rad_s"]),
    lateral_accel_mps2=float(sample["lateral_accel_mps2"]),
    longitudinal_accel_mps2=float(sample["longitudinal_accel_mps2"]),
    eps_voltage_v=float(sample["eps_voltage_v"]),
    column_torque_nm=float(sample["column_torque_nm"]),
    driver_torque_nm=float(sample["driver_torque_nm"]),
    lat_active=bool(sample["lat_active"]),
    lat_limit=int(sample["lat_limit"]),
    lat_status=int(sample["lat_status"]),
    lat_mode=int(sample["lat_mode"]),
  )


def _normalize_to_20hz(samples: np.ndarray) -> np.ndarray:
  """Resample onto strict 20 Hz chunks; gaps start a new dynamics history."""
  normalized = []
  for segment_id in np.unique(samples["segment_id"]):
    segment = samples[samples["segment_id"] == segment_id]
    segment = segment[np.argsort(segment["mono_time_ns"])]
    if len(segment) < 2:
      normalized.append(segment.copy())
      continue
    times = segment["mono_time_ns"]
    targets = np.arange(times[0], times[-1] + 1, SAMPLE_PERIOD_NS, dtype=np.int64)
    right = np.searchsorted(times, targets, side="left")
    right = np.clip(right, 0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    use_left = np.abs(times[left] - targets) <= np.abs(times[right] - targets)
    selected = np.where(use_left, left, right)
    valid = np.abs(times[selected] - targets) <= RESAMPLE_TOLERANCE_NS
    selected = selected[valid]
    selected_targets = targets[valid]
    if not len(selected):
      continue
    resampled = segment[selected].copy()
    resampled["mono_time_ns"] = selected_targets
    chunk_starts = np.concatenate(([0], np.flatnonzero(np.diff(selected_targets) != SAMPLE_PERIOD_NS) + 1))
    chunk_ends = np.concatenate((chunk_starts[1:], [len(resampled)]))
    multiple_chunks = len(chunk_starts) > 1
    for chunk_number, (start, end) in enumerate(zip(chunk_starts, chunk_ends, strict=True)):
      chunk = resampled[start:end]
      if multiple_chunks:
        chunk["segment_id"] = f"{str(segment_id)[:101]}~{chunk_number}"
      normalized.append(chunk)
  return np.concatenate(normalized) if normalized else np.empty(0, dtype=SAMPLE_DTYPE)


def _extract_segment(path: Path) -> Iterable[tuple]:
  route_id, segment_id = route_and_segment(path)
  rx = CANParser(FORD_DBC, [(message, 0) for message in REQUIRED_TELEMETRY], 0)
  tx = CANParser(FORD_DBC, [(LMC2, 0)], 0)
  seen: set[str] = set()
  last_seen_ns: dict[str, int] = {}
  car_state = None
  car_state_time_ns = 0
  car_control = None
  car_control_time_ns = 0

  for event in LogReader(str(path), sort_by_time=True):
    which = event.which()
    if which == "carState":
      car_state = event.carState
      car_state_time_ns = event.logMonoTime
      continue
    if which == "carControl":
      car_control = event.carControl
      car_control_time_ns = event.logMonoTime
      continue
    if which == "can":
      updated = rx.update([event.logMonoTime, _frames(event)])
      for address in updated:
        message = rx.message_states[address].name
        seen.add(message)
        last_seen_ns[message] = event.logMonoTime
      continue
    if which != "sendcan":
      continue

    updated = tx.update([event.logMonoTime, _frames(event)], sendcan=True)
    telemetry_fresh = REQUIRED_TELEMETRY.issubset(seen) and all(
      0 <= event.logMonoTime - last_seen_ns[message] <= MAX_SAMPLE_AGE_NS for message in REQUIRED_TELEMETRY
    )
    state_fresh = car_state is not None and 0 <= event.logMonoTime - car_state_time_ns <= MAX_SAMPLE_AGE_NS
    control_fresh = car_control is not None and 0 <= event.logMonoTime - car_control_time_ns <= MAX_SAMPLE_AGE_NS
    if tx.dbc.name_to_msg[LMC2].address not in updated or not telemetry_fresh or not state_fresh or not control_fresh:
      continue

    command = tx.vl[LMC2]
    epas = rx.vl["EPAS_INFO"]
    pinion = rx.vl["SteeringPinion_Data"]
    limit = rx.vl["Lane_Assist_Data3_FD1"]
    yaw = rx.vl["Yaw_Data_FD1"]
    accel = rx.vl["Accel_Data_FD1"]
    speed = float(car_state.vEgoRaw)
    steering_rate = float(car_state.steeringRateDeg)
    steering_pressed = bool(car_state.steeringPressed)
    desired_angle = float(car_control.actuators.steeringAngleDeg)
    model_path = car_control.actuators.lateralPath
    lat_active = bool(car_control.latActive)

    yield (
      route_id,
      segment_id,
      event.logMonoTime,
      command["LatCtlPathOffst_L_Actl"],
      command["LatCtlPath_An_Actl"],
      command["LatCtlCurv_No_Actl"],
      command["LatCtlCrv_NoRate2_Actl"],
      pinion["StePinComp_An_Est"],
      steering_rate,
      speed,
      yaw["VehYaw_W_Actl"],
      accel["VehLat2_A_Actl"],
      accel["VehLong2_A_Actl"],
      epas["SteMdule_I_Est"],
      epas["SteMdule_U_Meas"],
      epas["SteeringColumnTorque"],
      epas["DrvSte_Tq_Actl"],
      desired_angle,
      model_path.pathOffset,
      model_path.pathAngle,
      model_path.curvature,
      model_path.curvatureRate,
      model_path.valid,
      round(limit["LatCtlLim_D_Stat"]),
      round(limit["LatCtlSte_D_Stat"]),
      round(command["LatCtl_D2_Rq"]),
      lat_active,
      steering_pressed,
    )
