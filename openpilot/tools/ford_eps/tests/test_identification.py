import math
from pathlib import Path
import random

import openpilot.cereal.messaging as messaging
from opendbc.can import CANPacker
import numpy as np
import pytest
from openpilot.tools.ford_eps import AnalysisConfig, FordEpsDataset, FordEpsInput, FordEpsModel, fit, identify
from openpilot.tools.ford_eps.dataset import device_id_from_route, route_and_segment
from openpilot.tools.lib.logreader import save_log


DBC = "ford_lincoln_base_pt"


def can_event(service: str, mono_time: int, frames: list[tuple[int, bytes, int]]):
  event = messaging.new_message(service, len(frames))
  event.logMonoTime = mono_time
  for out, (address, dat, src) in zip(getattr(event, service), frames, strict=True):
    out.address = address
    out.dat = dat
    out.src = src
  return event.as_reader()


def test_extracts_synchronized_pscm_sample_from_rlog(tmp_path: Path):
  packer = CANPacker(DBC)
  mono_time = 1_000_000_000
  telemetry = [
    packer.make_can_msg("EPAS_INFO", 0, {
      "SteMdule_U_Meas": 14.4,
      "SteMdule_I_Est": 3.0,
      "SteeringColumnTorque": 1.0,
      "DrvSte_Tq_Actl": 0.5,
      "SteMdule_D_Stat": 2,
    }),
    packer.make_can_msg("SteeringPinion_Data", 0, {
      "StePinComp_An_Est": 12.3,
      "StePinCompAnEst_D_Qf": 3,
    }),
    packer.make_can_msg("Lane_Assist_Data3_FD1", 0, {
      "LatCtlLim_D_Stat": 1,
      "LatCtlSte_D_Stat": 2,
    }),
    packer.make_can_msg("Yaw_Data_FD1", 0, {
      "VehYaw_W_Actl": 0.1,
      "VehRol_W_Actl": 0.0,
    }),
    packer.make_can_msg("Accel_Data_FD1", 0, {
      "VehLong2_A_Actl": 0.2,
      "VehLat2_A_Actl": 0.5,
    }),
  ]
  command = packer.make_can_msg("LateralMotionControl2", 0, {
    "LatCtl_D2_Rq": 2,
    "LatCtlPathOffst_L_Actl": 0.1,
    "LatCtlPath_An_Actl": 0.02,
    "LatCtlCurv_No_Actl": 0.01,
    "LatCtlCrv_NoRate2_Actl": 0.0001,
  })

  car_state = messaging.new_message("carState")
  car_state.logMonoTime = mono_time
  car_state.carState.vEgo = 7.5
  car_state.carState.vEgoRaw = 7.6
  car_state.carState.steeringRateDeg = 4.0
  car_state.carState.steeringPressed = False

  car_control = messaging.new_message("carControl")
  car_control.logMonoTime = mono_time
  car_control.carControl.latActive = True
  car_control.carControl.actuators.steeringAngleDeg = 15.0

  rlog = tmp_path / "device_00000001--routehash--0--rlog.zst"
  save_log(str(rlog), [
    can_event("can", mono_time, telemetry),
    car_state.as_reader(),
    car_control.as_reader(),
    can_event("sendcan", mono_time + 10_000_000, [command]),
    can_event("sendcan", mono_time + 500_000_000, [command]),
  ])

  dataset = FordEpsDataset.from_rlogs([rlog])

  assert len(dataset) == 1
  sample = dataset.samples[0]
  assert sample["route_id"] == "device_00000001--routehash"
  assert sample["segment_id"].endswith("--0")
  assert sample["c0"] == pytest.approx(0.1)
  assert sample["c1"] == pytest.approx(0.02)
  assert sample["c2"] == pytest.approx(0.01)
  assert sample["c3"] == pytest.approx(0.0001)
  assert sample["pinion_angle_deg"] == pytest.approx(12.3)
  assert sample["steering_rate_deg_s"] == 4.0
  assert sample["speed_mps"] == pytest.approx(7.6)
  assert sample["eps_current_a"] == pytest.approx(3.0)
  assert sample["eps_voltage_v"] == pytest.approx(14.4)
  assert sample["column_torque_nm"] == pytest.approx(1.0)
  assert sample["desired_angle_deg"] == pytest.approx(15.0)
  assert sample["lat_limit"] == 1
  assert sample["lat_active"]

  cache = tmp_path / "ford-eps-dataset.npz"
  dataset.save(cache)
  restored = FordEpsDataset.load(cache)
  assert len(restored) == 1
  assert restored.samples[0]["segment_id"] == sample["segment_id"]
  assert restored.samples[0]["pinion_angle_deg"] == sample["pinion_angle_deg"]


def synthetic_eps_rlog(path: Path, phase: float, sample_period_s: float = 0.05, sample_count: int = 320,
                       response_delay_steps: int = 0, rapid_commands: bool = False) -> None:
  packer = CANPacker(DBC)
  messages = []
  angle = 0.0
  targets: list[float] = []
  rng = random.Random(round(phase * 1000))
  for i in range(sample_count):
    mono_time = 1_000_000_000 + round(i * sample_period_s * 1e9)
    target_angle = rng.uniform(-28.0, 28.0) if rapid_commands else \
                   32.0 * math.sin(i * 0.045 + phase) + 7.0 * math.sin(i * 0.13 + phase)
    targets.append(target_angle)
    applied_target = targets[max(0, i - response_delay_steps)]
    next_angle = angle + 0.16 * (applied_target - angle)
    rate = (next_angle - angle) / sample_period_s
    angle = next_angle
    c2 = target_angle / 2000.0
    current = 0.2 + 0.08 * abs(applied_target - angle)

    telemetry = [
      packer.make_can_msg("EPAS_INFO", 0, {
        "SteMdule_U_Meas": 14.4,
        "SteMdule_I_Est": current,
        "SteeringColumnTorque": 0.0,
        "DrvSte_Tq_Actl": 0.0,
        "SteMdule_D_Stat": 2,
      }),
      packer.make_can_msg("SteeringPinion_Data", 0, {
        "StePinComp_An_Est": angle,
        "StePinCompAnEst_D_Qf": 3,
      }),
      packer.make_can_msg("Lane_Assist_Data3_FD1", 0, {
        "LatCtlLim_D_Stat": int(abs(angle) > 28.0),
        "LatCtlSte_D_Stat": 2,
      }),
      packer.make_can_msg("Yaw_Data_FD1", 0, {
        "VehYaw_W_Actl": angle * 0.001,
        "VehRol_W_Actl": 0.0,
      }),
      packer.make_can_msg("Accel_Data_FD1", 0, {
        "VehLong2_A_Actl": 0.0,
        "VehLat2_A_Actl": angle * 0.01,
      }),
    ]
    command = packer.make_can_msg("LateralMotionControl2", 0, {
      "LatCtl_D2_Rq": 2,
      "LatCtlPathOffst_L_Actl": 0.0,
      "LatCtlPath_An_Actl": 0.0,
      "LatCtlCurv_No_Actl": c2,
      "LatCtlCrv_NoRate2_Actl": 0.0,
    })
    car_state = messaging.new_message("carState")
    car_state.logMonoTime = mono_time
    car_state.carState.vEgoRaw = 10.0 + 0.5 * math.sin(i * 0.02)
    car_state.carState.steeringRateDeg = rate
    car_control = messaging.new_message("carControl")
    car_control.logMonoTime = mono_time
    car_control.carControl.latActive = True
    car_control.carControl.actuators.steeringAngleDeg = target_angle
    messages.extend([
      can_event("can", mono_time, telemetry),
      car_state.as_reader(),
      car_control.as_reader(),
      can_event("sendcan", mono_time + 10_000_000, [command]),
    ])
  save_log(str(path), messages)


def test_normalizes_high_rate_command_logs_to_20hz(tmp_path: Path):
  rlog = tmp_path / "device_00000003--high-rate--0--rlog.zst"
  synthetic_eps_rlog(rlog, 0.0, sample_period_s=0.01, sample_count=100)

  dataset = FordEpsDataset.from_rlogs([rlog])

  assert 19 <= len(dataset) <= 21
  assert np.median(np.diff(dataset.samples["mono_time_ns"])) == pytest.approx(50_000_000, abs=10_000_000)


def test_normalizes_irregular_samples_to_strict_20hz_chunks():
  samples = np.zeros(6, dtype=FordEpsDataset.from_rlogs([]).samples.dtype)
  samples["route_id"] = "device_route"
  samples["segment_id"] = "device_route--0"
  samples["mono_time_ns"] = np.asarray([0, 49_000_000, 101_000_000, 202_000_000, 251_000_000, 299_000_000])

  dataset = FordEpsDataset(samples)

  for segment_id in np.unique(dataset.samples["segment_id"]):
    segment = dataset.samples[dataset.samples["segment_id"] == segment_id]
    np.testing.assert_array_equal(np.diff(segment["mono_time_ns"]), 50_000_000)
  assert len(np.unique(dataset.samples["segment_id"])) == 2


def test_derives_causal_steering_rate_without_future_angle_leakage():
  samples = np.zeros(3, dtype=FordEpsDataset.from_rlogs([]).samples.dtype)
  samples["route_id"] = "device_route"
  samples["segment_id"] = "device_route--0"
  samples["mono_time_ns"] = np.asarray([0, 50_000_000, 100_000_000])
  samples["pinion_angle_deg"] = np.asarray([0.0, 1.0, 3.0])

  dataset = FordEpsDataset(samples)

  np.testing.assert_allclose(dataset.samples["steering_rate_deg_s"], [0.0, 20.0, 40.0])


def test_parses_flattened_and_native_logger_route_paths():
  route = "0123456789abcdef|2026-08-24--12-00-00"

  assert route_and_segment(Path(f"/tmp/{route}--7--rlog.zst")) == (route, f"{route}--7")
  assert route_and_segment(Path(f"/tmp/{route}--7/rlog.zst")) == (route, f"{route}--7")
  assert device_id_from_route(route) == "0123456789abcdef"
  assert device_id_from_route("0123456789abcdef_0000001d--routehash") == "0123456789abcdef"


def test_identifies_eps_response_on_held_out_route(tmp_path: Path):
  train = tmp_path / "device_00000001--train--0--rlog.zst"
  validation = tmp_path / "device_00000002--validation--0--rlog.zst"
  synthetic_eps_rlog(train, 0.0)
  synthetic_eps_rlog(validation, 0.7)

  report = identify(
    [train, validation],
    AnalysisConfig(horizons_s=(0.25, 0.5), validation_route="device_00000002--validation"),
  )

  assert report.sample_count == 640
  assert report.train_route_count == 1
  assert report.validation_route_count == 1
  for metrics in report.horizons.values():
    assert metrics.model_angle_mae_deg < metrics.constant_angle_mae_deg
    assert metrics.model_angle_mae_deg < metrics.constant_rate_angle_mae_deg
    assert metrics.route_model_angle_mae_p90_deg >= 0.0
  assert report.limit_metrics.recall > 0.5
  assert report.limit_metrics.identifiable
  assert report.coefficient_excitation["c2"].standard_deviation > 0.0


def test_identifies_delayed_pscm_response(tmp_path: Path):
  train = tmp_path / "device_00000004--delayed-train--0--rlog.zst"
  validation = tmp_path / "device_00000005--delayed-validation--0--rlog.zst"
  synthetic_eps_rlog(train, 0.0, response_delay_steps=4, rapid_commands=True)
  synthetic_eps_rlog(validation, 1.1, response_delay_steps=4, rapid_commands=True)

  report = identify(
    [train, validation],
    AnalysisConfig(horizons_s=(0.25, 0.5), validation_route="device_00000005--delayed-validation"),
  )

  for metrics in report.horizons.values():
    assert metrics.model_angle_mae_deg < metrics.constant_rate_angle_mae_deg


def test_saves_and_reloads_replayable_virtual_eps(tmp_path: Path):
  train = tmp_path / "device_00000006--artifact-train--0--rlog.zst"
  validation = tmp_path / "device_00000006--artifact-validation--0--rlog.zst"
  synthetic_eps_rlog(train, 0.0)
  synthetic_eps_rlog(validation, 0.9)
  dataset = FordEpsDataset.from_rlogs([train, validation])

  result = fit(dataset, AnalysisConfig(validation_route="device_00000006--artifact-validation"))
  result.model.response_blend = 0.37
  initial_state = np.asarray([
    dataset.samples[350]["pinion_angle_deg"],
    dataset.samples[350]["steering_rate_deg_s"],
    dataset.samples[350]["eps_current_a"],
  ])
  next_state = result.model.step(dataset.samples, 350, initial_state)
  assert next_state[1] == pytest.approx((next_state[0] - initial_state[0]) / 0.05)
  before = result.model.rollout(dataset, 350, 5)
  artifact = tmp_path / "virtual-eps.npz"
  result.model.save(artifact, allow_unvalidated=True)
  restored = FordEpsModel.load(artifact)

  assert restored.response_blend == pytest.approx(0.37)
  assert len(restored.speed_dynamics) == 5
  np.testing.assert_allclose(restored.rollout(dataset, 350, 5), before)


def test_validation_excludes_complete_driver_override_window(tmp_path: Path):
  train = tmp_path / "device_00000007--override-train--0--rlog.zst"
  validation = tmp_path / "device_00000007--override-validation--0--rlog.zst"
  synthetic_eps_rlog(train, 0.0)
  synthetic_eps_rlog(validation, 0.4)
  dataset = FordEpsDataset.from_rlogs([train, validation])
  dataset.samples[340]["steering_pressed"] = True

  report = identify(
    dataset,
    AnalysisConfig(
      horizons_s=(0.25,), validation_route="device_00000007--override-validation", rollout_stride=1,
    ),
  )

  assert report.horizons["0.25"].sample_count == 309


def test_simulates_new_command_sequence_without_a_recorded_dataset(tmp_path: Path):
  train = tmp_path / "device_00000008--input-train--0--rlog.zst"
  validation = tmp_path / "device_00000008--input-validation--0--rlog.zst"
  synthetic_eps_rlog(train, 0.0)
  synthetic_eps_rlog(validation, 0.8)
  dataset = FordEpsDataset.from_rlogs([train, validation])
  result = fit(dataset, AnalysisConfig(validation_route="device_00000008--input-validation"))
  assert not result.report.screening_ready
  assert result.report.screening_failures
  sample = dataset.samples[100]
  inputs = FordEpsInput(
    c0=float(sample["c0"]), c1=float(sample["c1"]), c2=float(sample["c2"]), c3=float(sample["c3"]),
    speed_mps=float(sample["speed_mps"]), yaw_rate_rad_s=float(sample["yaw_rate_rad_s"]),
    lateral_accel_mps2=float(sample["lateral_accel_mps2"]),
    longitudinal_accel_mps2=float(sample["longitudinal_accel_mps2"]),
    eps_voltage_v=float(sample["eps_voltage_v"]), column_torque_nm=float(sample["column_torque_nm"]),
    driver_torque_nm=float(sample["driver_torque_nm"]), lat_active=bool(sample["lat_active"]),
  )
  simulator = result.model.simulator(
    pinion_angle_deg=float(sample["pinion_angle_deg"]), steering_rate_deg_s=float(sample["steering_rate_deg_s"]),
    eps_current_a=float(sample["eps_current_a"]), initial_input=inputs, allow_unvalidated=True,
  )

  outputs = [simulator.step(inputs, allow_ood=True) for _ in range(10)]

  ood_simulator = result.model.simulator(
    pinion_angle_deg=0.0, steering_rate_deg_s=0.0, eps_current_a=0.2, initial_input=inputs,
    allow_unvalidated=True,
  )
  before_ood = ood_simulator.state
  with pytest.raises(ValueError, match="outside the identified joint support"):
    ood_simulator.step(FordEpsInput(c0=100.0, c1=0.0, c2=0.01, c3=0.0, speed_mps=10.0))
  np.testing.assert_array_equal(ood_simulator.state, before_ood)
  out_of_distribution = ood_simulator.step(
    FordEpsInput(c0=100.0, c1=0.0, c2=0.01, c3=0.0, speed_mps=10.0), allow_ood=True,
  )

  assert outputs[-1].pinion_angle_deg != pytest.approx(outputs[0].pinion_angle_deg)
  assert math.isfinite(outputs[-1].limit_score)
  assert outputs[0].confidence > out_of_distribution.confidence
  assert not out_of_distribution.in_distribution

  with pytest.raises(ValueError, match="50 ms"):
    simulator.step(inputs, dt=0.04)


def test_unvalidated_model_cannot_create_screening_simulator(tmp_path: Path):
  model = FordEpsModel(ridge=0.0)
  inputs = FordEpsInput(c0=0.0, c1=0.0, c2=0.0, c3=0.0, speed_mps=10.0)

  with pytest.raises(ValueError, match="did not pass held-out validation"):
    model.simulator(0.0, 0.0, 0.0, inputs)
  with pytest.raises(ValueError, match="did not pass held-out validation"):
    model.save(tmp_path / "unvalidated.npz")
