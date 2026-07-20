#!/usr/bin/env python3
import math
import time

import openpilot.cereal.messaging as messaging
from openpilot.common.params import Params
from opendbc.car import structs
from opendbc.car.ford.lateral_path_state import driver_steering_opposes_command
from opendbc.car.vehicle_model import VehicleModel

from openpilot.selfdrive.debug.ford_lmc2_shadow import (
  decode_lmc2_command,
  Lmc2Command,
  SimpleLmc2Controller,
)


LMC2_ADDRESS = 982


def _command_dict(command: Lmc2Command) -> dict[str, float]:
  return {
    "pathOffset": command.path_offset,
    "pathAngle": command.path_angle,
    "curvature": command.curvature,
    "curvatureRate": command.curvature_rate,
  }


def _controller_sign(command: Lmc2Command) -> Lmc2Command:
  """Convert Ford's inverted CAN sign back to the controller convention."""
  return Lmc2Command(
    path_offset=-command.path_offset,
    path_angle=-command.path_angle,
    curvature=-command.curvature,
    curvature_rate=-command.curvature_rate,
  )


def main() -> None:
  CP = messaging.log_from_bytes(Params().get("CarParams", block=True), structs.CarParams)
  vehicle_model = VehicleModel(CP)
  shadow_controller = SimpleLmc2Controller()

  sm = messaging.SubMaster(["carControl", "carState", "sendcan"], poll="sendcan")
  pm = messaging.PubMaster(["fordLmc2Shadow"])

  while True:
    sm.update()
    if not sm.updated["sendcan"]:
      continue

    CC = sm["carControl"]
    CS = sm["carState"]
    for can in sm["sendcan"]:
      if can.address != LMC2_ADDRESS:
        continue

      start_time = time.perf_counter_ns()
      live_command = _controller_sign(decode_lmc2_command(bytes(can.dat)))
      measured_curvature = -float(vehicle_model.calc_curvature(
        math.radians(CS.steeringAngleDeg), CS.vEgo, 0.0,
      ))
      steering_angle_error_deg = CC.actuators.steeringAngleDeg - CS.steeringAngleDeg
      driver_override = driver_steering_opposes_command(
        CC.latActive and CS.steeringPressed,
        CS.steeringTorque,
        steering_angle_error_deg,
      )
      shadow_command = shadow_controller.update(
        CC.actuators.curvature,
        measured_curvature,
        CS.vEgo,
        CC.latActive,
        driver_override,
      )
      computation_time_s = (time.perf_counter_ns() - start_time) * 1e-9

      msg = messaging.new_message(
        "fordLmc2Shadow",
        valid=sm.all_checks(["carControl", "carState", "sendcan"]),
      )
      msg.fordLmc2Shadow.active = CC.latActive
      msg.fordLmc2Shadow.driverOverride = driver_override
      msg.fordLmc2Shadow.vEgo = CS.vEgo
      msg.fordLmc2Shadow.desiredCurvature = CC.actuators.curvature
      msg.fordLmc2Shadow.desiredSteeringAngleDeg = CC.actuators.steeringAngleDeg
      msg.fordLmc2Shadow.measuredCurvature = measured_curvature
      msg.fordLmc2Shadow.measuredSteeringAngleDeg = CS.steeringAngleDeg
      msg.fordLmc2Shadow.liveCommand = _command_dict(live_command)
      msg.fordLmc2Shadow.shadowCommand = _command_dict(shadow_command)
      msg.fordLmc2Shadow.computationTimeS = computation_time_s
      pm.send("fordLmc2Shadow", msg)


if __name__ == "__main__":
  main()
