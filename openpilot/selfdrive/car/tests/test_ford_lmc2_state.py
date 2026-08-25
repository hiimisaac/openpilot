from types import SimpleNamespace

import openpilot.cereal.messaging as messaging
from opendbc.car.ford.interface import CarInterface
from opendbc.car.ford.values import CAR
from opendbc.car.structs import CarControl

from openpilot.selfdrive.car.card import ford_lmc2_control_state


def test_ford_lmc2_control_state_uses_actual_command_and_pscm_status():
  CP = CarInterface.get_non_essential_params(CAR.FORD_F_150_LIGHTNING_MK1)
  actuators = CarControl.Actuators()
  actuators.lateralPath.valid = True
  actuators.lateralPath.curvature = 0.002
  ford_car_state = SimpleNamespace(lat_ctl_limit=1)

  utilization, limit = ford_lmc2_control_state(CP, ford_car_state, actuators, lat_active=True)

  assert utilization == 0.8
  assert limit == 1


def test_ford_lmc2_control_state_is_zero_when_inactive():
  CP = CarInterface.get_non_essential_params(CAR.FORD_F_150_LIGHTNING_MK1)
  actuators = CarControl.Actuators()
  actuators.lateralPath.curvature = 0.02
  ford_car_state = SimpleNamespace(lat_ctl_limit=2)

  assert ford_lmc2_control_state(CP, ford_car_state, actuators, lat_active=False) == (0.0, 0)


def test_ford_lmc2_custom_message_round_trip():
  msg = messaging.new_message("fordLmc2ControlState")
  msg.fordLmc2ControlState.utilization = -0.8
  msg.fordLmc2ControlState.limit = "close"
  msg.fordLmc2ControlState.adaptiveEnabled = True
  msg.fordLmc2ControlState.adaptiveGain = 0.08
  msg.fordLmc2ControlState.adaptiveReferenceCurvature = 0.012
  msg.fordLmc2ControlState.adaptiveTrackingError = 0.003
  msg.fordLmc2ControlState.adapting = True

  assert msg.fordLmc2ControlState.utilization < -0.79
  assert msg.fordLmc2ControlState.limit.raw == 1
  assert msg.fordLmc2ControlState.adaptiveEnabled
  assert msg.fordLmc2ControlState.adaptiveGain > 0.079
  assert msg.fordLmc2ControlState.adaptiveReferenceCurvature > 0.011
  assert msg.fordLmc2ControlState.adaptiveTrackingError > 0.002
  assert msg.fordLmc2ControlState.adapting
