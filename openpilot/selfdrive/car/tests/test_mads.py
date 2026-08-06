from opendbc.car.structs import car
from opendbc.car.ford.interface import CarInterface
from opendbc.car.ford.values import CAR, FordFlags
from opendbc.safety import ALTERNATIVE_EXPERIENCE

from openpilot.selfdrive.car.mads import is_mads_available, is_mads_configured, is_mads_lateral_only


def test_mads_is_available_on_all_ford_canfd_platforms():
  canfd_platforms = []
  for candidate in CAR:
    CP = CarInterface.get_non_essential_params(candidate)
    if CP.flags & FordFlags.CANFD:
      canfd_platforms.append(candidate)
      assert is_mads_available(CP), candidate

  assert CAR.FORD_F_150_LIGHTNING_MK1 in canfd_platforms
  assert len(canfd_platforms) > 1


def test_mads_is_unavailable_on_other_brands():
  CP = car.CarParams.new_message()
  CP.brand = "ford"
  assert is_mads_available(CP)

  CP.dashcamOnly = True
  assert not is_mads_available(CP)
  CP.dashcamOnly = False

  for brand in ("hyundai", "mock", ""):
    CP.brand = brand
    assert not is_mads_available(CP)


def test_mads_configuration_and_lateral_only_state_use_existing_car_params():
  CP = car.CarParams.new_message()
  CP.brand = "ford"
  CP.pcmCruise = True
  CS = car.CarState.new_message()
  CS.cruiseState.available = True

  assert not is_mads_configured(CP)
  assert not is_mads_lateral_only(CP, CS)

  CP.alternativeExperience = ALTERNATIVE_EXPERIENCE.ENABLE_MADS
  assert is_mads_configured(CP)
  assert is_mads_lateral_only(CP, CS)

  CS.cruiseState.enabled = True
  assert not is_mads_lateral_only(CP, CS)
