from opendbc.car.structs import car

from openpilot.selfdrive.car.mads import is_mads_available


def test_mads_is_available_on_ford_only():
  CP = car.CarParams.new_message()
  CP.brand = "ford"
  assert is_mads_available(CP)

  CP.dashcamOnly = True
  assert not is_mads_available(CP)
  CP.dashcamOnly = False

  for brand in ("hyundai", "mock", ""):
    CP.brand = brand
    assert not is_mads_available(CP)
