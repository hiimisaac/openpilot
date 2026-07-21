from opendbc.car.structs import car


def is_mads_available(CP: car.CarParams) -> bool:
  return CP.brand == "ford" and not CP.dashcamOnly
