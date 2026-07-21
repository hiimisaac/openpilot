from opendbc.car.structs import car
from opendbc.safety import ALTERNATIVE_EXPERIENCE


def is_mads_available(CP: car.CarParams) -> bool:
  return CP.brand == "ford" and not CP.dashcamOnly


def is_mads_configured(CP: car.CarParams) -> bool:
  return is_mads_available(CP) and bool(CP.alternativeExperience & ALTERNATIVE_EXPERIENCE.ENABLE_MADS)


def is_mads_lateral_only(CP: car.CarParams, CS: car.CarState) -> bool:
  return is_mads_configured(CP) and CP.pcmCruise and CS.cruiseState.available and not CS.cruiseState.enabled
