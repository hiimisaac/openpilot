from dataclasses import dataclass
import math

from opendbc.can.dbc import DBC
from opendbc.can.parser import get_raw_value


_LMC2_MESSAGE = DBC("ford_lincoln_base_pt").name_to_msg["LateralMotionControl2"]


@dataclass(frozen=True)
class Lmc2Command:
  path_offset: float = 0.0
  path_angle: float = 0.0
  curvature: float = 0.0
  curvature_rate: float = 0.0


def polynomial_state(command: Lmc2Command, distance: float) -> tuple[float, float, float]:
  """Return lateral position, heading, and curvature at path distance."""
  distance = max(_finite(distance), 0.0)
  path_offset = command.path_offset + command.path_angle * distance + \
                0.5 * command.curvature * distance ** 2 + command.curvature_rate * distance ** 3 / 6.0
  path_angle = command.path_angle + command.curvature * distance + \
               0.5 * command.curvature_rate * distance ** 2
  curvature = command.curvature + command.curvature_rate * distance
  return path_offset, path_angle, curvature


def _finite(value: float) -> float:
  return float(value) if math.isfinite(value) else 0.0


def _decode_signal(dat: bytes, signal_name: str) -> float:
  signal = _LMC2_MESSAGE.sigs[signal_name]
  raw = get_raw_value(dat, signal)
  if signal.is_signed:
    raw -= ((raw >> (signal.size - 1)) & 1) * (1 << signal.size)
  return raw * signal.factor + signal.offset


def decode_lmc2_command(dat: bytes) -> Lmc2Command:
  """Decode all four polynomial coefficients from a transmitted LMC2 frame."""
  return Lmc2Command(
    path_offset=_decode_signal(dat, "LatCtlPathOffst_L_Actl"),
    path_angle=_decode_signal(dat, "LatCtlPath_An_Actl"),
    curvature=_decode_signal(dat, "LatCtlCurv_No_Actl"),
    curvature_rate=_decode_signal(dat, "LatCtlCrv_NoRate2_Actl"),
  )
