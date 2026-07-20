from dataclasses import dataclass
import math

from opendbc.can.dbc import DBC
from opendbc.can.parser import get_raw_value


LMC2_DT = 0.05
LMC2_C0_LIMITS = (-4.61, 4.60)
LMC2_C1_LIMITS = (-0.475, 0.497)
LMC2_C2_LIMITS = (-0.02, 0.02)
LMC2_C3_LIMITS = (-0.001024, 0.001023)
LMC2_C2_FULL_CURVATURE = 0.006
LMC2_C2_ZERO_CURVATURE = 0.012
LMC2_MIN_LOOKAHEAD = 7.0
LMC2_C0_LOOKAHEAD = 7.0

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


def _clip(value: float, limits: tuple[float, float]) -> float:
  return min(max(value, limits[0]), limits[1])


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


class SimpleLmc2Controller:
  """Encode one desired local curve into Ford's four LMC2 coefficients."""

  def __init__(self):
    self._last_desired_curvature: float | None = None

  def update(self, desired_curvature: float, measured_curvature: float, v_ego: float,
             active: bool, driver_override: bool) -> Lmc2Command:
    desired_curvature = _finite(desired_curvature)
    measured_curvature = _finite(measured_curvature)
    v_ego = max(_finite(v_ego), 0.0)

    if not active:
      self._last_desired_curvature = None
      return Lmc2Command()

    d_look = max(v_ego, LMC2_MIN_LOOKAHEAD)
    if driver_override:
      self._last_desired_curvature = measured_curvature
      return Lmc2Command(
        path_offset=_clip(0.5 * measured_curvature * LMC2_C0_LOOKAHEAD ** 2, LMC2_C0_LIMITS),
        path_angle=_clip(measured_curvature * d_look, LMC2_C1_LIMITS),
      )

    c2_share = _clip(
      (LMC2_C2_ZERO_CURVATURE - abs(desired_curvature)) /
      (LMC2_C2_ZERO_CURVATURE - LMC2_C2_FULL_CURVATURE),
      (0.0, 1.0),
    )
    delivered_curvature = 0.0
    if measured_curvature * desired_curvature > 0.0:
      delivered_curvature = math.copysign(
        min(abs(measured_curvature), abs(desired_curvature)),
        desired_curvature,
      )
    fast_curvature = desired_curvature - delivered_curvature * c2_share

    curvature_rate = 0.0
    if self._last_desired_curvature is not None and v_ego >= 1.0:
      curvature_rate = (desired_curvature - self._last_desired_curvature) / (v_ego * LMC2_DT)
    self._last_desired_curvature = desired_curvature

    return Lmc2Command(
      path_offset=_clip(0.5 * fast_curvature * LMC2_C0_LOOKAHEAD ** 2, LMC2_C0_LIMITS),
      path_angle=_clip(fast_curvature * d_look, LMC2_C1_LIMITS),
      curvature=_clip(desired_curvature * c2_share, LMC2_C2_LIMITS),
      curvature_rate=_clip(curvature_rate, LMC2_C3_LIMITS),
    )
