import pyray as rl
from dataclasses import dataclass
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.car.mads import is_mads_configured
from openpilot.selfdrive.ui.onroad.exp_button import ExpButton
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget

# Constants
SET_SPEED_NA = 255
KM_TO_MILE = 0.621371
CRUISE_DISABLED_CHAR = '–'
GHOST_WHEEL_SCALE = 1.1


@dataclass(frozen=True)
class UIConfig:
  header_height: int = 300
  border_size: int = 30
  button_size: int = 192
  set_speed_width_metric: int = 200
  set_speed_width_imperial: int = 172
  wheel_icon_size: int = 144
  steering_wheel_size: int = 144


@dataclass(frozen=True)
class FontSizes:
  current_speed: int = 176
  speed_unit: int = 44
  max_speed: int = 34
  set_speed: int = 78
  mads: int = 42


@dataclass(frozen=True)
class Colors:
  WHITE = rl.WHITE
  DISENGAGED = rl.Color(145, 155, 149, 255)
  OVERRIDE = rl.Color(145, 155, 149, 255)  # Added
  ENGAGED = rl.Color(128, 216, 166, 255)
  DISENGAGED_BG = rl.Color(0, 0, 0, 153)
  OVERRIDE_BG = rl.Color(145, 155, 149, 204)
  ENGAGED_BG = rl.Color(128, 216, 166, 204)
  GREY = rl.Color(166, 166, 166, 255)
  DARK_GREY = rl.Color(114, 114, 114, 255)
  WHITE_TRANSLUCENT = rl.Color(255, 255, 255, 200)
  HEADER_GRADIENT_START = rl.Color(0, 0, 0, 114)
  HEADER_GRADIENT_END = rl.BLANK
  GHOST_WHEEL = rl.Color(70, 220, 255, 160)


UI_CONFIG = UIConfig()
FONT_SIZES = FontSizes()
COLORS = Colors()


class HudRenderer(Widget):
  def __init__(self):
    super().__init__()
    """Initialize the HUD renderer."""
    self.is_cruise_set: bool = False
    self.is_cruise_available: bool = True
    self.set_speed: float = SET_SPEED_NA
    self.speed: float = 0.0
    self.v_ego_cluster_seen: bool = False

    self._font_semi_bold: rl.Font = gui_app.font(FontWeight.SEMI_BOLD)
    self._font_bold: rl.Font = gui_app.font(FontWeight.BOLD)
    self._font_medium: rl.Font = gui_app.font(FontWeight.MEDIUM)
    self._font_normal: rl.Font = gui_app.font(FontWeight.NORMAL)

    self._exp_button: ExpButton = ExpButton(UI_CONFIG.button_size, UI_CONFIG.wheel_icon_size)
    self._txt_steering_wheel: rl.Texture = gui_app.texture(
      'icons/chffr_wheel.png', UI_CONFIG.steering_wheel_size, UI_CONFIG.steering_wheel_size,
    )
    self._steering_wheel_alpha_filter = FirstOrderFilter(0.0, 0.12, 1 / gui_app.target_fps)

  def _update_state(self) -> None:
    """Update HUD state based on car state and controls state."""
    sm = ui_state.sm
    if sm.recv_frame["carState"] < ui_state.started_frame:
      self.is_cruise_set = False
      self.set_speed = SET_SPEED_NA
      self.speed = 0.0
      return

    controls_state = sm['controlsState']
    car_state = sm['carState']

    v_cruise_cluster = car_state.vCruiseCluster
    self.set_speed = (
      controls_state.deprecated.vCruise if v_cruise_cluster == 0.0 else v_cruise_cluster
    )
    self.is_cruise_set = 0 < self.set_speed < SET_SPEED_NA
    self.is_cruise_available = self.set_speed != -1

    if self.is_cruise_set and not ui_state.is_metric:
      self.set_speed *= KM_TO_MILE

    v_ego_cluster = car_state.vEgoCluster
    self.v_ego_cluster_seen = self.v_ego_cluster_seen or v_ego_cluster != 0.0
    v_ego = v_ego_cluster if self.v_ego_cluster_seen else car_state.vEgo
    speed_conversion = CV.MS_TO_KPH if ui_state.is_metric else CV.MS_TO_MPH
    self.speed = max(0.0, v_ego * speed_conversion)

  def _render(self, rect: rl.Rectangle) -> None:
    """Render HUD elements to the screen."""
    # Draw the header background
    rl.draw_rectangle_gradient_v(
      int(rect.x),
      int(rect.y),
      int(rect.width),
      UI_CONFIG.header_height,
      COLORS.HEADER_GRADIENT_START,
      COLORS.HEADER_GRADIENT_END,
    )

    if self.is_cruise_available:
      self._draw_set_speed(rect)

    self._draw_current_speed(rect)
    self._draw_mads_status(rect)
    self._draw_steering_wheel(rect)

    button_x = rect.x + rect.width - UI_CONFIG.border_size - UI_CONFIG.button_size
    button_y = rect.y + UI_CONFIG.border_size
    self._exp_button.render(rl.Rectangle(button_x, button_y, UI_CONFIG.button_size, UI_CONFIG.button_size))

  def _draw_steering_wheel(self, rect: rl.Rectangle) -> None:
    """Show desired steering behind the measured wheel while lateral control is active."""
    sm = ui_state.sm
    angle_control = sm['controlsState'].lateralControlState.which() in ('angleState', 'pidState')
    messages_valid = all(sm.alive[s] and sm.valid[s] for s in ('carState', 'carControl', 'controlsState'))
    alpha = self._steering_wheel_alpha_filter.update(float(
      ui_state.ghost_wheel_enabled and sm['carControl'].latActive and angle_control and messages_valid,
    ))
    if alpha < 1e-2:
      return

    # Balance the driver-monitoring icon by placing the wheel on the opposite side.
    is_rhd = sm['driverMonitoringState'].isRHD
    edge_offset = UI_CONFIG.border_size + UI_CONFIG.button_size / 2
    pos_x = rect.x + edge_offset if is_rhd else rect.x + rect.width - edge_offset
    pos_y = rect.y + rect.height - edge_offset

    rl.draw_circle(int(pos_x), int(pos_y), UI_CONFIG.button_size / 2, rl.Color(0, 0, 0, int(70 * alpha)))

    wheel_txt = self._txt_steering_wheel
    src_rect = rl.Rectangle(0, 0, wheel_txt.width, wheel_txt.height)
    dest_rect = rl.Rectangle(pos_x, pos_y, wheel_txt.width, wheel_txt.height)
    origin = rl.Vector2(wheel_txt.width / 2, wheel_txt.height / 2)
    ghost_width = wheel_txt.width * GHOST_WHEEL_SCALE
    ghost_height = wheel_txt.height * GHOST_WHEEL_SCALE
    ghost_dest_rect = rl.Rectangle(pos_x, pos_y, ghost_width, ghost_height)
    ghost_origin = rl.Vector2(ghost_width / 2, ghost_height / 2)

    actual_angle = -sm['carState'].steeringAngleDeg
    desired_angle = -sm['carControl'].actuators.steeringAngleDeg

    ghost_color = rl.Color(COLORS.GHOST_WHEEL.r, COLORS.GHOST_WHEEL.g, COLORS.GHOST_WHEEL.b,
                           int(COLORS.GHOST_WHEEL.a * alpha))
    rl.draw_texture_pro(wheel_txt, src_rect, ghost_dest_rect, ghost_origin, desired_angle, ghost_color)
    rl.draw_texture_pro(wheel_txt, src_rect, dest_rect, origin, actual_angle, rl.fade(rl.WHITE, alpha))

  def user_interacting(self) -> bool:
    return self._exp_button.is_pressed

  def _draw_set_speed(self, rect: rl.Rectangle) -> None:
    """Draw a lightweight MAX speed indicator beside current speed."""
    set_speed_width = UI_CONFIG.set_speed_width_metric if ui_state.is_metric else UI_CONFIG.set_speed_width_imperial
    x = rect.x + 270 + (UI_CONFIG.set_speed_width_imperial - set_speed_width) // 2
    y = rect.y + 45

    max_color = COLORS.GREY
    set_speed_color = COLORS.DARK_GREY
    if self.is_cruise_set:
      set_speed_color = COLORS.WHITE
      if ui_state.status == UIStatus.ENGAGED:
        max_color = COLORS.ENGAGED
      elif ui_state.status == UIStatus.DISENGAGED:
        max_color = COLORS.DISENGAGED
      elif ui_state.status == UIStatus.OVERRIDE:
        max_color = COLORS.OVERRIDE

    max_text = tr("MAX")
    max_text_width = measure_text_cached(self._font_semi_bold, max_text, FONT_SIZES.max_speed).x
    rl.draw_text_ex(
      self._font_semi_bold,
      max_text,
      rl.Vector2(x + (set_speed_width - max_text_width) / 2, y + 27),
      FONT_SIZES.max_speed,
      0,
      max_color,
    )

    set_speed_text = CRUISE_DISABLED_CHAR if not self.is_cruise_set else str(round(self.set_speed))
    speed_text_width = measure_text_cached(self._font_normal, set_speed_text, FONT_SIZES.set_speed).x
    rl.draw_text_ex(
      self._font_normal,
      set_speed_text,
      rl.Vector2(x + (set_speed_width - speed_text_width) / 2, y + 77),
      FONT_SIZES.set_speed,
      0,
      set_speed_color,
    )

  def _draw_current_speed(self, rect: rl.Rectangle) -> None:
    """Draw the current vehicle speed and unit."""
    speed_text = str(round(self.speed))
    speed_text_size = measure_text_cached(self._font_normal, speed_text, FONT_SIZES.current_speed)
    speed_center_x = rect.x + 145
    speed_pos = rl.Vector2(speed_center_x - speed_text_size.x / 2, rect.y + 48)
    rl.draw_text_ex(self._font_normal, speed_text, speed_pos, FONT_SIZES.current_speed, 0, COLORS.WHITE)

    unit_text = tr("km/h") if ui_state.is_metric else tr("mph")
    unit_text_size = measure_text_cached(self._font_normal, unit_text, FONT_SIZES.speed_unit)
    unit_pos = rl.Vector2(speed_center_x - unit_text_size.x / 2, speed_pos.y + speed_text_size.y - 2)
    rl.draw_text_ex(self._font_normal, unit_text, unit_pos, FONT_SIZES.speed_unit, 0, COLORS.WHITE_TRANSLUCENT)

  def _draw_mads_status(self, rect: rl.Rectangle) -> None:
    if ui_state.CP is None or not is_mads_configured(ui_state.CP):
      return

    text = tr("MADS ON") if ui_state.sm["carControl"].latActive else tr("MADS OFF")
    text_size = measure_text_cached(self._font_normal, text, FONT_SIZES.mads)
    position = rl.Vector2(rect.x + rect.width - text_size.x - UI_CONFIG.border_size,
                          rect.y + UI_CONFIG.header_height + UI_CONFIG.border_size)
    rl.draw_text_ex(self._font_normal, text, position, FONT_SIZES.mads, 0, COLORS.WHITE)
