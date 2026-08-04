import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("SCALE", "1")

import pyray as rl

from openpilot.selfdrive.ui.mici.onroad import hud_renderer as mici_hud
from openpilot.selfdrive.ui.onroad import hud_renderer as big_hud
from openpilot.selfdrive.ui.ui_state import UIStatus


class PassthroughFilter:
  def __init__(self):
    self.x = 0.0

  def update(self, value):
    self.x = float(value)
    return self.x


class FakeSubMaster(dict):
  def __init__(self, control_state: str = 'angleState', lat_active: bool = True):
    super().__init__({
      'carState': SimpleNamespace(steeringAngleDeg=-18.0),
      'carControl': SimpleNamespace(
        latActive=lat_active,
        actuators=SimpleNamespace(steeringAngleDeg=28.0),
      ),
      'controlsState': SimpleNamespace(
        lateralControlState=SimpleNamespace(which=lambda: control_state),
      ),
      'driverMonitoringState': SimpleNamespace(isRHD=False),
    })
    self.alive = dict.fromkeys(('carState', 'carControl', 'controlsState'), True)
    self.valid = self.alive.copy()


class TestGhostWheel(unittest.TestCase):
  def test_big_speed_and_unit_move_as_one_regular_weight_group(self):
    renderer = object.__new__(big_hud.HudRenderer)
    renderer.speed = 44.0
    renderer._font_normal = object()
    rect = rl.Rectangle(0, 0, 2160, 1080)

    with patch.object(big_hud.ui_state, 'is_metric', False), \
         patch.object(big_hud, 'measure_text_cached', side_effect=(rl.Vector2(120, 150), rl.Vector2(72, 50))), \
         patch.object(big_hud.rl, 'draw_text_ex') as draw_text:
      renderer._draw_current_speed(rect)

    self.assertEqual(draw_text.call_count, 2)
    speed_call, unit_call = draw_text.call_args_list
    self.assertIs(speed_call.args[0], renderer._font_normal)
    self.assertIs(unit_call.args[0], renderer._font_normal)
    self.assertLess(speed_call.args[2].x, rect.width * 0.1)
    self.assertLess(unit_call.args[2].x, rect.width * 0.1)
    self.assertGreater(unit_call.args[2].y, speed_call.args[2].y)
    self.assertLess(unit_call.args[3], speed_call.args[3] / 3)

  def test_big_set_speed_is_text_only_without_card(self):
    renderer = object.__new__(big_hud.HudRenderer)
    renderer.set_speed = 45.0
    renderer.is_cruise_set = True
    renderer._font_semi_bold = object()
    renderer._font_normal = object()

    with patch.object(big_hud.ui_state, 'is_metric', False), \
         patch.object(big_hud.ui_state, 'status', UIStatus.ENGAGED), \
         patch.object(big_hud, 'measure_text_cached', side_effect=(rl.Vector2(70, 34), rl.Vector2(80, 78))), \
         patch.object(big_hud.rl, 'draw_rectangle_rounded') as draw_card, \
         patch.object(big_hud.rl, 'draw_rectangle_rounded_lines_ex') as draw_card_border, \
         patch.object(big_hud.rl, 'draw_text_ex') as draw_text:
      renderer._draw_set_speed(rl.Rectangle(0, 0, 2160, 1080))

    draw_card.assert_not_called()
    draw_card_border.assert_not_called()
    self.assertEqual(draw_text.call_count, 2)
    self.assertIs(draw_text.call_args_list[-1].args[0], renderer._font_normal)

  def test_big_hud_draws_desired_before_actual(self):
    renderer = object.__new__(big_hud.HudRenderer)
    renderer._steering_wheel_alpha_filter = PassthroughFilter()
    renderer._txt_steering_wheel = SimpleNamespace(width=144, height=144)
    sm = FakeSubMaster()
    actual_color = object()

    with patch.object(big_hud.ui_state, 'sm', sm), \
         patch.object(big_hud.ui_state, 'ghost_wheel_enabled', True), \
         patch.object(big_hud.rl, 'draw_circle') as draw_circle, \
         patch.object(big_hud.rl, 'draw_texture_pro') as draw_texture, \
         patch.object(big_hud.rl, 'fade', return_value=actual_color):
      renderer._draw_steering_wheel(rl.Rectangle(0, 0, 2160, 1080))

    self.assertEqual(draw_texture.call_count, 2)
    self.assertEqual(draw_texture.call_args_list[0].args[4], -28.0)
    self.assertEqual(draw_texture.call_args_list[1].args[4], 18.0)
    ghost_dest = draw_texture.call_args_list[0].args[2]
    actual_dest = draw_texture.call_args_list[1].args[2]
    ghost_color = draw_texture.call_args_list[0].args[5]
    self.assertGreater(ghost_dest.width, actual_dest.width)
    self.assertGreater(ghost_dest.height, actual_dest.height)
    self.assertGreaterEqual(ghost_color.a, 128)
    self.assertGreater(ghost_color.b, ghost_color.r)
    self.assertIs(draw_texture.call_args_list[1].args[5], actual_color)
    draw_circle.assert_called_once()

  def test_big_hud_hides_unsupported_or_disabled_ghost(self):
    for control_state, lat_active, enabled in (
      ('torqueState', True, True),
      ('angleState', False, True),
      ('angleState', True, False),
    ):
      with self.subTest(control_state=control_state, lat_active=lat_active, enabled=enabled):
        renderer = object.__new__(big_hud.HudRenderer)
        renderer._steering_wheel_alpha_filter = PassthroughFilter()
        renderer._txt_steering_wheel = SimpleNamespace(width=144, height=144)
        sm = FakeSubMaster(control_state, lat_active)

        with patch.object(big_hud.ui_state, 'sm', sm), \
             patch.object(big_hud.ui_state, 'ghost_wheel_enabled', enabled), \
             patch.object(big_hud.rl, 'draw_circle') as draw_circle, \
             patch.object(big_hud.rl, 'draw_texture_pro') as draw_texture:
          renderer._draw_steering_wheel(rl.Rectangle(0, 0, 2160, 1080))

        draw_circle.assert_not_called()
        draw_texture.assert_not_called()

  def test_mici_hud_draws_desired_before_actual(self):
    renderer = object.__new__(mici_hud.HudRenderer)
    renderer._show_wheel_critical = False
    renderer._txt_wheel = SimpleNamespace(width=50, height=50)
    renderer._txt_wheel_critical = SimpleNamespace(width=50, height=50)
    renderer._wheel_alpha_filter = PassthroughFilter()
    renderer._wheel_y_filter = PassthroughFilter()
    renderer._ghost_wheel_alpha_filter = PassthroughFilter()
    renderer._turn_intent = SimpleNamespace(render=MagicMock())
    sm = FakeSubMaster()

    with patch.object(mici_hud.ui_state, 'sm', sm), \
         patch.object(mici_hud.ui_state, 'status', UIStatus.ENGAGED), \
         patch.object(mici_hud.ui_state, 'ghost_wheel_enabled', True), \
         patch.object(mici_hud.rl, 'draw_texture_pro') as draw_texture:
      renderer._draw_steering_wheel(rl.Rectangle(0, 0, 536, 240))

    self.assertEqual(draw_texture.call_count, 2)
    self.assertEqual(draw_texture.call_args_list[0].args[4], -28.0)
    self.assertEqual(draw_texture.call_args_list[1].args[4], 18.0)
    ghost_dest = draw_texture.call_args_list[0].args[2]
    actual_dest = draw_texture.call_args_list[1].args[2]
    ghost_color = draw_texture.call_args_list[0].args[5]
    self.assertGreater(ghost_dest.width, actual_dest.width)
    self.assertGreater(ghost_dest.height, actual_dest.height)
    self.assertGreaterEqual(ghost_color.a, 128)
    self.assertGreater(ghost_color.b, ghost_color.r)


if __name__ == '__main__':
  unittest.main()
