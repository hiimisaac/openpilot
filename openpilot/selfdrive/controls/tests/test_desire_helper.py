from openpilot.cereal import log
from openpilot.common.constants import CV
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper


class FakeParams:
  def __init__(self, enabled=True, speed_mph=19.0):
    self.enabled = enabled
    self.speed_mph = speed_mph

  def get_bool(self, key):
    assert key == "LaneTurnDesire"
    return self.enabled

  def get(self, key, return_default=False):
    assert key == "LaneTurnValue"
    assert return_default
    return self.speed_mph


class DummyCarState:
  def __init__(self, v_ego=10 * CV.MPH_TO_MS, left_blinker=False, right_blinker=False,
               left_blindspot=False, right_blindspot=False, steering_pressed=False, steering_torque=0.0):
    self.vEgo = v_ego
    self.leftBlinker = left_blinker
    self.rightBlinker = right_blinker
    self.leftBlindspot = left_blindspot
    self.rightBlindspot = right_blindspot
    self.steeringPressed = steering_pressed
    self.steeringTorque = steering_torque


class TestTurnDesire(OpenpilotTestCase):
  def test_low_speed_blinker_requests_turn(self):
    helper = DesireHelper(params=FakeParams())

    helper.update(DummyCarState(left_blinker=True), lateral_active=True, lane_change_prob=1.0)

    assert helper.desire == log.Desire.turnLeft

  def test_turn_desire_conditions(self):
    cases = (
      (DummyCarState(left_blinker=True), log.Desire.turnLeft),
      (DummyCarState(right_blinker=True), log.Desire.turnRight),
      (DummyCarState(v_ego=20 * CV.MPH_TO_MS, left_blinker=True), log.Desire.none),
      (DummyCarState(left_blinker=True, left_blindspot=True), log.Desire.none),
      (DummyCarState(right_blinker=True, right_blindspot=True), log.Desire.none),
      (DummyCarState(left_blinker=True, right_blinker=True), log.Desire.none),
      (DummyCarState(), log.Desire.none),
    )

    for carstate, expected in cases:
      with self.subTest(expected=expected):
        helper = DesireHelper(params=FakeParams())
        helper.update(carstate, lateral_active=True, lane_change_prob=1.0)
        assert helper.desire == expected

  def test_disabled(self):
    helper = DesireHelper(params=FakeParams(enabled=False))
    helper.update(DummyCarState(left_blinker=True), lateral_active=True, lane_change_prob=1.0)
    assert helper.desire == log.Desire.none

  def test_speed_is_capped_at_20_mph(self):
    helper = DesireHelper(params=FakeParams(speed_mph=30.0))
    helper.update(DummyCarState(v_ego=21 * CV.MPH_TO_MS, right_blinker=True), lateral_active=True, lane_change_prob=1.0)
    assert helper.desire == log.Desire.none

  def test_settings_refresh(self):
    params = FakeParams(enabled=False)
    helper = DesireHelper(params=params)
    params.enabled = True

    for _ in range(49):
      helper.update(DummyCarState(left_blinker=True), lateral_active=True, lane_change_prob=1.0)

    assert helper.desire == log.Desire.none
    helper.update(DummyCarState(left_blinker=True), lateral_active=True, lane_change_prob=1.0)
    assert helper.desire == log.Desire.turnLeft

  def test_normal_lane_change_still_works_above_turn_speed(self):
    helper = DesireHelper(params=FakeParams())
    carstate = DummyCarState(v_ego=25 * CV.MPH_TO_MS, left_blinker=True)
    helper.update(carstate, lateral_active=True, lane_change_prob=1.0)

    carstate.steeringPressed = True
    carstate.steeringTorque = 1.0
    helper.update(carstate, lateral_active=True, lane_change_prob=1.0)

    assert helper.desire == log.Desire.laneChangeLeft
