import numpy as np

from openpilot.selfdrive.controls.lib.green_light_launch import GreenLightLaunch


T_IDXS = np.linspace(0.0, 10.0, 33)


def moving_plan():
  speeds = np.clip((T_IDXS - 1.5) * 1.2, 0.0, 4.0)
  accels = np.gradient(speeds, T_IDXS)
  return speeds, accels


def test_committed_model_motion_launches_from_standstill():
  launch = GreenLightLaunch()
  speeds, accels = moving_plan()

  held_accel = launch.update(-1.0, standstill=True, v_ego=0.0, enabled=True, should_stop=True,
                             model_speeds=speeds, model_accels=accels, t_idxs=T_IDXS,
                             action_t=0.4)
  launch_accel = launch.update(-1.0, standstill=True, v_ego=0.0, enabled=True, should_stop=False,
                               model_speeds=speeds, model_accels=accels, t_idxs=T_IDXS,
                               action_t=0.4)

  assert held_accel == -1.0
  assert 0.0 < launch_accel <= 1.5


def test_model_must_commit_to_moving_before_launch():
  launch = GreenLightLaunch()
  stopped_speeds = np.zeros_like(T_IDXS)
  stopped_accels = np.zeros_like(T_IDXS)

  output_accel = launch.update(-1.0, standstill=True, v_ego=0.0, enabled=True, should_stop=False,
                               model_speeds=stopped_speeds, model_accels=stopped_accels, t_idxs=T_IDXS,
                               action_t=0.4)

  assert output_accel == -1.0


def test_incomplete_model_plan_does_not_launch():
  launch = GreenLightLaunch()
  speeds, _ = moving_plan()

  output_accel = launch.update(-1.0, standstill=True, v_ego=0.0, enabled=True, should_stop=False,
                               model_speeds=speeds, model_accels=[], t_idxs=T_IDXS,
                               action_t=0.4)

  assert output_accel == -1.0


def test_launch_disarms_after_vehicle_is_moving():
  launch = GreenLightLaunch()
  speeds, accels = moving_plan()
  common = {
    "enabled": True, "should_stop": False, "model_speeds": speeds, "model_accels": accels,
    "t_idxs": T_IDXS, "action_t": 0.4,
  }

  launch.update(-1.0, standstill=True, v_ego=0.0, **common)
  moving_accel = launch.update(-1.0, standstill=False, v_ego=2.1, **common)

  assert moving_accel == -1.0
