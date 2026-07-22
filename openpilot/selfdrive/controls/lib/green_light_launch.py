import numpy as np

from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan


LAUNCH_DISARM_SPEED = 2.0
LAUNCH_COMMIT_T = 3.5
LAUNCH_MOVING_SPEED = 1.2
LAUNCH_MAX_ACCEL = 1.5


class GreenLightLaunch:
  def __init__(self):
    self.armed = False

  def update(self, output_accel, *, standstill, v_ego, enabled, should_stop,
             model_speeds, model_accels, t_idxs, action_t):
    if standstill:
      self.armed = True
    elif v_ego > LAUNCH_DISARM_SPEED:
      self.armed = False

    model_speeds = np.asarray(model_speeds)
    model_accels = np.asarray(model_accels)
    t_idxs = np.asarray(t_idxs)
    valid_plan = len(model_speeds) == len(model_accels) == len(t_idxs) and len(t_idxs) > 0
    committed_motion = valid_plan and np.interp(LAUNCH_COMMIT_T, t_idxs, model_speeds) > LAUNCH_DISARM_SPEED
    if not (self.armed and enabled and not should_stop and committed_motion):
      return output_accel

    moving_idx = int(np.argmax(model_speeds > LAUNCH_MOVING_SPEED))
    t_cut = min(float(t_idxs[moving_idx]), LAUNCH_COMMIT_T)
    t_shifted = t_idxs + t_cut
    v_shifted = np.interp(t_shifted, t_idxs, model_speeds)
    a_shifted = np.interp(t_shifted, t_idxs, model_accels)
    launch_accel = get_accel_from_plan(v_shifted, a_shifted, t_idxs, action_t=action_t)[0]
    launch_accel_max = np.interp(v_ego, [LAUNCH_MOVING_SPEED, LAUNCH_DISARM_SPEED], [LAUNCH_MAX_ACCEL, 0.0])
    return max(output_accel, min(launch_accel, launch_accel_max))
