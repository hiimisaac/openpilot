import openpilot.cereal.messaging as messaging

from types import SimpleNamespace
from opendbc.car.ford.interface import CarInterface as FordCarInterface
from opendbc.car.ford.values import CAR as FORD
from opendbc.car.toyota.values import CAR as TOYOTA
from openpilot.selfdrive.controls.radard import get_lead, should_use_low_speed_override
from openpilot.selfdrive.test.process_replay import replay_process_with_name


class FakeLowSpeedTrack:
  dRel = 5.0

  def potential_low_speed_lead(self, v_ego):
    return True

  def get_RadarState(self, model_prob=0.0):
    return {"status": True, "radar": True, "dRel": self.dRel, "modelProb": model_prob}


class TestLeads:
  def test_ford_canfd_disables_low_speed_radar_only_lead(self):
    CP = FordCarInterface.get_non_essential_params(FORD.FORD_F_150_LIGHTNING_MK1)

    assert not should_use_low_speed_override(CP)

    lead = get_lead(1.0, True, {0: FakeLowSpeedTrack()}, SimpleNamespace(), 1.0, 0.0, low_speed_override=False)
    assert not lead["status"]

  def test_radar_fault(self):
    # if there's no radar-related can traffic, radard should either not respond or respond with an error
    # this is tightly coupled with underlying car radar_interface implementation, but it's a good sanity check
    def single_iter_pkg():
      # single iter package, with meaningless cans and empty carState/modelV2
      msgs = []
      for _ in range(500):
        can = messaging.new_message("can", 1)
        cs = messaging.new_message("carState")
        cp = messaging.new_message("carParams")
        msgs.append(can.as_reader())
        msgs.append(cs.as_reader())
        msgs.append(cp.as_reader())
      model = messaging.new_message("modelV2")
      msgs.append(model.as_reader())

      return msgs

    msgs = [m for _ in range(3) for m in single_iter_pkg()]
    out = replay_process_with_name("card", msgs, fingerprint=TOYOTA.TOYOTA_COROLLA_TSS2)
    states = [m for m in out if m.which() == "liveTracks"]
    failures = [not state.valid for state in states]

    assert len(states) == 0 or all(failures)
