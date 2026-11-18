"""controlsd's V40 Resume glue, read from real cereal messages.

test_volvo_resume_shadow.py drives VolvoResumeShadow with plain values, so it cannot catch controlsd reading a
radarState field that no longer exists (LeadData.status was renamed to present). Here the inputs are real capnp
readers, as controlsd sees them on the car."""
from types import SimpleNamespace

import openpilot.cereal.messaging as messaging
from opendbc.car.structs import car
from openpilot.selfdrive.controls import controlsd
from openpilot.selfdrive.controls.controlsd import Controls
from openpilot.selfdrive.controls.lib.volvo_resume_shadow import ResumeAssistFlag, VolvoResumeShadow

DT = 0.05


class FakeSubMaster:
  def __init__(self):
    self.seen = {'radarState': True}
    self.data = {}

  def __getitem__(self, key):
    return self.data[key]


def radar_state(d_rel, v_lead, present=True, radar=True, track=7):
  msg = messaging.new_message('radarState')
  lead = msg.radarState.leadOne
  lead.present = present
  lead.dRel = d_rel
  lead.vLead = v_lead
  lead.radar = radar
  lead.radarTrackId = track
  return msg.as_reader().radarState


def car_state(standstill, v_ego=0.):
  msg = messaging.new_message('carState')
  cs = msg.carState
  cs.standstill = standstill
  cs.cruiseState.enabled = True
  cs.vEgo = v_ego
  return msg.as_reader().carState


def run_stop(monkeypatch, tmp_path, radar=True):
  clock = SimpleNamespace(now=0.)
  monkeypatch.setattr(controlsd.time, 'monotonic', lambda: clock.now)
  logged = []
  monkeypatch.setattr(controlsd.cloudlog, 'info', logged.append)

  fake = SimpleNamespace(sm=FakeSubMaster(), volvo_resume_shadow=VolvoResumeShadow(),
                         volvo_resume_assist_flag=ResumeAssistFlag(str(tmp_path / 'off')))

  # Held at a standstill behind a lead, the lead pulls away, then the car moves off
  steps = [(True, 4.0, 0.0)] * int(8.0 / DT)
  steps += [(True, 4.0 + 1.5 * i * DT, 1.5) for i in range(int(3.5 / DT))]
  steps += [(False, 12.0, 3.0)] * int(1.0 / DT)
  for i, (standstill, d_rel, v_lead) in enumerate(steps):
    clock.now = i * DT
    fake.sm.data['radarState'] = radar_state(d_rel, v_lead, radar=radar)
    CC = car.CarControl()
    Controls.update_volvo_resume(fake, CC, car_state(standstill, 0. if standstill else 2.))
    assert fake.volvo_resume_shadow is not None, "update_volvo_resume hit an exception and disabled the shadow"
  return [m for m in logged if m.startswith("volvo_resume_shadow ")]


def test_real_radar_lead_keeps_shadow_running(monkeypatch, tmp_path):
  events = run_stop(monkeypatch, tmp_path)
  assert any('"lead_moved"' in e for e in events)
  assert all('"radar": true' in e and '"track": 7' in e for e in events)


def test_vision_only_lead_has_no_track(monkeypatch, tmp_path):
  events = run_stop(monkeypatch, tmp_path, radar=False)
  assert events
  assert all('"radar": false' in e and '"track": -1' in e for e in events)
