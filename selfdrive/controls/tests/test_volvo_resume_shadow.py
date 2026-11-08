from openpilot.selfdrive.controls.lib.volvo_resume_shadow import (
  BASELINE_S, LEAD_GAP_GROW_M, STOCK_GRACE_S, VolvoResumeShadow,
)

DT = 0.05


def run(steps):
  """steps: iterable of (standstill, acc, lead_ok, d_rel, v_lead, resume_pressed); returns all events with time."""
  shadow = VolvoResumeShadow()
  out = []
  for i, s in enumerate(steps):
    for ev in shadow.update(i * DT, *s):
      out.append((round(i * DT, 2), ev))
  return out


def held(seconds, d=4.0, v=0.0, lead=True, pressed=False):
  return [(True, True, lead, d, v, pressed)] * int(seconds / DT)


def leaving(seconds, d0=4.0, speed=1.5):
  return [(True, True, True, d0 + speed * i * DT, speed, False) for i in range(int(seconds / DT))]


def moving(seconds=1.0):
  return [(False, True, True, 12.0, 3.0, False)] * int(seconds / DT)


def kinds(events):
  return [e["ev"] for _, e in events]


def test_long_stop_lead_leaves_driver_presses_resume():
  steps = held(8.0) + leaving(3.5) + held(0.2, d=9.0, v=1.5, pressed=True) + moving()
  ev = run(steps)
  assert kinds(ev) == ["baseline", "lead_moved", "would_resume", "stop_end"]
  end = ev[-1][1]
  assert end["outcome"] == "driver_resume"
  assert end["would_resume"]
  assert end["button_after_lead_s"] is not None


def test_would_resume_waits_for_stock_grace():
  ev = run(held(3.0) + leaving(STOCK_GRACE_S - 0.3) + moving())
  assert "would_resume" not in kinds(ev)
  assert ev[-1][1]["outcome"] == "acc_auto_resume"
  assert not ev[-1][1]["would_resume"]


def test_short_stop_acc_resumes_itself():
  ev = run(held(2.0) + leaving(1.4) + moving())
  assert kinds(ev) == ["baseline", "lead_moved", "stop_end"]
  assert ev[-1][1]["outcome"] == "acc_auto_resume"


def test_lead_never_leaves():
  ev = run(held(5.0) + moving())
  assert kinds(ev) == ["baseline", "stop_end"]
  assert ev[-1][1]["outcome"] == "no_lead_departure"


def test_no_lead_means_no_baseline_and_no_would_resume():
  ev = run(held(6.0, lead=False) + leaving(3.0) + moving())
  assert "baseline" not in kinds(ev) and "would_resume" not in kinds(ev)


def test_moving_lead_is_not_a_baseline():
  ev = run(held(3.0, v=1.0))
  assert kinds(ev) == []


def test_gap_noise_below_threshold_does_not_count_as_leaving():
  ev = run(held(2.0) + held(3.0, d=4.0 + LEAD_GAP_GROW_M - 0.2, v=1.5) + moving())
  assert "lead_moved" not in kinds(ev)


def test_acc_off_ends_the_stop():
  steps = held(3.0) + [(True, False, True, 4.0, 0.0, False)] * 5
  ev = run(steps)
  assert kinds(ev) == ["baseline", "stop_end"]
  assert ev[-1][1]["outcome"] == "acc_off"


def test_acc_off_stop_is_not_tracked():
  assert run([(True, False, True, 4.0, 0.0, False)] * 200) == []


def test_stop_shorter_than_baseline_is_not_logged():
  assert run(held(BASELINE_S - 0.3) + moving()) == []
