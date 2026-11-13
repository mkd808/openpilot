from openpilot.selfdrive.controls.lib.volvo_resume_shadow import (
  BASELINE_S, FLAG_CHECK_S, LEAD_GAP_GROW_M, LEAD_MIN_SPEED, STOCK_GRACE_S, ResumeAssistFlag, VolvoResumeShadow,
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
  ev = run(held(3.0) + leaving(0.7) + moving())  # lead confirmed at ~0.55 s, would_resume due ~0.95 s
  assert "would_resume" not in kinds(ev)
  assert ev[-1][1]["outcome"] == "acc_auto_resume"
  assert not ev[-1][1]["would_resume"]


def test_short_stop_acc_resumes_itself():
  ev = run(held(2.0) + leaving(0.7) + moving())
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


def test_resume_requested_only_between_would_resume_and_hold_end():
  shadow = VolvoResumeShadow()
  seen = []
  steps = held(BASELINE_S + 0.5) + leaving(STOCK_GRACE_S + 1.5) + moving(0.5)
  for i, s in enumerate(steps):
    shadow.update(i * DT, *s)
    seen.append(shadow.resume_requested)
  assert not any(seen[:int((BASELINE_S + 0.5) / DT)])
  assert seen[int((BASELINE_S + 0.5 + STOCK_GRACE_S + 1.0) / DT)]
  assert not shadow.resume_requested  # hold ended, car moving


def test_resume_not_requested_when_lead_stays():
  shadow = VolvoResumeShadow()
  for i, s in enumerate(held(10.0)):
    shadow.update(i * DT, *s)
    assert not shadow.resume_requested


def test_resume_flag_follows_file_and_caches(tmp_path):
  path = tmp_path / "flag"
  flag = ResumeAssistFlag(str(path))
  assert not flag.enabled(0.0)
  path.write_text("")
  assert not flag.enabled(FLAG_CHECK_S / 2)  # cached
  assert flag.enabled(FLAG_CHECK_S + 0.1)
  path.unlink()
  assert flag.enabled(FLAG_CHECK_S + 0.2)  # cached
  assert not flag.enabled(2 * FLAG_CHECK_S + 0.2)


def test_slow_lead_creep_is_not_a_departure():
  # lead creeps forward at 0.6 m/s, enough to grow the gap but under the lead speed threshold
  assert 0.6 < LEAD_MIN_SPEED
  ev = run(held(3.0) + leaving(4.0, speed=0.6) + held(1.0, d=6.0))
  assert "lead_moved" not in kinds(ev)
  assert "would_resume" not in kinds(ev)


def test_lead_pulling_away_above_threshold_still_counts():
  ev = run(held(3.0) + leaving(2.0, speed=1.2) + moving())
  assert "lead_moved" in kinds(ev)
  assert "would_resume" in kinds(ev)


# Launch outcome: steps carry (standstill, acc, lead_ok, d_rel, v_lead, resume_pressed, v_ego, gas_pressed)
def with_speed(steps, v_ego=0.0, gas=False):
  return [(*s, v_ego, gas) for s in steps]


def blip(seconds, v=0.2):
  """standstill flag dropped but the car is barely moving"""
  return [(False, True, True, 9.0, 3.0, False, v, False)] * int(seconds / DT)


def driving(seconds, v=3.0, gas=False):
  return [(False, True, True, 12.0, 3.0, False, v, gas)] * int(seconds / DT)


def stopped_again(seconds):
  return [(True, True, True, 9.0, 3.0, False, 0.0, False)] * int(seconds / DT)


def ends(events):
  return [e for _, e in events if e["ev"] == "stop_end"]


def test_real_launch_is_confirmed_by_sustained_speed():
  ev = run(with_speed(held(3.0) + leaving(0.7)) + driving(2.0))
  assert ends(ev)[-1]["outcome"] == "acc_auto_resume"
  assert ends(ev)[-1]["max_v"] >= 3.0


def test_creeping_blip_then_stopped_again_is_a_failed_launch():
  # hold released, ~0.2 m/s for half a second, held again: this is what the 14 Nov route showed
  steps = with_speed(held(8.0) + leaving(3.5)) + blip(0.6) + stopped_again(2.0)
  ev = run(steps)
  first = ends(ev)[0]
  assert first["outcome"] == "launch_failed"
  assert first["max_v"] < 0.55
  assert first["lead_moved_s"] is not None


def test_failed_launch_is_followed_by_a_new_stop():
  ev = run(with_speed(held(8.0) + leaving(3.5)) + blip(0.6) + stopped_again(2.5) + driving(2.0))
  assert [e["outcome"] for e in ends(ev)] == ["launch_failed", "no_lead_departure"]


def test_never_getting_going_times_out_as_failed():
  ev = run(with_speed(held(3.0) + leaving(0.7)) + blip(4.0, v=0.3))
  assert ends(ev)[0]["outcome"] == "launch_failed"


def test_driver_gas_during_the_launch_is_not_credited_to_acc():
  steps = with_speed(held(8.0) + leaving(3.5)) + blip(0.6) + driving(2.0, v=3.0, gas=True)
  assert ends(run(steps))[0]["outcome"] == "driver_gas"


def test_driver_resume_still_reported_when_the_car_really_goes():
  steps = with_speed(held(8.0) + leaving(3.5) + held(0.2, d=9.0, v=1.5, pressed=True)) + driving(2.0)
  assert ends(run(steps))[-1]["outcome"] == "driver_resume"


def test_acc_off_during_the_launch_watch():
  steps = with_speed(held(3.0) + leaving(0.7)) + [(False, False, True, 9.0, 3.0, False, 0.0, False)] * 4
  assert ends(run(steps))[-1]["outcome"] == "acc_off"


def test_speed_must_hold_for_the_confirm_time():
  # a 0.5 s lurch above the threshold, then back to a crawl, is not a launch
  steps = with_speed(held(3.0) + leaving(0.7)) + driving(0.5, v=1.0) + blip(4.0, v=0.2)
  assert ends(run(steps))[0]["outcome"] == "launch_failed"
