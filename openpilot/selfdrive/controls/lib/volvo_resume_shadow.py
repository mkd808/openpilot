"""Volvo V40 stop-and-go resume timing. The logging always runs; the optional Resume send is switched by the flag
file below and handled in controlsd, this class only decides when a Resume would be due.

Stock ACC holds the car at a standstill behind a lead. If the lead sits for a few seconds, stock ACC stops
resuming by itself and waits for the driver's Resume press. This class watches for that situation and logs
what a Resume sender would have done, so the timing can be checked against real drives before any button
frame is sent.

Events (dicts, the caller logs them):
  baseline      lead gap taken once the car has been held for BASELINE_S
  lead_moved    lead pulled away from the baseline gap and is moving
  would_resume  lead has been gone for STOCK_GRACE_S and the car is still held: a sender would fire here
  stop_end      the hold ended, with how it ended: acc_auto_resume, driver_resume, no_lead_departure, acc_off,
                launch_failed (the car came off the hold but never got going, or stopped again), driver_gas (the
                driver took over with the gas pedal while the car was leaving).
                Only a sustained speed counts as moving off: a standstill flag that drops for a moment at ~0.2 m/s
                is not a launch. Needs v_ego; without it the old behaviour (hold end = moved) is kept.
"""
import os
from dataclasses import dataclass

BASELINE_S = 1.0          # held this long before the lead gap is taken as the baseline
BASELINE_LEAD_SPEED = 0.3  # m/s, lead counts as stopped below this
LEAD_GAP_GROW_M = 0.4     # lead gap growth over the baseline that counts as pulling away
LEAD_MIN_SPEED = 1.0      # m/s, lead speed that counts as pulling away (a creep of ~0.6 m/s is not a departure)
LEAD_MOVED_CONFIRM_S = 0.2
STOCK_GRACE_S = 0.4       # time stock ACC gets to resume by itself before a sender would act
MIN_LOGGED_STOP_S = 1.0   # shorter stops are not logged
LAUNCH_MIN_SPEED = 0.55   # m/s (2 km/h), speed that counts as really moving off
LAUNCH_CONFIRM_S = 1.0    # the speed must hold this long
LAUNCH_WATCH_S = 3.0      # after the hold ends, how long to wait for a launch before calling it failed


# The Resume send is on only while this file exists (touch it over SSH, remove it to turn it off). A Params key would
# need the prebuilt params library rebuilt, so a file is used until that is done.
RESUME_ASSIST_FLAG = "/data/volvo_c1_resume_assist"
FLAG_CHECK_S = 1.0


class ResumeAssistFlag:
  def __init__(self, path: str = RESUME_ASSIST_FLAG):
    self.path = path
    self._checked_at: float | None = None
    self._on = False

  def enabled(self, now: float) -> bool:
    if self._checked_at is None or now - self._checked_at >= FLAG_CHECK_S:
      self._checked_at = now
      try:
        self._on = os.path.isfile(self.path)
      except OSError:
        self._on = False
    return self._on


@dataclass
class _Stop:
  t0: float
  d0: float | None = None
  candidate_t: float | None = None
  moved_t: float | None = None
  would_fired: bool = False
  button_seen: bool = False
  button_after_lead_s: float | None = None
  end_t: float | None = None       # the hold ended here and the launch is being watched
  move_since: float | None = None
  max_v: float = 0.0


class VolvoResumeShadow:
  def __init__(self):
    self.stop: _Stop | None = None

  @property
  def resume_requested(self) -> bool:
    """True from would_resume until the hold ends: the lead is gone and stock ACC is still holding the car."""
    return self.stop is not None and self.stop.would_fired

  def _finish(self, stop: _Stop, t: float, outcome: str, events: list[dict]) -> None:
    end_t = stop.end_t if stop.end_t is not None else t
    dur = end_t - stop.t0
    if dur >= MIN_LOGGED_STOP_S:
      events.append({"ev": "stop_end", "stop_s": round(dur, 2), "outcome": outcome,
                     "lead_moved_s": None if stop.moved_t is None else round(stop.moved_t - stop.t0, 2),
                     "would_resume": stop.would_fired, "button_after_lead_s": stop.button_after_lead_s,
                     "moved_after_lead_s": None if stop.moved_t is None else round(end_t - stop.moved_t, 2),
                     "max_v": round(stop.max_v, 2)})
    self.stop = None

  def update(self, t: float, standstill: bool, acc_enabled: bool, lead_ok: bool, d_rel: float, v_lead: float,
             resume_pressed: bool, v_ego: float | None = None, gas_pressed: bool = False) -> list[dict]:
    events: list[dict] = []
    holding = standstill and acc_enabled

    stop = self.stop
    if stop is not None and stop.end_t is not None:
      # The hold ended and the launch is being watched
      if v_ego is not None:
        stop.max_v = max(stop.max_v, v_ego)
      base = "driver_resume" if stop.button_seen else "acc_auto_resume"
      if not acc_enabled:
        self._finish(stop, t, "acc_off", events)
      elif gas_pressed:
        self._finish(stop, t, "driver_gas", events)
      elif v_ego is not None and v_ego >= LAUNCH_MIN_SPEED and not standstill:
        if stop.move_since is None:
          stop.move_since = t
        if t - stop.move_since >= LAUNCH_CONFIRM_S:
          self._finish(stop, t, base, events)
      else:
        stop.move_since = None
        if holding or t - stop.end_t >= LAUNCH_WATCH_S:
          self._finish(stop, t, "launch_failed", events)
      if self.stop is not None:
        return events

    if holding and self.stop is None:
      self.stop = _Stop(t0=t)

    stop = self.stop
    if stop is not None and holding:
      dur = t - stop.t0

      if resume_pressed and not stop.button_seen:
        stop.button_seen = True
        if stop.moved_t is not None:
          stop.button_after_lead_s = round(t - stop.moved_t, 2)

      if stop.d0 is None:
        if dur >= BASELINE_S and lead_ok and abs(v_lead) < BASELINE_LEAD_SPEED:
          stop.d0 = d_rel
          events.append({"ev": "baseline", "stop_s": round(dur, 2), "d0": round(d_rel, 2)})
      else:
        pulling_away = lead_ok and (d_rel - stop.d0) > LEAD_GAP_GROW_M and v_lead > LEAD_MIN_SPEED
        if pulling_away:
          if stop.candidate_t is None:
            stop.candidate_t = t
          if stop.moved_t is None and t - stop.candidate_t >= LEAD_MOVED_CONFIRM_S:
            stop.moved_t = t
            events.append({"ev": "lead_moved", "stop_s": round(dur, 2), "gap": round(d_rel - stop.d0, 2),
                           "v_lead": round(v_lead, 2)})
        else:
          stop.candidate_t = None

        if stop.moved_t is not None and not stop.would_fired and t - stop.moved_t >= STOCK_GRACE_S:
          stop.would_fired = True
          events.append({"ev": "would_resume", "stop_s": round(dur, 2), "since_lead_moved_s": round(t - stop.moved_t, 2)})

    if stop is not None and not holding:
      if not acc_enabled:
        self._finish(stop, t, "acc_off", events)
      elif stop.button_seen or stop.moved_t is not None:
        if v_ego is None:
          self._finish(stop, t, "driver_resume" if stop.button_seen else "acc_auto_resume", events)
        else:
          # Wait for a real launch before calling it one
          stop.end_t = t
          stop.max_v = max(stop.max_v, v_ego)
          if v_ego >= LAUNCH_MIN_SPEED and not standstill:
            stop.move_since = t
          if gas_pressed:
            self._finish(stop, t, "driver_gas", events)
      else:
        self._finish(stop, t, "no_lead_departure", events)

    return events
