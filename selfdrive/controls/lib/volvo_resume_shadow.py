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
  stop_end      the hold ended, with how it ended: acc_auto_resume, driver_resume, no_lead_departure, acc_off
"""
import os
from dataclasses import dataclass

BASELINE_S = 1.0          # held this long before the lead gap is taken as the baseline
BASELINE_LEAD_SPEED = 0.3  # m/s, lead counts as stopped below this
LEAD_GAP_GROW_M = 0.4     # lead gap growth over the baseline that counts as pulling away
LEAD_MIN_SPEED = 0.5      # m/s, lead speed that counts as pulling away
LEAD_MOVED_CONFIRM_S = 0.2
STOCK_GRACE_S = 1.0       # time stock ACC gets to resume by itself before a sender would act
MIN_LOGGED_STOP_S = 1.0   # shorter stops are not logged


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


class VolvoResumeShadow:
  def __init__(self):
    self.stop: _Stop | None = None

  @property
  def resume_requested(self) -> bool:
    """True from would_resume until the hold ends: the lead is gone and stock ACC is still holding the car."""
    return self.stop is not None and self.stop.would_fired

  def update(self, t: float, standstill: bool, acc_enabled: bool, lead_ok: bool, d_rel: float, v_lead: float,
             resume_pressed: bool) -> list[dict]:
    events: list[dict] = []
    holding = standstill and acc_enabled

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
      dur = t - stop.t0
      if not acc_enabled:
        outcome = "acc_off"
      elif stop.button_seen:
        outcome = "driver_resume"
      elif stop.moved_t is not None:
        outcome = "acc_auto_resume"
      else:
        outcome = "no_lead_departure"
      if dur >= MIN_LOGGED_STOP_S:
        events.append({"ev": "stop_end", "stop_s": round(dur, 2), "outcome": outcome,
                       "lead_moved_s": None if stop.moved_t is None else round(stop.moved_t - stop.t0, 2),
                       "would_resume": stop.would_fired, "button_after_lead_s": stop.button_after_lead_s,
                       "moved_after_lead_s": None if stop.moved_t is None else round(t - stop.moved_t, 2)})
      self.stop = None

    return events
