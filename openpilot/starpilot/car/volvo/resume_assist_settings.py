"""Parked preference for Volvo V40 Resume Assist.

Backed by a flag file, not a Params key (see volvo_resume_shadow.py): controlsd re-reads it about once a second, so it
takes effect without a restart. Turning it on needs the car parked; turning it off works any time."""

import os

from opendbc.car.volvo.values import CAR
from openpilot.selfdrive.controls.lib.volvo_resume_shadow import RESUME_ASSIST_FLAG
from openpilot.starpilot.ui.feature_settings_state import FeatureRow
from openpilot.starpilot.vehicle_selection import read_selection

KEY = 'VolvoC1ResumeAssist'
LABEL = 'Resume Assist'
DESCRIPTION = ("When stock ACC holds the car at a standstill and the lead pulls away, send Resume so you don't have to "
               "press it. Stock ACC still does all the accelerating.")


def enabled(path: str = RESUME_ASSIST_FLAG) -> bool:
  try:
    return os.path.isfile(path)
  except OSError:
    return False


def capability(owner):
  cp = owner.vehicle_params()
  if cp is None or cp.brand != 'volvo' or cp.carFingerprint != CAR.VOLVO_V40 or not cp.pcmCruise:
    return None
  selection = read_selection(owner.params)
  return (str(cp.carFingerprint), str(cp.brand), selection.raw) if selection.readable and selection.valid else None


def row(owner, parked: bool, cap) -> FeatureRow:
  on = enabled()
  # Off is always allowed; On only while parked
  allowed = on or (parked and owner.authority('parked_preferences'))
  return FeatureRow(KEY, LABEL, 'On' if on else 'Off', b'1' if on else b'0', ('Off', 'On'), available=allowed,
                    default_value='Off', capability=cap,
                    reason=DESCRIPTION if allowed else 'Can be turned on only while parked')


def apply(owner, request, path: str = RESUME_ASSIST_FLAG) -> bool:
  if not request.confirmation or request.value not in ('Off', 'On') or request.dependencies:
    return False
  turning_on = request.value == 'On'
  if (request.capability is None or request.capability != capability(owner) or
      request.vehicle_fingerprint != owner.vehicle_fingerprint() or
      request.expected not in (None, b'0', b'1') or (request.expected is not None and request.expected != (b'1' if enabled(path) else b'0'))):
    return False
  if turning_on and not owner.authority('parked_preferences'):
    return False
  try:
    if turning_on:
      with open(path, 'a'):
        pass
    else:
      try:
        os.remove(path)
      except FileNotFoundError:
        pass
  except OSError:
    return False
  return enabled(path) == turning_on
