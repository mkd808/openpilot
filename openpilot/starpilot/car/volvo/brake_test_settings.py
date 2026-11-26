"""Parked preference for the Volvo V40 brake test (brake-test branch only).

Backed by the flag file the car controller checks about once a second (CarControllerParams.C1_BRAKE_TEST_FLAG), so it
takes effect without a restart. Shown only when the car runs with the C1_BRAKE_TEST safety flag, i.e. on the brake-test
branch with alpha longitudinal. Turning it on needs the car parked; turning it off works any time."""

import os

from opendbc.car.volvo.values import CAR, CarControllerParams, VolvoSafetyFlags
from openpilot.starpilot.ui.feature_settings_state import FeatureRow
from openpilot.starpilot.vehicle_selection import read_selection

KEY = 'VolvoC1BrakeTest'
LABEL = 'Brake Test (LKA button)'
DESCRIPTION = ("Test only. With OP long active above 30 km/h and no lead, flipping the LKA button makes OP brake at " +
               "-1.0 m/s^2 for up to 4 s or 15 km/h. May switch stock ACC off until the next ignition cycle.")
FLAG = CarControllerParams.C1_BRAKE_TEST_FLAG


def enabled(path: str = FLAG) -> bool:
  try:
    return os.path.isfile(path)
  except OSError:
    return False


def capability(owner):
  cp = owner.vehicle_params()
  if cp is None or cp.brand != 'volvo' or cp.carFingerprint != CAR.VOLVO_V40 or not cp.openpilotLongitudinalControl:
    return None
  if not any(cfg.safetyParam & VolvoSafetyFlags.C1_BRAKE_TEST for cfg in cp.safetyConfigs):
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


def apply(owner, request, path: str = FLAG) -> bool:
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
