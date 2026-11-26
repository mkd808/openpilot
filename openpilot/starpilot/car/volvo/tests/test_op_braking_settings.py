from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from opendbc.car.volvo.interface import CarInterface
from opendbc.car.volvo.values import CAR
from openpilot.starpilot.car.volvo import op_braking_settings as bt
from openpilot.starpilot.ui.vehicle_bool import VEHICLE_BOOL_KEYS, confirmation_question


def v40_params(alpha: bool):
  cp = CarInterface.get_non_essential_params(CAR.VOLVO_V40)
  return CarInterface._get_params(cp, CAR.VOLVO_V40, {0: {}, 1: {}, 2: {}}, [], alpha, False, False)


class Owner:
  def __init__(self, cp, parked=True):
    self.cp, self.parked = cp, parked
    self.params = None

  def vehicle_params(self):
    return self.cp

  def vehicle_fingerprint(self):
    return str(self.cp.carFingerprint)

  def authority(self, name):
    return name == 'parked_preferences' and self.parked


@dataclass
class Request:
  value: str
  capability: object
  vehicle_fingerprint: str
  expected: bytes | None = None
  confirmation: bool = True
  dependencies: tuple = ()
  key: str = bt.KEY


@pytest.fixture
def selection(monkeypatch):
  monkeypatch.setattr(bt, 'read_selection', lambda params: SimpleNamespace(readable=True, valid=True, raw=b'v40'))


def test_only_on_the_brake_test_branch_with_alpha_long(selection):
  assert bt.capability(Owner(v40_params(alpha=True))) is not None  # safety param 50 on this branch
  assert bt.capability(Owner(v40_params(alpha=False))) is None


def test_toggle_writes_and_removes_the_flag_file(selection, tmp_path):
  flag = str(tmp_path / 'volvo_c1_op_braking')
  owner = Owner(v40_params(alpha=True))
  cap = bt.capability(owner)
  assert bt.apply(owner, Request('On', cap, owner.vehicle_fingerprint(), b'0'), flag) and bt.enabled(flag)
  assert bt.apply(owner, Request('Off', cap, owner.vehicle_fingerprint(), b'1'), flag) and not bt.enabled(flag)


def test_on_only_while_parked_off_any_time(selection, tmp_path):
  flag = str(tmp_path / 'volvo_c1_op_braking')
  driving = Owner(v40_params(alpha=True), parked=False)
  cap = bt.capability(driving)
  assert not bt.apply(driving, Request('On', cap, driving.vehicle_fingerprint(), b'0'), flag) and not bt.enabled(flag)
  open(flag, 'a').close()
  assert bt.apply(driving, Request('Off', cap, driving.vehicle_fingerprint(), b'1'), flag) and not bt.enabled(flag)


def test_needs_confirmation_and_matching_capability(selection, tmp_path):
  flag = str(tmp_path / 'volvo_c1_op_braking')
  owner = Owner(v40_params(alpha=True))
  cap = bt.capability(owner)
  assert not bt.apply(owner, Request('On', cap, owner.vehicle_fingerprint(), b'0', confirmation=False), flag)
  assert not bt.apply(owner, Request('On', ('other',), owner.vehicle_fingerprint(), b'0'), flag)
  assert not bt.enabled(flag)


def test_registered_with_a_confirmation_question():
  assert bt.KEY in VEHICLE_BOOL_KEYS
  for value in ('On', 'Off'):
    assert 'OP braking' in confirmation_question(SimpleNamespace(key=bt.KEY, value=value, vehicle_fingerprint='VOLVO_V40'))
