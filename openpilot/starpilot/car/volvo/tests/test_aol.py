"""V60 and V40 AOL: admission only for the exact profile, intent follows the car's state."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
from opendbc.car import structs
from opendbc.car.volvo.interface import CarInterface
from opendbc.car.volvo.values import CAR
from openpilot.starpilot.aol.intent import AOL_TOGGLE, ButtonType, read_settings
from openpilot.starpilot.car.volvo import aol


def params(identity=CAR.VOLVO_V60, alternative=32):
  cp = CarInterface.get_non_essential_params(identity)
  cp.alternativeExperience = alternative
  return cp


def test_only_the_v60_and_v40_are_admitted():
  for identity in (CAR.VOLVO_V60, CAR.VOLVO_V40):
    assert aol.qualified(params(identity), marked_only=True)
    assert aol.qualified(params(identity, alternative=0)) and not aol.qualified(params(identity, alternative=0), marked_only=True)
    assert aol.policy_for(params(identity)).intent_supported
  for identity in (CAR.VOLVO_XC40_RECHARGE, CAR.VOLVO_S60_RECHARGE, CAR.POLESTAR_2):
    assert not aol.qualified(params(identity))
    assert not aol.policy_for(params(identity)).intent_supported


def test_profile_must_match_the_platform():
  for identity, wrong in ((CAR.VOLVO_V60, 2), (CAR.VOLVO_V40, 8)):
    cp = params(identity)
    cp.safetyConfigs[0].safetyParam = wrong
    assert not aol.qualified(cp)


def test_v40_keeps_aol_with_op_long():
  # route 7c7d3d5e5ff31d51/00000021: OP long's profile (50) dropped AOL, so no steering without ACC engaged
  model = int(structs.CarParams.SafetyModel.volvo)
  cp = CarInterface.get_params(CAR.VOLVO_V40, {i: {} for i in range(8)}, [], True, False, False)
  cp.alternativeExperience = 32
  assert cp.openpilotLongitudinalControl and cp.safetyConfigs[0].safetyParam == 50
  assert aol.qualified(cp, marked_only=True) and aol.policy_for(cp).intent_supported
  for param in (18, 50):
    assert aol.native_profile_supported(model, param)
  assert aol.native_accepts_cp(cp, model, 50) and not aol.native_accepts_cp(cp, model, 2)
  for wrong in (16, 34, 48, 24):
    cp.safetyConfigs[0].safetyParam = wrong
    assert not aol.qualified(cp)
  v60 = params()
  v60.safetyConfigs[0].safetyParam = 8 | 16
  assert not aol.qualified(v60)


def test_exact_native_profile():
  model = int(structs.CarParams.SafetyModel.volvo)
  assert aol.native_profile_supported(model, 8)
  assert aol.native_profile_supported(model, 2)
  for param in (0, 1, 4, 5, 16, 24, 34):
    assert not aol.native_profile_supported(model, param)
  assert not aol.native_profile_supported(model + 1, 8)
  assert aol.native_accepts_cp(params(), model, 8)
  assert not aol.native_accepts_cp(params(), model, 2)
  assert aol.native_accepts_cp(params(CAR.VOLVO_V40), model, 2)
  assert not aol.native_accepts_cp(params(CAR.VOLVO_V40), model, 8)
  assert not aol.native_accepts_cp(params(alternative=0), model, 8)


@pytest.mark.parametrize('kwargs, restricted', [
  ({'gearShifter': structs.CarState.GearShifter.drive}, False),
  ({'gearShifter': structs.CarState.GearShifter.reverse}, True),
  ({'gearShifter': structs.CarState.GearShifter.drive, 'steerFaultTemporary': True}, True),
])
def test_temporary_restriction(kwargs, restricted):
  assert aol.temporary_restriction(params(), structs.CarState(canValid=True, **kwargs)) == restricted


def test_native_observation_waits_then_times_out():
  native = SimpleNamespace(requestedLateral=True, lateralAllowed=False)
  common = {'latched': True, 'panda_ready': True, 'restricted': False}
  assert aol.native_observation(native, now_ns=1, pending_since_ns=0, **common) == (1, False, False)
  assert aol.native_observation(native, now_ns=200_000_001, pending_since_ns=1, **common) == (1, False, True)
  assert aol.native_observation(None, now_ns=1, pending_since_ns=0, **common)[1]


class FakeParams:
  def get_param_path(self, key):
    return '/nonexistent/' + key

  def get_bool(self, key, **kwargs):
    return key == 'AlwaysOnLateral'

  def get(self, key, **kwargs):
    return None


def intent(lkas_action):
  return aol.VolvoEucdCardIntent(replace(read_settings(FakeParams()), lkas_action=lkas_action))


def step(card, *, available=False, lka=False, enabled=False, press=False):
  card.observe_vehicle_state(SimpleNamespace(eucd_lka_on=lka))
  state = structs.CarState(canValid=True, gearShifter=structs.CarState.GearShifter.drive)
  state.cruiseState.available = available
  state.cruiseState.enabled = enabled
  if press:
    state.buttonEvents = [structs.CarState.ButtonEvent(type=ButtonType.lkas, pressed=True)]
  card.update(state.as_reader(), fault_active=False, native_rejection_ns=0, now_ns=1, standard_enabled=False)
  return card.allowed_latch


def test_lkas_mapped_follows_the_lka_not_the_cruise():
  card = intent(AOL_TOGGLE)
  assert step(card, available=False, lka=True)
  assert not step(card, available=True, lka=False)
  assert step(card, available=True, lka=True)


def test_lka_pause_ends_with_the_next_engage():
  card = intent(AOL_TOGGLE)
  common = {'available': True}
  step(card, enabled=True, lka=True, **common)
  assert not card.pause_lateral
  step(card, enabled=True, lka=False, press=True, **common)  # LKA off while engaged pauses steering
  assert card.pause_lateral
  step(card, enabled=True, lka=False, **common)
  assert card.pause_lateral
  step(card, enabled=False, lka=False, **common)  # still paused with the ACC off
  assert card.pause_lateral
  step(card, enabled=True, lka=False, **common)  # the next engage steers again
  assert not card.pause_lateral


def test_lka_back_on_resumes_a_paused_steering():
  card = intent(AOL_TOGGLE)
  step(card, available=True, enabled=True, lka=True)
  step(card, available=True, enabled=True, lka=False, press=True)
  assert step(card, available=True, enabled=True, lka=True, press=True)
  assert not card.pause_lateral


def test_lka_off_while_not_engaged_is_no_pause():
  card = intent(AOL_TOGGLE)
  step(card, available=True, lka=True)
  assert not step(card, available=True, lka=False, press=True)
  assert not card.pause_lateral


def test_lka_off_engage_steers_with_the_latch_until_the_lka_pauses():
  # Route 00000007, 296 s: LKA off and ACC engaged steered with the AOL latch off, so the frame was pink
  card = intent(AOL_TOGGLE)
  assert not step(card, available=True, lka=False)
  assert step(card, available=True, enabled=True, lka=False)  # engaged: steering, latch on
  assert not step(card, available=True, lka=False)  # ACC off with the LKA off: no AOL
  assert step(card, available=True, enabled=True, lka=False)
  step(card, available=True, enabled=True, lka=True, press=True)  # LKA on: still steering
  assert step(card, available=True, enabled=True, lka=True)
  assert not step(card, available=True, enabled=True, lka=False, press=True)  # LKA off pauses
  assert card.pause_lateral


def test_lkas_unmapped_follows_the_acc_main():
  card = intent(0)
  assert not step(card, available=False, lka=True)
  assert step(card, available=True, lka=False)
  assert not step(card, available=False, lka=False)


def c1_step(card, *, available=False, lka=False, enabled=False):
  card.observe_vehicle_state(SimpleNamespace(c1_aol_allowed=lka and available))
  state = structs.CarState(canValid=True, gearShifter=structs.CarState.GearShifter.drive)
  state.cruiseState.available = available
  state.cruiseState.enabled = enabled
  card.update(state.as_reader(), fault_active=False, native_rejection_ns=0, now_ns=1, standard_enabled=False)
  return card.allowed_latch


def test_v40_intent_is_created_for_the_v40():
  settings = read_settings(FakeParams())
  assert isinstance(aol.create_intent(params(CAR.VOLVO_V40), settings), aol.VolvoC1CardIntent)
  assert isinstance(aol.create_intent(params(), settings), aol.VolvoEucdCardIntent)


def test_v40_latch_needs_the_lka_and_the_acc_main():
  card = aol.VolvoC1CardIntent(read_settings(FakeParams()))
  assert not c1_step(card)
  assert not c1_step(card, lka=True)
  assert not c1_step(card, available=True)
  assert c1_step(card, available=True, lka=True)
  assert not c1_step(card, available=True)  # LKA switched off
  assert not c1_step(card, lka=True)  # ACC main switched off


def test_v40_engaged_acc_keeps_the_latch_with_the_lka_off():
  card = aol.VolvoC1CardIntent(read_settings(FakeParams()))
  assert c1_step(card, available=True, enabled=True)
  assert not c1_step(card, available=True)
