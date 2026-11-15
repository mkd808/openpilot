"""V40 (C1) rules ported from the 2017volvov40port branch: stock ACC brake ownership, the stop-and-go Resume burst,
and the optional always on lateral profile."""
import unittest
from opendbc.can import CANPacker
from opendbc.safety.tests.libsafety import libsafety_py
from opendbc.safety.tests.test_volvo_c1 import packet

lib = libsafety_py.libsafety
packer = CANPacker('volvo_v40_2017_pt')
VOLVO = 36
C1 = 2
AOL = 32
LATERAL = 1
LONGITUDINAL = 2
DRIVE = 3
REVERSE = 1
STOCK_BTN = {"byte0": 0xF8, "byte1": 0xB1, "byte2": 0xA0, "byte3": 0x6F, "byte4": 0, "byte5": 0, "byte6": 0x10}


def msg(name, bus, values):
  return packet(packer.make_can_msg(name, bus, values))


def rx(name, bus, values):
  return lib.safety_rx_hook(msg(name, bus, values))


def tx(name, bus, values):
  return lib.safety_tx_hook(msg(name, bus, values))


class TestVolvoC1V40(unittest.TestCase):
  def setUp(self):
    lib.set_alternative_experience(0)
    self.assertEqual(0, lib.set_safety_hooks(VOLVO, C1))
    lib.init_tests()

  def _speed(self, kph):
    rx('VehicleSpeed1', 0, {'VehicleSpeed': kph})

  def _acc(self, active, main=True):
    rx('FSM0', 2, {'ACCStatusActive': int(active), 'ACCStatusOnOff': int(main)})

  def _brake_status(self, status, hold=0):
    rx('BrakeMessages', 0, {'BrakeStatus': status, 'BrakeHold': hold})

  def _pressure(self, active):
    rx('PedalandBrake', 0, {'BrakePedalActive2': int(active), 'BrakePedalActive': int(active)})

  # ---- brake ownership ----

  def test_brake_status_and_lka_bit_positions(self):
    self.assertEqual(5, packer.make_can_msg('BrakeMessages', 0, {'BrakeStatus': 5})[1][2] & 0x7)
    self.assertEqual(0x40, packer.make_can_msg('BrakeMessages', 0, {'BrakeHold': 1})[1][2] & 0x40)
    self.assertEqual(0x04, packer.make_can_msg('CEM0', 0, {'LKA_State': 1})[1][0])

  def test_acc_braking_masks_pressure_bits(self):
    lib.set_controls_allowed(True)
    self._speed(54)
    self._brake_status(5)
    self._pressure(True)
    self.assertFalse(lib.get_brake_pressed_prev())
    self.assertTrue(lib.get_controls_allowed())

  def test_pedal_without_acc_braking_drops_controls(self):
    for status in (0, 1):
      self.setUp()
      lib.set_controls_allowed(True)
      self._speed(54)
      self._brake_status(status)
      self._pressure(True)
      self.assertTrue(lib.get_brake_pressed_prev(), status)
      self.assertFalse(lib.get_controls_allowed(), status)

  def test_pedal_tap_during_acc_braking_drops_controls(self):
    lib.set_controls_allowed(True)
    self._speed(54)
    self._brake_status(5)
    self._pressure(True)
    self.assertTrue(lib.get_controls_allowed())
    self._brake_status(0)
    self.assertFalse(lib.get_controls_allowed())

  def test_acc_hold_and_transition_masked_only_while_acc_active(self):
    for status, hold in ((1, 0), (1, 1), (0, 1)):
      for acc_active in (True, False):
        self.setUp()
        self._speed(7)
        self._acc(acc_active)
        self._brake_status(status, hold)
        lib.set_controls_allowed(True)
        self._pressure(True)
        self.assertEqual(acc_active, bool(lib.get_controls_allowed()), f"{status=} {hold=} {acc_active=}")

  def test_brake_status_not_sticky_after_init(self):
    self._brake_status(5)
    self.setUp()
    lib.set_controls_allowed(True)
    self._speed(54)
    self._pressure(True)
    self.assertFalse(lib.get_controls_allowed())

  # ---- Resume ----

  def _resume_ready(self, kph=0, acc_active=True, hold=1, controls=True, gas=0):
    self._speed(kph)
    rx('PedalandBrake', 0, {'AccPedal': gas})
    self._acc(acc_active)
    self._brake_status(0, hold)
    lib.set_controls_allowed(controls)

  def _resume(self, **extra):
    return tx('CCButtons', 0, {'ACCResumeBtn': 1, **extra})

  def _stock_buttons(self, **over):
    rx('CCButtons', 0, {**STOCK_BTN, **over})

  def _stock_resume(self, **over):
    return tx('CCButtons', 0, {**STOCK_BTN, 'ACCResumeBtn': 1, **over})

  def test_resume_allowed_at_held_standstill(self):
    self._resume_ready()
    self.assertTrue(self._resume())

  def test_resume_blocked_without_each_condition(self):
    for name, kwargs in {"controls": dict(controls=False), "acc": dict(acc_active=False), "hold": dict(hold=0),
                         "moving": dict(kph=7), "gas": dict(gas=20)}.items():
      self.setUp()
      self._resume_ready(**kwargs)
      self.assertFalse(self._resume(), name)

  def test_resume_is_one_burst_per_hold(self):
    self._resume_ready()
    sent = 0
    while self._resume() and sent < 1000:
      sent += 1
    self.assertEqual(40, sent)
    self._brake_status(0, 1)
    self.assertFalse(self._resume())
    self._brake_status(0, 0)
    self.assertFalse(self._resume())
    self._resume_ready()
    self.assertTrue(self._resume())

  def test_resume_budget_rearms_when_the_car_moves(self):
    self._resume_ready()
    while self._resume():
      pass
    self._speed(7)
    self._speed(0)
    self.assertTrue(self._resume())

  def test_resume_only_the_resume_bit(self):
    self._resume_ready()
    for extra in ('ACCStopBtn', 'ACCOnOffBtn', 'ACCMinusBtn'):
      self.assertFalse(self._resume(**{extra: 1}), extra)
    self.assertTrue(self._resume())

  def test_cancel_still_sent_regardless_of_resume_rules(self):
    for kwargs in (dict(), dict(hold=0), dict(kph=36, hold=0), dict(acc_active=False)):
      self.setUp()
      self._resume_ready(**kwargs)
      self.assertTrue(tx('CCButtons', 0, {'ACCStopBtn': 1}), kwargs)

  def test_resume_with_the_stock_bytes(self):
    self._resume_ready()
    self.assertFalse(self._stock_resume())  # needs a stock frame first
    self._stock_buttons()
    self.assertTrue(self._stock_resume())
    self.assertTrue(self._stock_resume(byte0=0xF9))  # byte 0 toggles on its own
    for field, value in (("byte1", 0xB2), ("byte2", 0), ("byte3", 0x2F), ("byte4", 1), ("byte5", 1), ("byte6", 0x11)):
      self.assertFalse(self._stock_resume(**{field: value}), field)
    self._stock_buttons(byte2=0xD0)
    self.assertFalse(self._stock_resume())
    self.assertTrue(self._stock_resume(byte2=0xD0))

  def test_a_rejected_resume_does_not_use_up_the_budget(self):
    self._resume_ready()
    self._stock_buttons()
    for _ in range(100):
      self.assertFalse(self._stock_resume(byte1=0xB2))
    sent = 0
    while self._stock_resume() and sent < 1000:
      sent += 1
    self.assertEqual(40, sent)

  def test_stock_bytes_do_not_open_up_cancel_or_other_buttons(self):
    self._resume_ready()
    self._stock_buttons()
    for btn in ("ACCStopBtn", "ACCOnOffBtn", "ACCSetBtn"):
      self.assertFalse(tx('CCButtons', 0, {**STOCK_BTN, btn: 1}), btn)
    self.assertFalse(self._stock_resume(ACCStopBtn=1))
    self.assertFalse(tx('CCButtons', 0, STOCK_BTN))


class TestVolvoC1V40Aol(unittest.TestCase):
  """The optional AOL profile: the vehicle's states grant, the host's request only narrows."""

  def setUp(self):
    lib.set_alternative_experience(AOL)
    self.assertEqual(0, lib.set_safety_hooks(VOLVO, C1))
    lib.init_tests()
    self.time = 1000
    lib.set_timer(self.time)
    lib.set_aol_test_heartbeat(True)

  def tearDown(self):
    lib.set_alternative_experience(0)

  def _healthy(self, *, main=False, lka=False, gear=DRIVE, engaged=False):
    self.time += 20000
    lib.set_timer(self.time)
    rx('PSCM1', 0, {'SteeringAngleServo': 0})
    rx('FSM0', 2, {'ACCStatusOnOff': int(main), 'ACCStatusActive': int(engaged)})
    rx('PedalandBrake', 0, {})
    rx('VehicleSpeed1', 0, {'VehicleSpeed': 72})
    rx('BrakeMessages', 0, {})
    rx('CEM0', 0, {'LKA_State': int(lka)})
    rx('TCM0', 0, {'GearShifter': gear})

  def _permission(self, request=LATERAL | LONGITUDINAL, **state):
    self._healthy(**state)
    lib.aol_set_host_request(request)
    return lib.aol_get_permission_mask()

  def test_lateral_needs_the_lka_and_the_acc_main(self):
    self.assertEqual(0, self._permission(request=LATERAL))
    self.assertEqual(0, self._permission(request=LATERAL, main=True))
    self.assertEqual(0, self._permission(request=LATERAL, lka=True))
    self.assertEqual(LATERAL, self._permission(request=LATERAL, main=True, lka=True))

  def test_engaged_acc_steers_without_the_lka(self):
    self.assertEqual(LATERAL, self._permission(request=LATERAL, main=True, engaged=True))

  def test_never_longitudinal(self):
    self.assertEqual(LATERAL, self._permission(main=True, lka=True, engaged=True))
    self.assertEqual(0, self._permission(request=LONGITUDINAL, main=True, lka=True, engaged=True))

  def test_only_in_drive(self):
    self.assertEqual(0, self._permission(request=LATERAL, main=True, lka=True, gear=REVERSE))
    self.assertEqual(LATERAL, self._permission(request=LATERAL, main=True, lka=True))

  def test_stale_gear_withdraws(self):
    self.assertEqual(LATERAL, self._permission(request=LATERAL, main=True, lka=True))
    lib.set_timer(self.time + 300001)
    self.assertEqual(0, lib.aol_get_permission_mask())

  def test_steering_uses_the_lateral_axis(self):
    from opendbc.car.volvo import volvocan as codec
    self._healthy(main=True, lka=True)
    lib.set_angle_meas(0, 0)
    lib.aol_set_host_request(0)
    self.assertFalse(lib.safety_tx_hook(packet(codec.create_c1_steering_control(packer, 0, True))))
    self.assertEqual(LATERAL, self._permission(request=LATERAL, main=True, lka=True))
    lib.set_angle_meas(0, 0)
    self.assertTrue(lib.safety_tx_hook(packet(codec.create_c1_steering_control(packer, 0, True))))

  def test_without_the_aol_flag_there_is_no_profile(self):
    lib.set_alternative_experience(0)
    self.assertEqual(0, lib.set_safety_hooks(VOLVO, C1))
    lib.init_tests()
    self._healthy(main=True, lka=True)
    lib.aol_set_host_request(LATERAL)
    self.assertEqual(0, lib.aol_get_permission_mask())


if __name__ == "__main__":
  unittest.main()
