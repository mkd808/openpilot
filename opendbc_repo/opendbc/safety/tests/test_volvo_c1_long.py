"""V40 (C1) experimental OP longitudinal, drive and coast only (safety param flag 16): OP's 0x160/0x280 are the camera's
own recent frames with only the request or the mode and traction replaced, and the camera keeps the brakes."""
import unittest
from opendbc.can import CANPacker
from opendbc.safety.tests.libsafety import libsafety_py
from opendbc.safety.tests.test_volvo_c1 import packet

lib = libsafety_py.libsafety
packer = CANPacker('volvo_v40_2017_pt')
VOLVO = 36
C1 = 2
C1_LONG = 2 | 16
ACC_REQUEST, ACC_MODE = 0x160, 0x280

# stock camera frames, route 7c7d3d5e5ff31d51/0000000d
REQUEST = bytes.fromhex("b000dd54cfd001af")  # ACC active, request raw 47
REQUEST_MASKED = bytes.fromhex("20000814cf00012f")
REQUEST_BRAKE = bytes.fromhex("9000fd5433f001a5")  # hydraulic brake flags set
MODE_DRIVE = bytes.fromhex("40000000001dde62")
MODE_COAST = bytes.fromhex("400000000002dda3")
MODE_BRAKE = bytes.fromhex("400000000015dc00")


def with_request(dat: bytes, counts: int) -> bytes:
  return dat[:7] + bytes([(dat[7] & 0x80) | (counts + 64)])


def with_mode(dat: bytes, mode: int, traction: int) -> bytes:
  return dat[:6] + bytes([(dat[6] & 0xFC) | mode, traction & 0xFF])


class TestVolvoC1Long(unittest.TestCase):
  def setUp(self, param=C1_LONG):
    lib.set_alternative_experience(0)
    self.assertEqual(0, lib.set_safety_hooks(VOLVO, param))
    lib.init_tests()
    lib.set_timer(1_000_000)

  def _cam(self, addr, dat):
    lib.safety_rx_hook(packet((addr, dat, 2)))

  def _tx(self, addr, dat):
    return lib.safety_tx_hook(packet((addr, dat, 0)))

  def _engage(self):
    lib.set_controls_allowed(True)

  def test_param_admitted_and_stock_profile_unchanged(self):
    self.setUp(C1)
    self._engage()
    self._cam(ACC_REQUEST, REQUEST)
    self._cam(ACC_MODE, MODE_DRIVE)
    self.assertFalse(self._tx(ACC_REQUEST, REQUEST))
    self.assertFalse(self._tx(ACC_MODE, MODE_DRIVE))
    for addr in (ACC_REQUEST, ACC_MODE):
      self.assertEqual(0, lib.safety_fwd_hook(2, addr))

  def test_forwarding_blocked_only_while_op_may_send(self):
    for allowed in (False, True):
      self.setUp()
      lib.set_controls_allowed(allowed)
      for addr in (ACC_REQUEST, ACC_MODE):
        self.assertEqual(-1 if allowed else 0, lib.safety_fwd_hook(2, addr))
        self.assertEqual(2, lib.safety_fwd_hook(0, addr))
      self.assertEqual(0, lib.safety_fwd_hook(2, 0x30))
      self.assertEqual(-1, lib.safety_fwd_hook(2, 0xD0))  # steering stays OP's, as before

  def test_gas_hands_back_to_the_camera(self):
    self._engage()
    self._cam(ACC_REQUEST, REQUEST)
    lib.safety_rx_hook(packet(packer.make_can_msg('PedalandBrake', 0, {'AccPedal': 10})))
    self.assertTrue(lib.get_gas_pressed_prev())
    self.assertEqual(0, lib.safety_fwd_hook(2, ACC_REQUEST))
    self.assertFalse(self._tx(ACC_REQUEST, with_request(REQUEST, 0)))

  def test_no_tx_without_controls(self):
    self._cam(ACC_REQUEST, REQUEST)
    self._cam(ACC_MODE, MODE_DRIVE)
    self.assertFalse(self._tx(ACC_REQUEST, REQUEST))
    self.assertFalse(self._tx(ACC_MODE, MODE_DRIVE))

  def test_request_limits(self):
    self._engage()
    self._cam(ACC_REQUEST, REQUEST)
    for counts in range(-64, 64):
      with self.subTest(counts=counts):
        unchanged = counts == -17  # the camera's own frame (raw 47) may go back as it is
        self.assertEqual(-7 <= counts <= 25 or unchanged, bool(self._tx(ACC_REQUEST, with_request(REQUEST, counts))))

  def test_request_only_replaces_the_request_bits(self):
    self._engage()
    self._cam(ACC_REQUEST, REQUEST)
    self._cam(ACC_REQUEST, REQUEST_MASKED)
    self.assertTrue(self._tx(ACC_REQUEST, with_request(REQUEST_MASKED, 5)))
    for byte in range(8):
      for bit in range(8):
        if byte == 7 and bit < 7:
          continue
        dat = bytearray(with_request(REQUEST, 5))
        dat[byte] ^= 1 << bit
        with self.subTest(byte=byte, bit=bit):
          self.assertFalse(self._tx(ACC_REQUEST, bytes(dat)))

  def test_camera_keeps_the_brakes(self):
    self._engage()
    self._cam(ACC_REQUEST, REQUEST_BRAKE)
    self.assertTrue(self._tx(ACC_REQUEST, REQUEST_BRAKE))
    for counts in (-7, 0, 5):
      self.assertFalse(self._tx(ACC_REQUEST, with_request(REQUEST_BRAKE, counts)))
    self._cam(ACC_MODE, MODE_BRAKE)
    self.assertTrue(self._tx(ACC_MODE, MODE_BRAKE))
    self.assertFalse(self._tx(ACC_MODE, with_mode(MODE_BRAKE, 1, -20)))
    self.assertFalse(self._tx(ACC_MODE, with_mode(MODE_BRAKE, 2, 20)))

  def test_camera_keeps_the_brakes_with_either_flag(self):
    # the camera has the brakes when either hydraulic flag is set, not only when both are
    for byte in (2, 5):
      self.setUp()
      self._engage()
      cam = bytearray(REQUEST)
      cam[byte] |= 0x20
      self._cam(ACC_REQUEST, bytes(cam))
      self.assertTrue(self._tx(ACC_REQUEST, bytes(cam)))
      for counts in (-7, 0, 5):
        with self.subTest(byte=byte, counts=counts):
          self.assertFalse(self._tx(ACC_REQUEST, with_request(bytes(cam), counts)))

  def test_brake_flags_are_bit_5_only(self):
    # other bits in the two flag bytes aren't brake flags: OP may still replace the request
    self._engage()
    cam = bytearray(REQUEST)  # byte 2 bit 0 is already set
    cam[5] |= 0x01
    self._cam(ACC_REQUEST, bytes(cam))
    self.assertTrue(self._tx(ACC_REQUEST, with_request(bytes(cam), 5)))

  def test_mode_and_traction(self):
    self._engage()
    self._cam(ACC_MODE, MODE_DRIVE)
    for traction in range(256):
      with self.subTest(traction=traction):
        self.assertEqual(traction <= 186, bool(self._tx(ACC_MODE, with_mode(MODE_DRIVE, 2, traction))))
        self.assertEqual(traction >= 0x81, bool(self._tx(ACC_MODE, with_mode(MODE_DRIVE, 1, traction))))
        self.assertFalse(self._tx(ACC_MODE, with_mode(MODE_DRIVE, 0, traction)))  # OP never picks brake mode
        self.assertFalse(self._tx(ACC_MODE, with_mode(MODE_DRIVE, 3, traction)))
    self._cam(ACC_MODE, MODE_COAST)
    self.assertTrue(self._tx(ACC_MODE, with_mode(MODE_COAST, 2, 50)))
    for byte in range(7):
      for bit in range(8):
        if byte == 6 and bit < 2:
          continue
        dat = bytearray(with_mode(MODE_DRIVE, 2, 50))
        dat[byte] ^= 1 << bit
        with self.subTest(byte=byte, bit=bit):
          self.assertFalse(self._tx(ACC_MODE, bytes(dat)))

  def test_camera_frames_expire(self):
    self._engage()
    self._cam(ACC_REQUEST, REQUEST)
    self._cam(ACC_MODE, MODE_DRIVE)
    lib.set_timer(1_100_000)
    self.assertTrue(self._tx(ACC_REQUEST, with_request(REQUEST, 0)))
    self.assertTrue(self._tx(ACC_MODE, with_mode(MODE_DRIVE, 2, 0)))
    lib.set_timer(1_100_001)
    self.assertFalse(self._tx(ACC_REQUEST, with_request(REQUEST, 0)))
    self.assertFalse(self._tx(ACC_MODE, with_mode(MODE_DRIVE, 2, 0)))

  def test_no_camera_frame_no_tx(self):
    self._engage()
    self.assertFalse(self._tx(ACC_REQUEST, REQUEST))
    self.assertFalse(self._tx(ACC_MODE, MODE_DRIVE))

  def test_wrong_bus(self):
    self._engage()
    self._cam(ACC_REQUEST, REQUEST)
    self.assertFalse(lib.safety_tx_hook(packet((ACC_REQUEST, REQUEST, 2))))
    self.assertFalse(lib.safety_tx_hook(packet((ACC_REQUEST, REQUEST[:7], 0))))


C1_BRAKE_TEST = C1_LONG | 32


def with_flags(dat: bytes, a: int, b: int) -> bytes:
  out = bytearray(dat)
  out[2] = (out[2] & 0xDF) | (0x20 * a)
  out[5] = (out[5] & 0xDF) | (0x20 * b)
  return bytes(out)


class TestVolvoC1BrakeTest(TestVolvoC1Long):
  """Brake-test branch (flag 32 with 16): OP may also brake lightly the way the camera does, never while it brakes.
  Inherits the long tests that hold with the flag; the ones that change are overridden below."""
  def setUp(self, param=C1_BRAKE_TEST):
    super().setUp(param)

  def test_param_admitted_and_stock_profile_unchanged(self):
    self.assertEqual(0, lib.set_safety_hooks(VOLVO, C1_BRAKE_TEST))
    self.assertEqual(-1, lib.set_safety_hooks(VOLVO, C1 | 32))  # the brake test needs long

  def test_request_limits(self):
    self._engage()
    self._cam(ACC_REQUEST, REQUEST)
    for counts in range(-64, 64):
      with self.subTest(counts=counts):
        unchanged = counts == -17
        self.assertEqual(-16 <= counts <= 25 or unchanged, bool(self._tx(ACC_REQUEST, with_request(REQUEST, counts))))

  def test_request_only_replaces_the_request_bits(self):
    # as in step 1, except that the two flags may now change together
    self._engage()
    self._cam(ACC_REQUEST, REQUEST)
    for byte in range(8):
      for bit in range(8):
        if (byte == 7 and bit < 7) or (byte in (2, 5) and bit == 5):
          continue
        dat = bytearray(with_request(REQUEST, -5))
        dat[byte] ^= 1 << bit
        with self.subTest(byte=byte, bit=bit):
          self.assertFalse(self._tx(ACC_REQUEST, bytes(dat)))

  def test_op_brake_flags(self):
    self._engage()
    self._cam(ACC_REQUEST, REQUEST)
    for counts in range(-20, 10):
      for a, b in ((0, 0), (1, 1), (1, 0), (0, 1)):
        unchanged = counts == -17 and (a, b) == (0, 0)  # the camera's own frame (raw 47)
        expected = unchanged or (-16 <= counts <= 25 and ((a, b) == (0, 0) or ((a, b) == (1, 1) and counts <= 0)))
        with self.subTest(counts=counts, flags=(a, b)):
          self.assertEqual(expected, bool(self._tx(ACC_REQUEST, with_flags(with_request(REQUEST, counts), a, b))))

  def test_no_op_braking_while_the_camera_brakes(self):
    self._engage()
    self._cam(ACC_REQUEST, REQUEST_BRAKE)
    for counts in (-13, -7, 0):
      self.assertFalse(self._tx(ACC_REQUEST, with_request(REQUEST_BRAKE, counts)))
    self._cam(ACC_MODE, MODE_BRAKE)
    self.assertFalse(self._tx(ACC_MODE, with_mode(MODE_BRAKE, 1, -20)))

  def test_mode_and_traction(self):
    self._engage()
    self._cam(ACC_MODE, MODE_DRIVE)
    for traction in range(256):
      with self.subTest(traction=traction):
        self.assertEqual(traction <= 186, bool(self._tx(ACC_MODE, with_mode(MODE_DRIVE, 2, traction))))
        self.assertEqual(traction >= 0x81, bool(self._tx(ACC_MODE, with_mode(MODE_DRIVE, 1, traction))))
        self.assertEqual(traction == 0, bool(self._tx(ACC_MODE, with_mode(MODE_DRIVE, 0, traction))))  # OP brake mode
        self.assertFalse(self._tx(ACC_MODE, with_mode(MODE_DRIVE, 3, traction)))


# car 0x1B0 frames, route 00000019: ACC braking (status 5, pressure 51/36), driving, and a driver's braking (status 0, pressure)
BRAKE_ACC = bytes.fromhex("fc33a5020024c000")
BRAKE_NONE = bytes.fromhex("fc00a0000000c000")
BRAKE_DRIVER = bytes.fromhex("fc3ba003002ac000")
BRAKE_MSGS = 0x1B0


def masked(dat: bytes) -> bytes:
  out = bytearray(dat)
  out[0] &= 0xFC
  out[1] = 0
  out[2] &= 0xF8
  out[4] &= 0xFC
  out[5] = 0
  return bytes(out)


class TestVolvoC1BrakeRelay(unittest.TestCase):
  """Test 2: while OP brakes, OP may relay the car's 0x1B0 to the camera with the ACC braking status and pressures zeroed."""
  def setUp(self, param=C1_BRAKE_TEST):
    lib.set_alternative_experience(0)
    self.assertEqual(0, lib.set_safety_hooks(VOLVO, param))
    lib.init_tests()
    lib.set_timer(1_000_000)
    lib.set_controls_allowed(True)
    lib.safety_rx_hook(packet((ACC_REQUEST, REQUEST, 2)))

  def _car(self, dat):
    lib.safety_rx_hook(packet((BRAKE_MSGS, dat, 0)))

  def _relay(self, dat):
    return bool(lib.safety_tx_hook(packet((BRAKE_MSGS, dat, 2))))

  def _op_brakes(self):
    self.assertTrue(lib.safety_tx_hook(packet((ACC_REQUEST, with_flags(with_request(REQUEST, -12), 1, 1), 0))))

  def test_only_after_op_braked(self):
    self._car(BRAKE_ACC)
    self.assertFalse(self._relay(masked(BRAKE_ACC)))
    self._op_brakes()
    self.assertTrue(self._relay(masked(BRAKE_ACC)))
    self.assertTrue(self._relay(BRAKE_ACC))  # unchanged is fine too

  def test_window_after_op_braking(self):
    self._op_brakes()
    lib.set_timer(1_300_000)
    self._car(BRAKE_ACC)
    self.assertTrue(self._relay(masked(BRAKE_ACC)))
    lib.set_timer(1_300_001)
    self._car(BRAKE_ACC)
    self.assertFalse(self._relay(masked(BRAKE_ACC)))

  def test_never_hides_a_drivers_braking(self):
    self._op_brakes()
    self._car(BRAKE_DRIVER)
    self.assertTrue(self._relay(BRAKE_DRIVER))
    self.assertFalse(self._relay(masked(BRAKE_DRIVER)))

  def test_only_the_braking_fields_change(self):
    self._op_brakes()
    self._car(BRAKE_ACC)
    keep = (0xFC, 0x00, 0xF8, 0xFF, 0xFC, 0x00, 0xFF, 0xFF)
    for byte in range(8):
      for bit in range(8):
        if not keep[byte] >> bit & 1:
          continue
        dat = bytearray(masked(BRAKE_ACC))
        dat[byte] ^= 1 << bit
        with self.subTest(byte=byte, bit=bit):
          self.assertFalse(self._relay(bytes(dat)))
    partly = bytearray(masked(BRAKE_ACC))
    partly[1] = BRAKE_ACC[1]  # one pressure left in: all three fields zero together or none
    self.assertFalse(self._relay(bytes(partly)))

  def test_car_frames_expire(self):
    self._op_brakes()
    self._car(BRAKE_ACC)
    lib.set_timer(1_100_001)
    lib.safety_rx_hook(packet((ACC_REQUEST, REQUEST, 2)))  # a fresh camera frame, but the car's 0x1B0 is now stale
    self._op_brakes()
    self.assertFalse(self._relay(masked(BRAKE_ACC)))

  def test_not_without_controls_or_on_the_wrong_bus(self):
    self._op_brakes()
    self._car(BRAKE_ACC)
    self.assertFalse(bool(lib.safety_tx_hook(packet((BRAKE_MSGS, masked(BRAKE_ACC), 0)))))
    lib.set_controls_allowed(False)
    self.assertFalse(self._relay(masked(BRAKE_ACC)))

  def test_forwarding_pauses_while_op_relays(self):
    self.assertEqual(2, lib.safety_fwd_hook(0, BRAKE_MSGS))
    self._op_brakes()
    self._car(BRAKE_ACC)
    self.assertTrue(self._relay(masked(BRAKE_ACC)))
    self.assertEqual(-1, lib.safety_fwd_hook(0, BRAKE_MSGS))
    lib.set_timer(1_025_001)
    self.assertEqual(2, lib.safety_fwd_hook(0, BRAKE_MSGS))

  def test_not_without_the_brake_test_flag(self):
    self.setUp(C1_LONG)
    self._car(BRAKE_ACC)
    self.assertFalse(self._relay(masked(BRAKE_ACC)))
    self.assertFalse(self._relay(BRAKE_ACC))


# camera FSM0 (0x30) frames, routes 00000016/0000001b: ACC active, active + tracking, and ACC off (after the fault)
FSM0_ACTIVE = bytes.fromhex("b1906f000000638e")
FSM0_OFF = bytes.fromhex("b1906f102000638a")
FSM0 = 0x30


def tracking(dat: bytes) -> bytes:
  return dat[:7] + bytes([dat[7] | 0x01])


class TestVolvoC1Fsm0Relay(TestVolvoC1BrakeRelay):
  """Test 3: while OP brakes, OP may relay the camera's FSM0 to the car with only ACCStatusTracking set."""
  def _fsm0(self, dat):
    lib.safety_rx_hook(packet((FSM0, dat, 2)))

  def _relay0(self, dat):
    return bool(lib.safety_tx_hook(packet((FSM0, dat, 0))))

  def test_fsm0_only_after_op_braked(self):
    self._fsm0(FSM0_ACTIVE)
    self.assertFalse(self._relay0(tracking(FSM0_ACTIVE)))
    self._op_brakes()
    self._fsm0(FSM0_ACTIVE)
    self.assertTrue(self._relay0(tracking(FSM0_ACTIVE)))
    self.assertTrue(self._relay0(FSM0_ACTIVE))

  def test_fsm0_tracking_only_while_acc_active(self):
    self._op_brakes()
    self._fsm0(FSM0_OFF)  # ACC off also ends controls, so nothing goes out, least of all a set tracking bit
    self.assertFalse(self._relay0(tracking(FSM0_OFF)))

  def test_fsm0_only_the_tracking_bit(self):
    self._op_brakes()
    self._fsm0(FSM0_ACTIVE)
    for byte in range(8):
      for bit in range(8):
        if byte == 7 and bit == 0:
          continue
        dat = bytearray(tracking(FSM0_ACTIVE))
        dat[byte] ^= 1 << bit
        with self.subTest(byte=byte, bit=bit):
          self.assertFalse(self._relay0(bytes(dat)))

  def test_fsm0_forwarding_pauses_while_op_relays(self):
    self.assertEqual(0, lib.safety_fwd_hook(2, FSM0))
    self._op_brakes()
    self._fsm0(FSM0_ACTIVE)
    self.assertTrue(self._relay0(tracking(FSM0_ACTIVE)))
    self.assertEqual(-1, lib.safety_fwd_hook(2, FSM0))
    lib.set_timer(1_025_001)
    self.assertEqual(0, lib.safety_fwd_hook(2, FSM0))

  def test_fsm0_not_without_the_brake_test_flag(self):
    self.setUp(C1_LONG)
    self._fsm0(FSM0_ACTIVE)
    self.assertFalse(self._relay0(FSM0_ACTIVE))


class TestVolvoC1AccBrakeGrace(unittest.TestCase):
  """ACC's brake pressure bleeds off after the status clears: while ACC stays active it isn't the driver's for 1 s."""
  def setUp(self):
    self.assertEqual(0, lib.set_safety_hooks(VOLVO, C1_LONG))
    lib.init_tests()
    lib.set_timer(1_000_000)

  def _state(self, status, pressure, acc_active):
    lib.safety_rx_hook(packet((FSM0, FSM0_ACTIVE if acc_active else FSM0_OFF, 2)))
    lib.safety_rx_hook(packet(packer.make_can_msg('BrakeMessages', 0, {'BrakeStatus': status, 'BrakePress0': 60 if pressure else 0})))
    lib.safety_rx_hook(packet(packer.make_can_msg('PedalandBrake', 0, {'BrakePedalActive': int(pressure)})))
    return bool(lib.get_brake_pressed_prev())

  def test_pressure_after_acc_braking_is_not_the_driver(self):
    self.assertFalse(self._state(5, True, True))
    lib.set_timer(1_500_000)
    self.assertFalse(self._state(0, True, True))  # route 0000001b at 89.3 s: status 0, pressure still up
    lib.set_timer(2_400_000)
    self.assertFalse(self._state(0, True, True))  # 0.9 s after the last status 5 (seen at 1.5 s, before the status 0 frame)
    lib.set_timer(2_500_001)
    self.assertTrue(self._state(0, True, True))

  def test_acc_off_ends_the_grace(self):
    self.assertFalse(self._state(5, True, True))
    self.assertTrue(self._state(0, True, False))  # the driver's tap cancels ACC

  def test_no_grace_without_acc_braking_first(self):
    self.assertTrue(self._state(0, True, True))


if __name__ == "__main__":
  unittest.main()
