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


if __name__ == "__main__":
  unittest.main()
