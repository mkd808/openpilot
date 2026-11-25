"""V40 (C1) steering command limits in volvo_c1_angle_cmd_checks: the angle rate limits by speed, the relaxed rate back
towards the measured angle when the command is outside the measured-angle envelope, and the max angle. Mutation testing
left these unchecked: the existing tests only hit the envelope, which hides most rate-limit mistakes."""
import unittest
import numpy as np
from opendbc.can import CANPacker
from opendbc.safety.tests.libsafety import libsafety_py
from opendbc.safety.tests.test_volvo_c1 import packet
from opendbc.car.volvo import volvocan as codec

lib = libsafety_py.libsafety
packer = CANPacker('volvo_v40_2017_pt')
VOLVO, C1 = 36, 2
FSM1 = 0xD0
MAX_ANGLE = 8189
DEG_TO_CAN = np.float32(22.753128)
ENVELOPE = 455
RATE_BP = [np.float32(x) for x in (7.0, 17.0, 36.0)]  # m/s, same lookup up and down
RATE_V = [np.float32(y) for y in (2.0, 0.25, 0.1)]  # deg per frame


def steer(raw: int, active: bool = True) -> tuple:
  # FSM1 with the angle written straight in CAN counts, so tests hit exact limits
  _, base, bus = codec.create_c1_steering_control(packer, 0, active)
  data = bytearray(base)
  encoded = raw + 8192
  data[4] = (data[4] & 0xC0) | ((encoded >> 8) & 0x3F)
  data[5] = encoded & 0xFF
  data[6] = codec.create_c1_checksum(data)
  return FSM1, bytes(data), bus


def interpolate(x: np.float32) -> np.float32:
  # helpers.h safety_interpolate, in float32
  if x <= RATE_BP[0]:
    return RATE_V[0]
  for i in range(len(RATE_BP) - 1):
    if x < RATE_BP[i + 1]:
      dx = max(RATE_BP[i + 1] - RATE_BP[i], np.float32(0.0001))
      return np.float32((RATE_V[i + 1] - RATE_V[i]) * (x - RATE_BP[i]) / dx + RATE_V[i])
  return RATE_V[-1]


def limits(last: int, meas: int, v_min: int, v_max: int) -> tuple[int, int]:
  # volvo_c1_angle_cmd_checks with steering active and the car moving; speeds in the sample's 1/1000 m/s
  delta = int(float(interpolate(np.float32(v_min / 1000.0 - 1.)) * DEG_TO_CAN) + 1.)
  relaxed = int(float(interpolate(np.float32(v_max / 1000.0 + 1.)) * DEG_TO_CAN) - 1.)
  highest, lowest = last + delta, last - delta
  low_error, high_error = meas - ENVELOPE - 1, meas + ENVELOPE + 1
  if last > high_error:
    highest = max(last - relaxed, high_error)
  elif last < low_error:
    lowest = min(last + relaxed, low_error)
  else:
    highest, lowest = min(highest, high_error), max(lowest, low_error)
  return max(lowest, -MAX_ANGLE), min(highest, MAX_ANGLE)


class TestVolvoC1Steering(unittest.TestCase):
  def setUp(self):
    self.assertEqual(0, lib.set_safety_hooks(VOLVO, C1))
    lib.init_tests()
    lib.set_timer(0)

  def _speed(self, kph: float) -> tuple[int, int]:
    for _ in range(6):
      lib.safety_rx_hook(packet(packer.make_can_msg('VehicleSpeed1', 0, {'VehicleSpeed': kph})))
    return round(lib.get_vehicle_speed_min() * 1000), round(lib.get_vehicle_speed_max() * 1000)

  def _tx(self, raw: int, last: int, meas: int, active: bool = True) -> bool:
    lib.set_controls_allowed(True)
    lib.set_angle_meas(meas, meas)
    lib.set_desired_angle_last(last)
    return bool(lib.safety_tx_hook(packet(steer(raw, active))))

  def test_rate_and_envelope_limits(self):
    # (last command, measured angle): inside the envelope at and away from zero, both signs; right at the envelope's edge,
    # where it binds before the rate limit; outside the envelope on either side, where only the relaxed rate back towards
    # the measured angle is allowed, including across zero (the relaxed "up" rate). 2 km/h checks the envelope still
    # applies below 1 m/s.
    cases = [(0, 0), (200, 200), (-200, -200), (100, -300), (450, 0), (-450, 0), (683, 0), (-683, 0),
             (2000, 1000), (-2000, -1000), (-200, -1000), (200, 1000)]
    for kph in (2., 10., 25., 36., 72., 100., 130.):
      v_min, v_max = self._speed(kph)
      for last, meas in cases:
        lowest, highest = limits(last, meas, v_min, v_max)
        for raw in [*range(lowest - 3, lowest + 4), *range(highest - 3, highest + 4)]:
          with self.subTest(kph=kph, last=last, meas=meas, raw=raw):
            self.assertEqual(lowest <= raw <= highest, self._tx(raw, last, meas))

  def test_accepted_command_becomes_the_last_command(self):
    # the rate limit runs from the last accepted command; a rejected one resets it to the newest measured angle sample
    # (0 here: set_angle_meas only sets the sample's min and max)
    self._speed(72.)
    self.assertTrue(self._tx(203, 200, 200))
    self.assertEqual(203, lib.get_desired_angle_last())
    self.assertFalse(self._tx(300, 200, 200))
    self.assertEqual(0, lib.get_desired_angle_last())

  def test_rate_limit_shrinks_with_speed(self):
    deltas = []
    for kph in (10., 36., 72., 130.):
      self._speed(kph)
      deltas.append(max(d for d in range(1, 120) if self._tx(200 + d, 200, 200)))
    self.assertEqual(deltas, sorted(deltas, reverse=True))
    self.assertGreater(deltas[0], deltas[-1])

  def test_max_angle(self):
    # with steering off a command may only sit on the measured angle; past the max angle it's blocked even there
    for raw, allowed in ((MAX_ANGLE, True), (MAX_ANGLE + 1, False), (-MAX_ANGLE, True), (-MAX_ANGLE - 1, False)):
      with self.subTest(raw=raw):
        self.assertEqual(allowed, self._tx(raw, raw, raw, active=False))


if __name__ == "__main__":
  unittest.main()
