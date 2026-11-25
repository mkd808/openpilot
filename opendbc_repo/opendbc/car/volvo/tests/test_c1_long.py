import copy
from types import SimpleNamespace

import pytest

from opendbc.can import CANPacker, CANParser
from opendbc.car import structs
from opendbc.car.volvo.carcontroller import CarController
from opendbc.car.volvo.eucd_params import LongMode
from opendbc.car.volvo.interface import CarInterface
from opendbc.car.volvo.values import CAR, DBC as DBCS, CarControllerParams
from opendbc.car.volvo.volvocan import C1_FSM2_MASK, c1_next_long_mode, c1_traction, create_c1_acc_mode, create_c1_acc_request
from opendbc.safety.tests.libsafety import libsafety_py
from opendbc.safety.tests.test_volvo_c1 import packet

DBC = "volvo_v40_2017_pt"
FSM2, FSM4 = 0x160, 0x280

# stock camera frames, route 7c7d3d5e5ff31d51/0000000d (ACC stop and go); tools/volvo_c1/replay_encode.py checks every frame
FSM2_ACC_OFF = bytes.fromhex("9000df5408d001bf")
FSM2_ENGAGE = bytes.fromhex("b000dd54dbd001b0")
FSM2_BRAKE = bytes.fromhex("9000fd5433f001a5")
# (normal, the masked copy after it)
FSM2_MASKED_PAIRS = [
  (FSM2_ACC_OFF, bytes.fromhex("00000a140800013f")),
  (bytes.fromhex("b000dd54cfd001af"), bytes.fromhex("20000814cf00012f")),
  (bytes.fromhex("9000fd53f9f001a3"), bytes.fromhex("00002813f9200123")),
]
FSM2_FRAMES = [FSM2_ACC_OFF, FSM2_ENGAGE, FSM2_BRAKE] + [f for pair in FSM2_MASKED_PAIRS for f in pair]
# (0x280 frame, the 0x160 request raw before it)
FSM4_BRAKE = (bytes.fromhex("400000000015dc00"), 64)
FSM4_COAST = (bytes.fromhex("400000000002dda3"), 52)
FSM4_DRIVE = (bytes.fromhex("40000000001dde62"), 77)


def parse(name: str, addr: int, freq: int, dat: bytes) -> dict:
  cp = CANParser(DBC, [(name, freq)], 0)
  cp.update([(0, [(addr, dat, 0)])])
  return copy.copy(cp.vl[name])


def traction_value(b: int, mode: LongMode) -> int:
  return b - 256 if mode == LongMode.COAST and b > 127 else b


def request(stock: bytes, accel=None, brake=None) -> bytes:
  fsm2 = parse("FSM2", FSM2, 50, stock)
  return create_c1_acc_request(CANPacker(DBC), fsm2, fsm2["ACC_AccelRequest"] if accel is None else accel, brake)[1]


def mode_msg(stock: bytes, accel=None, mode=None) -> bytes:
  return create_c1_acc_mode(CANPacker(DBC), parse("FSM4", FSM4, 40, stock), accel, mode)[1]


@pytest.mark.parametrize("stock", FSM2_FRAMES)
def test_request_copy_is_byte_identical(stock):
  assert request(stock) == stock


@pytest.mark.parametrize("stock", [FSM4_BRAKE[0], FSM4_COAST[0], FSM4_DRIVE[0]])
def test_mode_copy_is_byte_identical(stock):
  assert mode_msg(stock) == stock


@pytest.mark.parametrize("normal,masked", FSM2_MASKED_PAIRS)
def test_masked_copy_is_the_frame_and_mask(normal, masked):
  assert bytes(a & b for a, b in zip(normal, C1_FSM2_MASK, strict=True)) == masked


@pytest.mark.parametrize("raw", range(128))
def test_request_writes_only_the_request_bits(raw):
  for stock in (FSM2_ENGAGE, FSM2_MASKED_PAIRS[1][1]):
    out = request(stock, (raw - 64) * 0.08)
    assert out[7] & 0x7F == raw
    assert out[7] & 0x80 == stock[7] & 0x80  # masked copies stay masked
    assert out[:7] == stock[:7]


def test_request_clamps_to_the_field():
  assert request(FSM2_ENGAGE, -10.)[7] & 0x7F == 0
  assert request(FSM2_ENGAGE, 10.)[7] & 0x7F == 127


def test_brake_sets_both_hydraulic_flags():
  out = request(FSM2_ENGAGE, -1.0, brake=True)
  assert (out[2] ^ FSM2_ENGAGE[2], out[5] ^ FSM2_ENGAGE[5]) == (0x20, 0x20)
  out = request(FSM2_BRAKE, -1.0, brake=False)
  assert (out[2] ^ FSM2_BRAKE[2], out[5] ^ FSM2_BRAKE[5]) == (0x20, 0x20)
  assert request(FSM2_BRAKE, brake=None) == FSM2_BRAKE


@pytest.mark.parametrize("stock,raw,mode", [(*FSM4_BRAKE, LongMode.BRAKE), (*FSM4_COAST, LongMode.COAST), (*FSM4_DRIVE, LongMode.DRIVE)])
def test_mode_and_traction_near_stock(stock, raw, mode):
  out = mode_msg(stock, (raw - 64) * 0.08, mode)
  assert out[:6] == stock[:6] and out[6] == stock[6]
  assert abs(traction_value(out[7], mode) - traction_value(stock[7], mode)) <= 5


def test_traction_range():
  assert c1_traction(-2.0, LongMode.BRAKE) == 0
  assert c1_traction(1.0, LongMode.DRIVE) == 93
  assert c1_traction(-1.0, LongMode.DRIVE) == 0
  assert c1_traction(5.0, LongMode.DRIVE) == 255
  assert c1_traction(-0.5, LongMode.COAST) == (-46 & 0xFF)
  assert c1_traction(0.5, LongMode.COAST) == 0xFF  # never positive in coast
  assert c1_traction(-5.0, LongMode.COAST) == (-127 & 0xFF)


def test_mode_hysteresis():
  assert c1_next_long_mode(LongMode.COAST, -0.56) == LongMode.COAST
  assert c1_next_long_mode(LongMode.COAST, -0.64) == LongMode.BRAKE
  assert c1_next_long_mode(LongMode.BRAKE, -0.08) == LongMode.BRAKE  # brake holds until the request is positive
  assert c1_next_long_mode(LongMode.BRAKE, 0.08) == LongMode.DRIVE
  assert c1_next_long_mode(LongMode.COAST, 0.0) == LongMode.COAST
  assert c1_next_long_mode(LongMode.COAST, 0.08) == LongMode.DRIVE
  assert c1_next_long_mode(LongMode.DRIVE, 0.0) == LongMode.DRIVE
  assert c1_next_long_mode(LongMode.DRIVE, -0.08) == LongMode.COAST


# Step 2b controller against the panda: everything it sends must pass volvo_c1.h
lib = libsafety_py.libsafety
FSM2_DRIVE = bytes.fromhex("b000dd54cfd001c5")  # FSM2_ENGAGE's bytes 0-6, request raw 69 (+0.4)
FSM4_DRIVE_FRAME = FSM4_DRIVE[0]


def long_params():
  cp = CarInterface.get_non_essential_params(CAR.VOLVO_V40)
  return CarInterface._get_params(cp, CAR.VOLVO_V40, {0: {}, 1: {}, 2: {}}, [], True, False, False)


class Drive:
  def __init__(self):
    self.cc = CarController(DBCS[CAR.VOLVO_V40], long_params())
    assert lib.set_safety_hooks(36, 18) == 0
    lib.init_tests()
    lib.set_timer(1_000_000)
    lib.set_controls_allowed(True)

  def step(self, accel, request=FSM2_DRIVE, mode=FSM4_DRIVE_FRAME, long_active=True, enabled=True, standstill=False):
    for addr, dat in ((FSM2, request), (FSM4, mode)):
      lib.safety_rx_hook(packet((addr, dat, 2)))
    cs = SimpleNamespace(out=structs.CarState.new_message(standstill=standstill).as_reader(), c1_lka_torque=0, c1_brake_hold=False, c1_buttons_stock=None,
                         c1_msg_pscm={'SteeringAngleServo': 0, 'byte0': 0, 'byte3': 0, 'byte4': 0, 'byte7': 0, 'LKAActive': 0},
                         c1_stock_acc_request_frames=[parse("FSM2", FSM2, 50, request)],
                         c1_stock_acc_mode_frames=[parse("FSM4", FSM4, 40, mode)])
    cc = structs.CarControl.new_message(longActive=long_active, enabled=enabled)
    cc.actuators.accel = accel
    _, sends = self.cc.update(cc.as_reader(), cs, 0)
    long_sends = [(a, d, b) for a, d, b in sends if a in (FSM2, FSM4)]
    for msg in long_sends:
      assert lib.safety_tx_hook(packet(msg)), msg
    return {a: d for a, d, _ in long_sends}


def test_controller_capped_to_camera_and_accepted():
  drive = Drive()
  for _ in range(200):
    out = drive.step(1.0)
  assert CarControllerParams.C1_LONG_CAMERA_CAP
  assert out[FSM2][7] & 0x7F == 69  # held to the camera's +0.4
  assert out[FSM4][6] & 0x3 == LongMode.DRIVE


def test_controller_brakes_to_its_floor():
  drive = Drive()
  for _ in range(300):
    out = drive.step(-3.0)
  assert (out[FSM2][7] & 0x7F) - 64 == -25  # held at -2.0
  assert out[FSM2][2] & 0x20 and out[FSM2][5] & 0x20  # both hydraulic flags
  assert out[FSM4][6] & 0x3 == LongMode.BRAKE and out[FSM4][7] == 0


def test_controller_brake_flags_follow_brake_mode():
  # coast at -0.48: no flags; past brake entry: flags; back above zero: flags off, drive
  drive = Drive()
  for _ in range(300):
    out = drive.step(-0.48)
  assert out[FSM2][2] & 0x20 == 0 and out[FSM4][6] & 0x3 == LongMode.COAST
  for _ in range(100):
    out = drive.step(-0.8)
  assert out[FSM2][2] & 0x20 and out[FSM4][6] & 0x3 == LongMode.BRAKE
  for _ in range(100):
    out = drive.step(0.3)
  assert out[FSM2][2] & 0x20 == 0 and out[FSM2][5] & 0x20 == 0 and out[FSM4][6] & 0x3 == LongMode.DRIVE


def test_controller_hands_standstill_to_camera():
  drive = Drive()
  for _ in range(300):
    drive.step(-3.0)
  assert drive.step(-3.0, standstill=True) == {FSM2: FSM2_DRIVE, FSM4: FSM4_DRIVE_FRAME}


def test_controller_jerk_limited():
  # starts from the camera's +0.4, then down at 4 m/s^3 (0.04 per 10 ms step)
  drive = Drive()
  for _ in range(10):
    out = drive.step(-3.0)
  assert out[FSM2][7] & 0x7F == 64


def test_controller_passes_camera_braking_unchanged():
  drive = Drive()
  drive.step(0.5)
  out = drive.step(0.5, request=FSM2_BRAKE, mode=FSM4_BRAKE[0])
  assert out == {FSM2: FSM2_BRAKE, FSM4: FSM4_BRAKE[0]}


def test_controller_silent_when_disengaged():
  drive = Drive()
  assert drive.step(0.5, long_active=False, enabled=False) == {}


def test_controller_relays_camera_in_an_override():
  drive = Drive()
  drive.step(1.0)
  assert drive.step(1.0, long_active=False) == {FSM2: FSM2_DRIVE, FSM4: FSM4_DRIVE_FRAME}


def test_controller_takes_over_after_camera_braking_within_limits():
  drive = Drive()
  drive.step(0.5)
  drive.step(0.5, request=FSM2_BRAKE, mode=FSM4_BRAKE[0])  # camera at -2.4 with the brake flags
  out = drive.step(0.5)  # the camera drives again: OP ramps from its own floor, not the camera's -2.4
  assert (out[FSM2][7] & 0x7F) - 64 in (-24, -25)
