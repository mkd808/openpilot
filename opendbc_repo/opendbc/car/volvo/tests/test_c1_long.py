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
BUTTONS = 0x10
BRAKE_MSGS = 0x1B0
FSM0_ID = 0x30

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


# Drive-and-coast controller against the panda: everything it sends must pass volvo_c1.h
lib = libsafety_py.libsafety
FSM2_DRIVE = bytes.fromhex("b000dd54cfd001c5")  # FSM2_ENGAGE's bytes 0-6, request raw 69 (+0.4)
FSM4_DRIVE_FRAME = FSM4_DRIVE[0]


def long_params():
  cp = CarInterface.get_non_essential_params(CAR.VOLVO_V40)
  return CarInterface._get_params(cp, CAR.VOLVO_V40, {0: {}, 1: {}, 2: {}}, [], True, False, False)


class Drive:
  def __init__(self, param=18):
    self.cc = CarController(DBCS[CAR.VOLVO_V40], long_params())
    assert lib.set_safety_hooks(36, param) == 0
    lib.init_tests()
    lib.set_timer(1_000_000)
    lib.set_controls_allowed(True)

  def step(self, accel, request=FSM2_DRIVE, mode=FSM4_DRIVE_FRAME, long_active=True, enabled=True, standstill=False, v_ego=20.,
           lka=False, lead=False, gas=False, brake_msg=None, fsm0=None, a_ego=0.):
    for addr, dat in ((FSM2, request), (FSM4, mode)):
      lib.safety_rx_hook(packet((addr, dat, 2)))
    if brake_msg is not None:
      lib.safety_rx_hook(packet((BRAKE_MSGS, brake_msg, 0)))
    if fsm0 is not None:
      lib.safety_rx_hook(packet((FSM0_ID, fsm0, 2)))
    out = structs.CarState.new_message(standstill=standstill, vEgo=0. if standstill else v_ego, gasPressed=gas, aEgo=a_ego).as_reader()
    cs = SimpleNamespace(out=out, c1_lka_torque=0, c1_brake_hold=False, c1_buttons_stock=None, c1_lka_on=lka,
                         c1_msg_pscm={'SteeringAngleServo': 0, 'byte0': 0, 'byte3': 0, 'byte4': 0, 'byte7': 0, 'LKAActive': 0},
                         c1_stock_acc_request_frames=[parse("FSM2", FSM2, 50, request)],
                         c1_stock_acc_mode_frames=[parse("FSM4", FSM4, 40, mode)],
                         c1_brake_msg_frames=[] if brake_msg is None else [parse("BrakeMessages", BRAKE_MSGS, 50, brake_msg)],
                         c1_fsm0_frames=[] if fsm0 is None else [parse("FSM0", FSM0_ID, 100, fsm0)])
    cc = structs.CarControl.new_message(longActive=long_active, enabled=enabled)
    cc.actuators.accel = accel
    cc.hudControl.leadVisible = lead
    _, sends = self.cc.update(cc.as_reader(), cs, 0)
    long_sends = [(a, d, b) for a, d, b in sends if a in (FSM2, FSM4)]
    for msg in long_sends:
      assert lib.safety_tx_hook(packet(msg)), msg
    self.cancels = [(a, d, b) for a, d, b in sends if a == BUTTONS]
    for msg in self.cancels:
      assert lib.safety_tx_hook(packet(msg)), msg
    self.relays = [(a, d, b) for a, d, b in sends if a in (BRAKE_MSGS, FSM0_ID)]
    for msg in self.relays:
      assert lib.safety_tx_hook(packet(msg)), msg
    return {a: d for a, d, _ in long_sends}


def test_controller_capped_to_camera_and_accepted():
  drive = Drive()
  for _ in range(200):
    out = drive.step(1.0)
  assert CarControllerParams.C1_LONG_CAMERA_CAP
  assert out[FSM2][7] & 0x7F == 69  # held to the camera's +0.4
  assert out[FSM4][6] & 0x3 == LongMode.DRIVE


def test_controller_never_brakes():
  # step 2a's own braking faulted stock ACC (route 00000013, 207 s): OP holds at its coast floor, no flags, no brake mode
  drive = Drive()
  for _ in range(300):
    out = drive.step(-3.0)
  assert (out[FSM2][7] & 0x7F) - 64 == -6  # -0.48
  assert out[FSM2][2] & 0x20 == 0 and out[FSM2][5] & 0x20 == 0
  assert out[FSM4][6] & 0x3 == LongMode.COAST


def test_controller_coasts_through_camera_engine_braking():
  # route 00000012 at 25 s: the camera flips 0x280 between coast and brake mode without the hydraulic flags (engine
  # braking). OP's request ramps down once without restarts, never with the flags; the camera's brake mode goes out as is.
  fsm2_coast = request(FSM2_ENGAGE, -0.16)
  drive = Drive()
  requests = []
  for i in range(150):
    mode = FSM4_BRAKE[0] if (i // 5) % 2 else FSM4_COAST[0]
    out = drive.step(-0.8, request=fsm2_coast, mode=mode)
    requests.append((out[FSM2][7] & 0x7F) - 64)
    assert out[FSM2][2] & 0x20 == 0 and out[FSM2][5] & 0x20 == 0
    if mode == FSM4_BRAKE[0]:
      assert out[FSM4] == FSM4_BRAKE[0]
    else:
      assert out[FSM4][6] & 0x3 == LongMode.COAST
  assert all(b <= a for a, b in zip(requests, requests[1:], strict=False))
  assert requests[-1] == -6


def test_controller_hands_standstill_to_camera():
  # the camera made this stop (it no longer asks to drive): at the standstill its frames go out, no cancel
  fsm2_zero = request(FSM2_ENGAGE, 0.)
  drive = Drive()
  for _ in range(300):
    drive.step(-3.0, request=fsm2_zero, v_ego=2.)
  assert drive.step(-3.0, request=fsm2_zero, standstill=True) == {FSM2: fsm2_zero, FSM4: FSM4_DRIVE_FRAME}
  assert not drive.cancels


def test_handoff_cancels_acc_when_op_would_stop_against_the_camera():
  # slow, OP braking, the camera asking to accelerate (+0.4): OP cancels stock ACC and keeps its own frames
  drive = Drive()
  drive.step(-1.0, v_ego=3.)
  assert drive.cancels and drive.cancels[0][1][7] == 0x10 and drive.cancels[0][2] == 0
  sent = 0
  for _ in range(30):
    out = drive.step(-1.0, v_ego=2.)
    sent += len(drive.cancels)
  assert sent == 3  # every 10 frames while latched
  assert (out[FSM2][7] & 0x7F) - 64 < 0  # OP's braking, not the camera's +0.4


def test_handoff_keeps_op_frames_at_standstill():
  drive = Drive()
  for _ in range(100):
    drive.step(-1.0, v_ego=3.)
  out = drive.step(-1.0, standstill=True)
  assert (out[FSM2][7] & 0x7F) - 64 == -6 and out[FSM2][2] & 0x20 == 0  # still OP's coast, never the camera's +0.4
  assert out[FSM4][6] & 0x3 == LongMode.COAST


def test_handoff_ends_when_acc_is_off():
  drive = Drive()
  drive.step(-1.0, v_ego=3.)
  assert drive.step(-1.0, long_active=False, enabled=False) == {}
  drive.step(0.3, v_ego=20.)
  assert not drive.cancels


@pytest.mark.parametrize("kwargs", [
  dict(accel=-1.0, v_ego=5.),  # above 14 km/h
  dict(accel=-0.2, v_ego=3.),  # only gentle coasting
  dict(accel=-0.46, v_ego=1.7),  # route 00000020 at 109.6 s: the lead pulls away, OP's planner lags behind
  dict(accel=-1.0, v_ego=3., request=request(FSM2_ENGAGE, 0.)),  # the camera isn't asking to accelerate
  dict(accel=-1.0, v_ego=3., request=FSM2_BRAKE, mode=FSM4_BRAKE[0]),  # the camera is braking
  dict(accel=-1.0, v_ego=3., long_active=False),  # override
])
def test_no_handoff(kwargs):
  drive = Drive()
  for _ in range(20):
    drive.step(**kwargs)
    assert not drive.cancels


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
  assert (out[FSM2][7] & 0x7F) - 64 == -6
  assert out[FSM2][2] & 0x20 == 0 and out[FSM4][6] & 0x3 == LongMode.COAST


# Brake test (brake-test branch): flag file + LKA switch flip -> fixed -1.0 brake request with the flags, panda param 50
@pytest.fixture
def flag_file(monkeypatch):
  import opendbc.car.volvo.carcontroller as carcontroller
  monkeypatch.setattr(carcontroller.os.path, "exists", lambda path: path == CarControllerParams.C1_BRAKE_TEST_FLAG)


def brake_test(drive, frames, **kwargs):
  for _ in range(5):
    drive.step(0.3, v_ego=15.)
  out = None
  for _ in range(frames):
    out = drive.step(0.3, v_ego=15., lka=True, **kwargs)
  return out


def braking(out) -> bool:
  return bool(out[FSM2][2] & 0x20 and out[FSM2][5] & 0x20 and out[FSM4][6] & 0x3 == LongMode.BRAKE)


def test_brake_test_param_on_this_branch():
  assert long_params().safetyConfigs[0].safetyParam == 50


def test_brake_test_runs_on_lka_flip(flag_file):
  drive = Drive(param=50)
  out = brake_test(drive, 60)
  assert braking(out) and out[FSM4][7] == 0
  counts = (out[FSM2][7] & 0x7F) - 64
  assert -12 < counts < 0  # still ramping in at 1 m/s^3 from the camera's +0.4
  out = brake_test(drive, 250)
  assert (out[FSM2][7] & 0x7F) - 64 == -16  # -1.25 from the brake table for -1.0, OP's planner wanting +0.3 is ignored


def test_brake_test_quick_onset(flag_file):
  # fast through the engine-braking zone, then 1 m/s^3: from the camera's +0.4 the request is at -0.5 within ~0.26 s
  drive = Drive(param=50)
  for _ in range(5):
    drive.step(0.3, v_ego=15.)
  for _ in range(30):
    out = drive.step(0.3, v_ego=15., lka=True)
  assert (out[FSM2][7] & 0x7F) - 64 <= -6  # past -0.48 after 0.3 s (at 1 m/s^3 it would still be near +0.1)
  for _ in range(45):
    out = drive.step(0.3, v_ego=15., lka=True)
  assert -16 < (out[FSM2][7] & 0x7F) - 64 <= -10  # then on the slow ramp


def test_brake_test_eases_off_when_braking_too_hard(flag_file):
  drive = Drive(param=50)
  brake_test(drive, 250)
  out = None
  for _ in range(100):
    out = drive.step(0.3, v_ego=15., lka=True, a_ego=-1.25)  # 0.25 harder than the target, under the limiter
  counts = (out[FSM2][7] & 0x7F) - 64
  assert braking(out) and -16 < counts <= -12  # eased by 1.0 * 0.25
  assert drive.cc.c1_bt_active


def test_brake_test_ends_after_max_time(flag_file):
  drive = Drive(param=50)
  out = brake_test(drive, round(CarControllerParams.C1_BRAKE_TEST_MAX_S / 0.01) + 20)
  assert not drive.cc.c1_bt_active and not braking(out)


@pytest.mark.parametrize("kwargs", [{"lead": True}, {"gas": True}])
def test_brake_test_needs_a_clear_road(flag_file, kwargs):
  drive = Drive(param=50)
  out = brake_test(drive, 60, **kwargs) if "gas" not in kwargs else None
  if "gas" in kwargs:
    brake_test(drive, 20)
    assert drive.cc.c1_bt_active
    drive.step(0.3, v_ego=15., lka=True, gas=True, long_active=False)
    assert not drive.cc.c1_bt_active
  else:
    assert not drive.cc.c1_bt_active and not braking(out)


def test_brake_test_sends_its_end_speed_as_the_target(flag_file):
  # 0x280 byte 5 (SpeedTarget): the camera drives with target 60, OP brakes from 54 km/h towards 54 - 15 = 39
  fsm4_60 = FSM4_DRIVE_FRAME[:5] + bytes([60]) + FSM4_DRIVE_FRAME[6:]
  drive = Drive(param=50)
  for _ in range(5):
    drive.step(0.3, v_ego=15., mode=fsm4_60)
  out = None
  for _ in range(60):
    out = drive.step(0.3, v_ego=15., lka=True, mode=fsm4_60)
  assert braking(out) and out[FSM4][5] == 39


def test_brake_test_target_over_camera_engine_braking(flag_file):
  # the camera's brake mode without flags (engine braking) mid-test: OP's brake frame with its target still goes out
  engine_brake_60 = FSM4_BRAKE[0][:5] + bytes([60]) + FSM4_BRAKE[0][6:]  # brake mode, no flags, cruising target
  drive = Drive(param=50)
  brake_test(drive, 60)
  out = drive.step(0.3, v_ego=15., lka=True, mode=engine_brake_60)
  assert braking(out) and out[FSM4][5] == 39  # OP's end speed, not the camera's 60
  out = drive.step(0.3, v_ego=15., lka=True, mode=FSM4_BRAKE[0])
  assert out[FSM4][5] == FSM4_BRAKE[0][5]  # the camera's 21 is already lower: kept


def test_brake_test_needs_the_flag_file():
  drive = Drive(param=50)
  out = brake_test(drive, 60)
  assert not drive.cc.c1_bt_active and not braking(out)


def test_brake_test_needs_speed(flag_file):
  drive = Drive(param=50)
  for _ in range(5):
    drive.step(0.3, v_ego=7.)
  out = None
  for _ in range(60):
    out = drive.step(0.3, v_ego=7., lka=True)
  assert not drive.cc.c1_bt_active and not braking(out)


# Test 2: the car's 0x1B0 relayed to the camera with the ACC braking status and pressures zeroed (route 00000019 frames)
BRAKE_ACC = bytes.fromhex("fc33a5020024c000")
BRAKE_NONE = bytes.fromhex("fc00a0000000c000")


def test_brake_test_relays_0x1b0_masked_to_the_camera(flag_file):
  drive = Drive(param=50)
  for _ in range(5):
    drive.step(0.3, v_ego=15., brake_msg=BRAKE_NONE)
  for _ in range(60):
    drive.step(0.3, v_ego=15., lka=True, brake_msg=BRAKE_ACC)
  relays = [r for r in drive.relays if r[0] == BRAKE_MSGS]
  assert drive.cc.c1_bt_active and len(relays) == 1
  _, dat, bus = relays[0]
  assert bus == 2 and dat[2] & 0x7 == 0 and dat[1] == 0 and dat[5] == 0 and dat[0] & 0x3 == 0
  assert dat[0] & 0xFC == BRAKE_ACC[0] & 0xFC and dat[3] == BRAKE_ACC[3] and dat[6:] == BRAKE_ACC[6:]


def test_relay_copies_frames_unchanged_when_not_acc_braking(flag_file):
  drive = Drive(param=50)
  brake_test(drive, 30)
  drive.step(0.3, v_ego=15., lka=True, brake_msg=BRAKE_NONE)
  assert [r[1] for r in drive.relays if r[0] == BRAKE_MSGS] == [BRAKE_NONE]


def test_relay_stops_after_the_test(flag_file):
  drive = Drive(param=50)
  for _ in range(5):
    drive.step(0.3, v_ego=15., brake_msg=BRAKE_NONE)
  for _ in range(round(CarControllerParams.C1_BRAKE_TEST_MAX_S / 0.01) + 5):
    drive.step(0.3, v_ego=15., lka=True, brake_msg=BRAKE_ACC)
  assert not drive.cc.c1_bt_active
  for _ in range(CarControllerParams.C1_BRAKE_TEST_RELAY_TAIL_FRAMES + 2):
    drive.step(0.3, v_ego=15., lka=True, brake_msg=BRAKE_NONE)
  assert not drive.relays


def test_no_relay_without_the_test():
  drive = Drive(param=50)
  for _ in range(50):
    drive.step(0.3, v_ego=15., brake_msg=BRAKE_ACC)
    assert not drive.relays


# Test 3: overshoot limiter and FSM0 ACCStatusTracking while OP brakes
FSM0_ACTIVE = bytes.fromhex("b1906f000000638e")


def test_overshoot_drops_and_reapplies_the_flags(flag_file):
  drive = Drive(param=50)
  out = brake_test(drive, 60, brake_msg=BRAKE_ACC, a_ego=-1.0)
  assert braking(out)
  out = drive.step(0.3, v_ego=15., lka=True, brake_msg=BRAKE_ACC, a_ego=-1.6)  # 0.6 harder than asked
  assert drive.cc.c1_bt_active and drive.cc.c1_bt_release
  assert out[FSM2][2] & 0x20 == 0 and out[FSM2][5] & 0x20 == 0  # flags off, pressure can release
  out = drive.step(0.3, v_ego=15., lka=True, brake_msg=BRAKE_NONE, a_ego=-1.2)
  assert drive.cc.c1_bt_release  # not yet back within 0.1 of the request
  out = drive.step(0.3, v_ego=15., lka=True, brake_msg=BRAKE_NONE, a_ego=-1.05)
  assert not drive.cc.c1_bt_release and braking(out)


def test_fsm0_relayed_with_tracking_while_op_brakes(flag_file):
  drive = Drive(param=50)
  for _ in range(5):
    drive.step(0.3, v_ego=15., fsm0=FSM0_ACTIVE)
  for _ in range(60):
    drive.step(0.3, v_ego=15., lka=True, brake_msg=BRAKE_ACC, fsm0=FSM0_ACTIVE)
  fsm0 = [r for r in drive.relays if r[0] == FSM0_ID]
  assert len(fsm0) == 1 and fsm0[0][2] == 0 and fsm0[0][1] == FSM0_ACTIVE[:7] + bytes([FSM0_ACTIVE[7] | 1])


def test_no_fsm0_relay_without_the_test():
  drive = Drive(param=50)
  for _ in range(30):
    drive.step(0.3, v_ego=15., fsm0=FSM0_ACTIVE)
    assert not drive.relays
