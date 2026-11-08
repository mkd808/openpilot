import pytest

from cereal import custom
from opendbc.can.packer import CANPacker
from opendbc.car import Bus, ButtonType, CanData, structs
from opendbc.car.volvo.carstate import CarState
from opendbc.car.volvo.interface import CarInterface
from opendbc.car.volvo.values import CAR, DBC


def _can_data(msg):
  address, data, bus = msg
  return CanData(address, data, bus)


def test_c1_carstate_decodes_vehicle_and_cruise_signals():
  cp = CarInterface.get_non_essential_params(CAR.VOLVO_V40)
  cs = CarState(cp, custom.StarPilotCarParams.new_message())
  parsers = CarState.get_can_parsers(cp)
  packer = CANPacker(DBC[cp.carFingerprint][Bus.pt])

  messages = [
    packer.make_can_msg("VehicleSpeed1", 0, {"VehicleSpeed": 72}),
    packer.make_can_msg("CCButtons", 0, {"ACCStopBtn": 1, "ACCSetBtn": 1}),
    packer.make_can_msg("PSCM1", 0, {"SteeringAngleServo": -12.5, "LKATorque": 7}),
    packer.make_can_msg("PedalandBrake", 0, {"AccPedal": 6, "BrakePedalActive2": 1}),
    packer.make_can_msg("TCM0", 0, {"GearShifter": 3}),
    packer.make_can_msg("ACC", 0, {"SpeedTargetACC": 100}),
    packer.make_can_msg("MiscCarInfo", 0, {"TurnSignal": 1}),
    packer.make_can_msg("FSM0", 2, {"ACCStatusOnOff": 1, "ACCStatusActive": 1}),
    packer.make_can_msg("FSM1", 2, {}),
  ]
  packets = [(1_000_000, [_can_data(msg) for msg in messages])]
  for parser in parsers.values():
    parser.update(packets)

  ret, _ = cs.update(parsers, None)
  assert ret.vEgoRaw == pytest.approx(20.0)
  assert ret.steeringAngleDeg == pytest.approx(-12.5, abs=0.05)
  assert ret.steeringTorque == 7
  assert ret.gasPressed and ret.brakePressed
  assert ret.gearShifter == structs.CarState.GearShifter.drive
  assert ret.cruiseState.available and ret.cruiseState.enabled
  assert ret.cruiseState.speed == pytest.approx(100 / 3.6)
  assert ret.leftBlinker and not ret.rightBlinker
  assert len(ret.buttonEvents) == 2
  assert any(event.type == ButtonType.cancel and event.pressed for event in ret.buttonEvents)
  assert any(event.type == ButtonType.setCruise and event.pressed for event in ret.buttonEvents)

  release = packer.make_can_msg("CCButtons", 0, {})
  packets = [(2_000_000, [_can_data(release)])]
  for parser in parsers.values():
    parser.update(packets)
  ret, _ = cs.update(parsers, None)
  assert len(ret.buttonEvents) == 2
  assert any(event.type == ButtonType.cancel and not event.pressed for event in ret.buttonEvents)
  assert any(event.type == ButtonType.setCruise and not event.pressed for event in ret.buttonEvents)


def test_acc_owns_brake_matches_logged_byte2():
  from opendbc.car.volvo.values import volvo_c1_acc_owns_brake as owns
  # byte 2 of 0x1B0 on 2026-11-08: 0xA0 idle, 0xA5 ACC braking, 0xA1 transition, 0xE0/0xE1 ACC hold at standstill
  assert not owns(0, False, False) and not owns(0, False, True)       # 0xA0
  assert owns(5, False, False) and owns(5, False, True)               # 0xA5, always ACC
  assert owns(1, False, True) and not owns(1, False, False)           # 0xA1 only while ACC active
  assert owns(0, True, True) and owns(1, True, True)                  # 0xE0 / 0xE1 while ACC active
  assert not owns(0, True, False) and not owns(1, True, False)        # hold bit without ACC active is not ACC
