from cereal import custom
from opendbc.can import CANParser
from opendbc.car import Bus, ButtonType, create_button_events, structs
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.interfaces import CarStateBase
from opendbc.car.volvo.values import DBC

GearShifter = structs.CarState.GearShifter


class CarState(CarStateBase):
  def __init__(self, CP, FPCP):
    super().__init__(CP, FPCP)
    self.c1_msg_pscm = {}
    self.c1_lka_torque = 0
    self.c1_button_states = dict.fromkeys(
      ("ACCOnOffBtn", "ACCStopBtn", "ACCSetBtn", "ACCResumeBtn", "ACCMinusBtn", "TimeGapIncreaseBtn", "TimeGapDecreaseBtn"), False
    )

  def update(self, can_parsers, starpilot_toggles):
    cp = can_parsers[Bus.pt]
    cp_cam = can_parsers[Bus.cam]
    ret = structs.CarState()

    ret.vEgoRaw = cp.vl["VehicleSpeed1"]["VehicleSpeed"] * CV.KPH_TO_MS
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = ret.vEgoRaw < 0.1

    ret.steeringAngleDeg = cp.vl["PSCM1"]["SteeringAngleServo"]
    ret.steeringTorque = cp.vl["PSCM1"]["LKATorque"]
    ret.steeringPressed = False

    ret.gasPressed = cp.vl["PedalandBrake"]["AccPedal"] > 5.0
    # BrakeStatus 5 = stock ACC braking (byte 2 = 0xA5, never set by the pedal). ACC braking trips the same pressure
    # bits as the pedal, so ignore them while it is 5, same as volvo.h. A pedal tap makes ACC cancel and the status drop.
    brake_status = int(cp.vl["BrakeMessages"]["BrakeStatus"])
    pressure_bits = bool(cp.vl["PedalandBrake"]["BrakePedalActive2"] or cp.vl["PedalandBrake"]["BrakePedalActive"])
    ret.brakePressed = pressure_bits and brake_status != 5
    # 5 = ACC braking, 1 = ACC hold/transition (0xA1 around a braking event, 0xE1 at standstill). Only used to skip OP's ACC cancel.
    ret.brakeHoldActive = brake_status in (1, 5)

    ret.gearShifter = {
      0: GearShifter.park,
      1: GearShifter.reverse,
      2: GearShifter.neutral,
      3: GearShifter.drive,
    }.get(int(cp.vl["TCM0"]["GearShifter"]), GearShifter.unknown)

    ret.cruiseState.available = bool(cp_cam.vl["FSM0"]["ACCStatusOnOff"])
    ret.cruiseState.enabled = bool(cp_cam.vl["FSM0"]["ACCStatusActive"])
    ret.cruiseState.speed = cp.vl["ACC"]["SpeedTargetACC"] * CV.KPH_TO_MS
    ret.cruiseState.nonAdaptive = False
    ret.cruiseState.standstill = ret.standstill

    turn_signal = int(cp.vl["MiscCarInfo"]["TurnSignal"])
    ret.leftBlinker, ret.rightBlinker = self.update_blinker_from_stalk(50, turn_signal == 1, turn_signal == 3)
    ret.doorOpen = False
    ret.seatbeltUnlatched = False

    button_types = {
      "ACCOnOffBtn": ButtonType.mainCruise,
      "ACCStopBtn": ButtonType.cancel,
      "ACCSetBtn": ButtonType.setCruise,
      "ACCResumeBtn": ButtonType.resumeCruise,
      "ACCMinusBtn": ButtonType.decelCruise,
      "TimeGapIncreaseBtn": ButtonType.gapAdjustCruise,
      "TimeGapDecreaseBtn": ButtonType.gapAdjustCruise,
    }
    button_events = []
    for signal, button_type in button_types.items():
      pressed = bool(cp.vl["CCButtons"][signal])
      button_events.extend(create_button_events(pressed, self.c1_button_states[signal], {True: button_type}))
      self.c1_button_states[signal] = pressed
    ret.buttonEvents = button_events

    # Dash Lane Keep Assist button, read from CEM0 (0x210) byte 0 bit 2.
    # Tentative: confirm with a stopwatch test whether this is momentary
    # (high only while held) or a latched on/off state.
    lka_state = bool(cp.vl["CEM0"]["LKA_State"])

    # Always on lateral follows the driver's LKA on/off.
    fp_ret = custom.StarPilotCarState.new_message()
    fp_ret.alwaysOnLateralAllowed = lka_state

    self.c1_msg_pscm = cp.vl["PSCM1"]
    self.c1_lka_torque = int(cp.vl["PSCM1"]["LKATorque"])

    return ret, fp_ret

  @staticmethod
  def get_can_parsers(CP):
    return {
      Bus.pt: CANParser(
        DBC[CP.carFingerprint][Bus.pt],
        [
          ("VehicleSpeed1", 50),
          ("CCButtons", 100),
          ("PSCM1", 50),
          ("PedalandBrake", 100),
          ("BrakeMessages", 50),
          ("TCM0", 10),
          ("ACC", 17),
          ("MiscCarInfo", 25),
          ("CEM0", 40),
        ],
        0,
      ),
      Bus.cam: CANParser(
        DBC[CP.carFingerprint][Bus.cam],
        [
          ("FSM0", 100),
          ("FSM1", 50),
        ],
        2,
      ),
    }
