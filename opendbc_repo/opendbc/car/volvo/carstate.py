from opendbc.can import CANParser
from opendbc.car import Bus, ButtonType, create_button_events, structs
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.interfaces import CarStateBase
from opendbc.car.volvo.values import C1_BUTTON_STOCK_SIGNALS, DBC, volvo_c1_acc_owns_brake, volvo_c1_aol_allowed

GearShifter = structs.CarState.GearShifter


class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    self.c1_msg_pscm = {}
    self.c1_lka_torque = 0
    self.c1_brake_hold = False  # stock ACC holding the car at standstill (0x1B0 byte 2 bit 6)
    # The stock CCButtons frame as the CEM sent it (bytes 0-6), so a Resume can carry the same bytes a real press does
    self.c1_buttons_stock = None
    self.c1_lka_on = False  # dash LKA switch, CEM0 (0x210) byte 0 bit 2
    self.c1_aol_allowed = False
    self.c1_button_states = dict.fromkeys(
      ("ACCOnOffBtn", "ACCStopBtn", "ACCSetBtn", "ACCResumeBtn", "ACCMinusBtn", "TimeGapIncreaseBtn", "TimeGapDecreaseBtn"), False
    )

  def update(self, can_parsers):
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
    # While stock ACC brakes or holds the car it trips the same pressure bits as the pedal. Ignore them then, same as
    # volvo_c1.h. A pedal tap makes ACC cancel, which clears the ACC-owned state. See volvo_c1_acc_owns_brake.
    brake_status = int(cp.vl["BrakeMessages"]["BrakeStatus"])
    self.c1_brake_hold = bool(cp.vl["BrakeMessages"]["BrakeHold"])
    acc_active = bool(cp_cam.vl["FSM0"]["ACCStatusActive"])
    pressure_bits = bool(cp.vl["PedalandBrake"]["BrakePedalActive2"] or cp.vl["PedalandBrake"]["BrakePedalActive"])
    ret.brakePressed = pressure_bits and not volvo_c1_acc_owns_brake(brake_status, self.c1_brake_hold, acc_active)
    # 5 = ACC braking, 1 = ACC transition/hold (0xA1 / 0xE1)
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
    self.c1_buttons_stock = {name: int(cp.vl["CCButtons"][name]) for name in C1_BUTTON_STOCK_SIGNALS}

    # Always on lateral follows the driver's LKA on/off, and only once stock ACC has been switched on (standby or active)
    self.c1_lka_on = bool(cp.vl["CEM0"]["LKA_State"])
    self.c1_aol_allowed = volvo_c1_aol_allowed(self.c1_lka_on, ret.cruiseState.available)

    self.c1_msg_pscm = cp.vl["PSCM1"]
    self.c1_lka_torque = int(cp.vl["PSCM1"]["LKATorque"])

    return ret

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
