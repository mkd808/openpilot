from opendbc.car.volvo.eucd_params import LongMode
from opendbc.car.volvo.values import CarControllerParams


def create_c1_pscm_message(packer, msg_pscm: dict):
  values = {
    "LKATorque": 0,
    "SteeringAngleServo": msg_pscm["SteeringAngleServo"],
    "byte0": msg_pscm["byte0"],
    "byte3": msg_pscm["byte3"],
    "byte4": msg_pscm["byte4"],
    "byte7": msg_pscm["byte7"],
    "LKAActive": int(msg_pscm["LKAActive"]) & 0xD,
  }
  return packer.make_can_msg("PSCM1", 2, values)


def create_c1_checksum(data: bytes) -> int:
  angle_raw = ((data[4] & 0x3F) << 8) | data[5]
  direction = data[7] & 0x3
  checksum_sum = (data[3] + direction + angle_raw + (angle_raw >> 8)) & 0xFF
  return checksum_sum ^ 0xFF


def create_c1_steering_control(packer, apply_angle: float, lat_active: bool):
  values = {
    "SET_X_E3": 0xE3,
    "SET_X_B4": 0xB4,
    "SET_X_08": 0x08,
    "TrqLim": 0,
    "LKAAngleReq": apply_angle,
    "LKASteerDirection": 3 if lat_active else 0,
    "SET_X_25": 0x25,
    "SET_X_02": 0x02,
  }
  data = packer.make_can_msg("FSM1", 0, values)[1]
  values["Checksum"] = create_c1_checksum(data)
  return packer.make_can_msg("FSM1", 0, values)


def create_c1_cancel(packer):
  return packer.make_can_msg("CCButtons", 0, {"ACCStopBtn": 1})


def create_c1_resume(packer, stock: dict | None = None):
  """Resume button. With the stock frame's bytes 0-6 (carstate.c1_buttons_stock) it carries what a real press carries;
  without them it falls back to the cancel frame's shape (all other bytes 0)."""
  values = {**stock} if stock else {}
  values["ACCResumeBtn"] = 1
  return packer.make_can_msg("CCButtons", 0, values)


# V40 longitudinal: the camera's 0x160 (FSM2, accel request) and 0x280 (FSM4, mode and traction) carry no counter or
# checksum, so OP's frames are the stock frame with its own values. Copying the stock frame keeps the camera's masked
# copy (every 5th 0x160) and every bit OP doesn't decide.
# The masked copy is the frame before it ANDed with this:
C1_FSM2_MASK = bytes((0x6F, 0xFF, 0x2A, 0xBF, 0xFF, 0x2F, 0xFF, 0x7F))

def c1_next_long_mode(mode: LongMode, accel: float) -> LongMode:
  CCP = CarControllerParams
  if mode == LongMode.BRAKE and accel < CCP.C1_BRAKE_EXIT:
    return LongMode.BRAKE
  if mode == LongMode.DRIVE and accel >= CCP.C1_DRIVE_EXIT:
    return LongMode.DRIVE
  if accel < CCP.C1_BRAKE_ENTER:
    return LongMode.BRAKE
  if accel >= CCP.C1_DRIVE_ENTER:
    return LongMode.DRIVE
  return LongMode.COAST


def c1_traction(accel: float, mode: LongMode) -> int:
  # 0x280 ACC_TractionRequest: unsigned in drive, signed in coast, 0 in brake
  if mode == LongMode.BRAKE:
    return 0
  traction = round(CarControllerParams.C1_TRACTION_GAIN * accel)
  if mode == LongMode.DRIVE:
    return min(max(traction, 0), 255)
  return min(max(traction, -127), -1) & 0xFF


def create_c1_acc_request(packer, stock_fsm2: dict, accel: float, brake: bool | None = None):
  # 0x160 with OP's accel request; brake sets both hydraulic request flags
  CCP = CarControllerParams
  values = {**stock_fsm2, "ACC_AccelRequest": min(max(accel, CCP.C1_ACCEL_REQUEST_MIN), CCP.C1_ACCEL_REQUEST_MAX)}
  if brake is not None:
    values["ACC_BrakeRequest"] = int(brake)
    values["ACC_BrakeRequest_B"] = int(brake)
  return packer.make_can_msg("FSM2", 0, values)


def create_c1_brake_relay(packer, stock: dict):
  # The car's 0x1B0 for the camera (bus 2): while the car reports ACC braking (status 5), with the status and both
  # pressures zeroed so the camera doesn't see braking it didn't ask for; otherwise unchanged
  values = {**stock}
  if int(stock["BrakeStatus"]) == 5:
    values.update({"BrakeStatus": 0, "BrakePress0": 0, "BrakePress1": 0})
  return packer.make_can_msg("BrakeMessages", 2, values)


def create_c1_fsm0_relay(packer, stock: dict, tracking: bool):
  # The camera's FSM0 (0x30) for the car (bus 0): ACCStatusTracking set while OP brakes and ACC is active, as the
  # camera sets it in every frame of its own braking; otherwise unchanged
  values = {**stock}
  if tracking and int(stock["ACCStatusActive"]):
    values["ACCStatusTracking"] = 1
  return packer.make_can_msg("FSM0", 0, values)


def create_c1_acc_mode(packer, stock_fsm4: dict, accel: float | None = None, mode: LongMode | None = None,
                       target_kph: float | None = None):
  # 0x280 with OP's mode and its traction for accel; without a mode the stock frame is copied. In brake mode a target
  # speed (byte 5 SpeedTarget, the speed the braking aims for) replaces the camera's when it is lower.
  values = {**stock_fsm4}
  if mode is not None:
    values["ACC_Mode"] = int(mode)
    values["ACC_TractionRequest"] = c1_traction(accel, mode)
    if mode == LongMode.BRAKE and target_kph is not None:
      values["SpeedTarget"] = min(int(stock_fsm4["SpeedTarget"]), max(int(round(target_kph)), 0))
  return packer.make_can_msg("FSM4", 0, values)
