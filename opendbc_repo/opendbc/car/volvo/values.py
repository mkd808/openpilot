from dataclasses import dataclass, field
from enum import IntFlag
from opendbc.car import DT_CTRL, Bus, CarSpecs, DbcDict, PlatformConfig, Platforms, uds
from opendbc.car.structs import CarParams
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.docs_definitions import CarDocs, CarHarness, CarParts
from opendbc.car.fw_query_definitions import FwQueryConfig, Request, p16
from opendbc.car.lateral import AngleSteeringLimits

# C1 support is adapted from the original dragonpilot V40 port:
# https://github.com/dragonpilot/dragonpilot/commit/773dce507082d931236b64dca8024dce9625446f
# V60 (EUCD) support is adapted from sunnypilot's volvo_oplong port.
Ecu = CarParams.Ecu


class VolvoSafetyFlags(IntFlag):
  SPA = 1
  C1 = 2
  LCA2_MOD15 = 4
  EUCD = 8
  C1_LONG = 16  # V40 experimental OP longitudinal, with C1
  C1_BRAKE_TEST = 32  # test branch only, with C1_LONG: OP's own braking for the button-triggered brake test


class CarControllerParams:
  C1_N_ZERO_TORQUE = 12
  C1_ANGLE_ERROR = 20.0
  C1_ANGLE_DELTA_BP = [0.0, 8.33, 13.89, 19.44, 25.0, 30.55, 36.1]
  C1_ANGLE_DELTA_UP = [2.0, 1.2, 0.25, 0.20, 0.15, 0.10, 0.10]
  C1_ANGLE_DELTA_DOWN = [2.0, 1.2, 0.25, 0.20, 0.15, 0.10, 0.10]
  C1_ANGLE_LIMITS = AngleSteeringLimits(359.9, (C1_ANGLE_DELTA_BP, C1_ANGLE_DELTA_UP), (C1_ANGLE_DELTA_BP, C1_ANGLE_DELTA_DOWN))
  # V40 stop and go: Resume held this many frames at a held standstill, one burst per hold. volvo_c1.h lets 40 through.
  C1_RESUME_HOLD_FRAMES = int(0.3 / DT_CTRL)
  C1_RESUME_REARM_FRAMES = int(2.0 / DT_CTRL)  # no second burst this soon after one, even if the hold flickers
  # V40 longitudinal encoders (0x160/0x280), fit to 7 stock-ACC routes
  C1_ACCEL_REQUEST_MIN = -5.12  # 0x160 ACC_AccelRequest field range, 7 bits at 0.08
  C1_ACCEL_REQUEST_MAX = 5.04
  # 0x280 traction is 93 * request in drive and coast, no offset (MAE 4.8 / 3.5 counts; the V60's 97/94 fit worse)
  C1_TRACTION_GAIN = 93.
  # Mode from the request alone, with the camera's hysteresis: brake below -0.6 until the request is positive,
  # drive above zero until it goes negative. Agrees with the stock 0x280 mode in 89% of ACC frames (V60's: 84%).
  C1_BRAKE_ENTER = -0.60  # m/s^2
  C1_BRAKE_EXIT = 0.04
  C1_DRIVE_ENTER = 0.04
  C1_DRIVE_EXIT = -0.04
  # V40 OP longitudinal: drive and coast only, the camera keeps all braking. Step 2a's own hydraulic braking (brake
  # mode and both flags while the camera only coasted) set the brake status to ACC braking but built no pressure, and
  # after about 1.5 s stock ACC shut down as unavailable until an ignition cycle (route 7c7d3d5e5ff31d51/00000013,
  # 205.6-207.1 s). Until the missing piece is found, OP never sends brake mode or the flags. Whenever the camera
  # brakes, or the car is at a standstill, its frames go out unchanged. volvo_c1.h allows -0.56 to +2.0.
  C1_LONG_ACCEL_MIN = -0.48  # m/s^2; below it the camera's frames go out unchanged
  C1_LONG_ACCEL_MAX = 1.0
  C1_LONG_JERK_UP = 2.0  # m/s^3
  C1_LONG_JERK_DOWN = 4.0
  # First drives: never more accel than the camera's own request, so OP can only be gentler than stock ACC
  C1_LONG_CAMERA_CAP = True
  # Stop handoff: OP never finishes a stop the camera isn't making. If OP brought the car to a standstill while the
  # camera still asks to drive (no lead, below the set speed), the camera's frames would go out at the standstill and
  # pull away. So below this speed, while OP asks to slow down and the camera isn't braking but asks to accelerate,
  # OP cancels stock ACC: the driver gets openpilot's disengage alert, the car coasts and the driver brakes. Until ACC
  # is off, OP's own frames keep going out, also at a standstill.
  C1_LONG_HANDOFF_SPEED = 14 * CV.KPH_TO_MS
  C1_LONG_HANDOFF_ACCEL = -0.3  # m/s^2; OP asks for more than gentle coasting
  # Brake test (test branch only): with the flag file on the device, flipping the LKA switch with OP long active, above
  # C1_BRAKE_TEST_MIN_SPEED, no lead shown and the camera not braking sends a fixed brake request the way the camera
  # brakes (brake mode, traction 0, both hydraulic flags) until the speed has dropped C1_BRAKE_TEST_DROP or
  # C1_BRAKE_TEST_MAX_S has passed. Gas, brake, a lead, the camera braking or OP long ending stop it at once.
  C1_BRAKE_TEST_FLAG = "/data/volvo_c1_brake_test"
  C1_BRAKE_TEST_ACCEL = -1.0  # m/s^2; the camera's -1.0 gave about -0.8 and pressure ~35 (route 00000016)
  C1_BRAKE_TEST_MIN_SPEED = 30 * CV.KPH_TO_MS
  C1_BRAKE_TEST_DROP = 15 * CV.KPH_TO_MS
  C1_BRAKE_TEST_MAX_S = 4.0
  C1_BRAKE_TEST_FLAG_CHECK_FRAMES = 100  # look for the flag file once a second
  # Test 2: while OP brakes (and this long after), OP relays the car's 0x1B0 to the camera with BrakeStatus and both
  # pressures zeroed whenever the car reports ACC braking (status 5); volvo_c1.h allows this 0.3 s after its last braking 0x160
  C1_BRAKE_TEST_RELAY_TAIL_FRAMES = 20
  # Test 3: with the camera no longer faulting the brake module ramped past OP's request (route 0000001b: -0.96 asked,
  # -2.4 reached). OP drops its hydraulic flags while the car brakes this much harder than asked, and brakes again
  # once it's back within C1_BRAKE_TEST_REAPPLY of the request. Pressure released within ~0.6 s when the flags dropped.
  C1_BRAKE_TEST_OVERSHOOT = 0.3  # m/s^2
  C1_BRAKE_TEST_REAPPLY = 0.1  # m/s^2
  # Test 4: on the camera's own braking the car decelerates ~0.25-0.3 m/s^2 less than the 0x160 request (actual =
  # 0.084 * count + 0.28, routes 00000012/13/16), so the request comes from this table of actual decel -> request,
  # like the V60's FSM4 brake target table. Ramped in like the camera does rather than stepped (a step wound the brake
  # module up: pressure 1.5-3 s late, then -1.5 for -0.96 asked), and eased off by up to C1_BRAKE_TEST_EASE when the
  # car brakes harder than the target. It never asks for more than the table.
  C1_BRAKE_ACCEL_BP = [-2.0, -1.5, -1.0, -0.75, -0.45, -0.1, 0.0]
  C1_BRAKE_REQUEST_V = [-2.2, -1.75, -1.25, -0.96, -0.76, -0.40, -0.32]
  C1_BRAKE_TEST_RAMP = 1.0  # m/s^3 into the brake request
  # Test 6 (route 0000001d): pressure only starts ~0.4-0.5 s after the request passes ~-1.1, so most of the ~1.9 s
  # onset was the 1 m/s^3 ramp from cruising. Down to C1_BRAKE_TEST_FAST_UNTIL, where the car only engine-brakes
  # anyway, the request ramps at C1_BRAKE_TEST_FAST_RAMP; below it at C1_BRAKE_TEST_RAMP. Easing harder (gain 1.0, was
  # 0.6) keeps the overshoot under the limiter so the flags rarely drop.
  C1_BRAKE_TEST_FAST_UNTIL = -0.5  # m/s^2
  C1_BRAKE_TEST_FAST_RAMP = 3.5  # m/s^3
  C1_BRAKE_TEST_EASE_GAIN = 1.0
  # Step 2b: OP's own planner brakes with the brake test's path (table, ramps, easing, limiter, relays), only with the
  # flag file on the device, OP long active, a lead shown, above C1_OP_BRAKE_MIN_SPEED, no pedal and the camera not
  # braking, once the planner asks for less than C1_OP_BRAKE_ENTER. Never deeper than C1_OP_BRAKE_ACCEL_MIN actual.
  # When the need ends (or a condition fails) the request ramps back to coasting at C1_LONG_JERK_UP. Stops stay the
  # camera's (and the stop handoff's); the panda rules are the brake test's (C1_BRAKE_TEST flag).
  C1_OP_BRAKE_FLAG = "/data/volvo_c1_op_braking"
  C1_OP_BRAKE_ACCEL_MIN = -1.0  # m/s^2 actual
  C1_OP_BRAKE_ENTER = -0.55  # m/s^2: the planner wants more than coasting gives
  C1_OP_BRAKE_EXIT = -0.4  # m/s^2: the planner is back to coasting
  C1_OP_BRAKE_MIN_SPEED = 20 * CV.KPH_TO_MS
  C1_BRAKE_TEST_EASE = 0.5  # m/s^2
  C1_BRAKE_TEST_PRESSURE_SEEN = 10  # 0x1B0 BrakePress0 counts: pressure has started building
  # Test 5: while the camera brakes, 0x280 byte 5 (SpeedTarget) is the speed it brakes towards (about the lead's speed,
  # 0 when stopping hard; routes 13/16) instead of its cruising target. OP's brake frames send the test's end speed.
  # volvo_c1.h never lets it go below the current speed - 17 km/h.


@dataclass
class VolvoCarDocs(CarDocs):
  package: str = "Pilot Assist & Adaptive Cruise Control"
  car_parts: CarParts = field(default_factory=lambda: CarParts.common([CarHarness.custom]))


@dataclass
class VolvoC1PlatformConfig(PlatformConfig):
  # Delphi ESR tracks (0x500..0x53F, 20 Hz) are on bus 1; read-only, nothing is sent to the radar.
  dbc_dict: DbcDict = field(default_factory=lambda: {Bus.pt: 'volvo_v40_2017_pt', Bus.cam: 'volvo_v40_2017_pt', Bus.radar: 'ESR'})


@dataclass
class VolvoCMAPlatformConfig(PlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: {
    Bus.main: 'volvo_mid_1',
    Bus.party: 'volvo_mid_1',
    Bus.pt: 'volvo_front_1_cma',
  })

@dataclass
class VolvoSPAPlatformConfig(PlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: {
    Bus.main: 'volvo_mid_1',
    Bus.party: 'volvo_mid_1',
    Bus.pt: 'volvo_front_1_spa',
  })

@dataclass
class VolvoEUCDPlatformConfig(PlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: {Bus.pt: 'volvo_v60_2015_pt', Bus.radar: 'ESR'})


class CAR(Platforms):
  VOLVO_V40 = VolvoC1PlatformConfig(
    [VolvoCarDocs("Volvo V40 2013-19")], CarSpecs(mass=1610, wheelbase=2.647, steerRatio=14.7, centerToFrontRatio=0.44, minSteerSpeed=1.0 * CV.KPH_TO_MS)
  )

  VOLVO_XC40_RECHARGE = VolvoCMAPlatformConfig(
      [VolvoCarDocs("Volvo XC40 Recharge 2021-23")],
      CarSpecs(
        mass=2170,
        wheelbase=2.702,
        steerRatio=15.8,
        centerToFrontRatio=0.52,
      ),
    )
  VOLVO_S60_RECHARGE = VolvoSPAPlatformConfig(
      [VolvoCarDocs("Volvo S60 Recharge 2024")],
      CarSpecs(
        mass=2020,
        wheelbase=2.872,
        steerRatio=16.2,
        centerToFrontRatio=0.516,
      ),
    )
  POLESTAR_2 = VolvoSPAPlatformConfig(
      [VolvoCarDocs("Polestar 2 2020-25")],
      CarSpecs(
        mass=2123,
        wheelbase=2.735,
        steerRatio=15.8,
        centerToFrontRatio=0.52,
      ),
    )
  VOLVO_V60 = VolvoEUCDPlatformConfig(
    [VolvoCarDocs("Volvo V60 2010-18")],
    # minSteerSpeed stays 0: EUCD_MIN_STEER_SPEED is the port's limit, above 0 an alert fires at every first stop
    CarSpecs(mass=1750, wheelbase=2.776, steerRatio=15.0, centerToFrontRatio=0.44, minSteerSpeed=0.),
  )

# The V60's CAN fingerprint is not exhaustive: it is identified by its engine FW. The others by CAN only.
VOLVO_VERSION_REQUEST = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + p16(0xf1a2)
VOLVO_VERSION_RESPONSE = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER + 0x40]) + p16(0xf1a2)
FW_QUERY_CONFIG = FwQueryConfig(
  requests=[Request([VOLVO_VERSION_REQUEST], [VOLVO_VERSION_RESPONSE], bus=0)],
  fw_version_regex=br"[0-9]{8} [A-Z]{2}\x00{13}",
)
DBC = CAR.create_dbc_map()


def volvo_c1_aol_allowed(lka_state: bool, acc_on: bool) -> bool:
  # Always on lateral needs the dash LKA switch on and stock ACC switched on (standby or active, FSM0 ACCStatusOnOff).
  # With the ignition on and ACC still off (0x30 byte 7 = 0x88) the LKA bit alone is not enough.
  return lka_state and acc_on


def volvo_c1_stock_acc_owns_cruise(car_fingerprint, pcm_cruise: bool, controls_enabled: bool) -> bool:
  # V40 is lateral-only: stock ACC owns speed, so OP never needs to cancel it. A cancel sent after OP disengages
  # (for example when stock ACC braking trips the brake-pressure bits) can drop stock ACC.
  # A driver brake still drops stock ACC by itself.
  return car_fingerprint == CAR.VOLVO_V40 and pcm_cruise and not controls_enabled


# CCButtons (0x10) signals other than the button bits in byte 7: the CEM's own bytes 0-6. A Resume that repeats them
# looks like a real press; the panda only lets it through if they equal the last stock frame it saw.
C1_BUTTON_STOCK_SIGNALS = ("byte0", "byte1", "byte2", "byte3", "byte4", "byte5", "byte6",
                           "ACCMinusBtn", "TimeGapIncreaseBtn", "TimeGapDecreaseBtn")


def volvo_c1_acc_owns_brake(brake_status: int, brake_hold: bool, acc_active: bool) -> bool:
  """True while stock ACC (not the driver) is braking or holding the car, so the 0x55 pressure bits are not a pedal.
  Mirrors volvo_c1_update_brake_pressed() in volvo_c1.h.
  0x1B0 byte 2: 0xA5 = ACC braking (status 5, always ACC), 0xA1 = just before/after a braking event (status 1),
  0xE0/0xE1 = ACC holding at standstill (hold bit). 0xA1 and the hold bit only count while ACC is active (0x30 byte 7 bit 2)."""
  return brake_status == 5 or (acc_active and (brake_status == 1 or brake_hold))


# After ACC braking ends its pressure bleeds off for up to ~0.7 s: while ACC stays active the pressure bits keep counting
# as ACC's this long (100 Hz frames) after ACC last owned the brake. Mirrors VOLVO_C1_ACC_BRAKE_GRACE_US in volvo_c1.h.
C1_ACC_BRAKE_GRACE_FRAMES = 100
