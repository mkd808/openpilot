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
