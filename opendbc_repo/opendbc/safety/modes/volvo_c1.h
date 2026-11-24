#pragma once
#include "opendbc/safety/declarations.h"
#define VOLVO_C1_BUTTONS 0x10U
#define VOLVO_C1_FSM_0 0x30U
#define VOLVO_C1_FSM_1 0xD0U
#define VOLVO_C1_PSCM_1 0x125U
#define VOLVO_C1_PEDAL_AND_BRAKE 0x55U
#define VOLVO_C1_SPEED 0x150U
#define VOLVO_C1_TCM_0 0x40U
#define VOLVO_C1_BRAKE_MESSAGES 0x1B0U  // BrakeMessages: byte 2 = BrakeStatus (bits 0-2) + hold flag (bit 6)
#define VOLVO_C1_CEM_0 0x210U  // CEM0, dash LKA switch
#define VOLVO_C1_ACC_REQUEST 0x160U  // camera ACC accel request and hydraulic brake flags
#define VOLVO_C1_ACC_MODE 0x280U  // camera ACC mode and traction
#define VOLVO_C1_PARAM_LONG 16U  // experimental OP longitudinal, drive and coast only
#define VOLVO_MAIN_BUS 0U
#define VOLVO_PARTY_BUS 2U
#define VOLVO_C1_ANGLE_DEG_TO_CAN 22.753128F
#define VOLVO_C1_MAX_ANGLE_CAN 8189
#define VOLVO_C1_RELAY_ANGLE_TOLERANCE 2
static int volvo_c1_pscm_angle(const CANPacket_t *msg) {
  const uint32_t raw_angle = ((uint32_t)msg->data[5] << 8U) | (uint32_t)msg->data[6];
  return (int)raw_angle - 32768;
}

static int volvo_c1_fsm_angle(const CANPacket_t *msg) {
  const uint32_t raw_angle = (((uint32_t)msg->data[4] & 0x3FU) << 8U) | (uint32_t)msg->data[5];
  return (int)raw_angle - 8192;
}

static uint8_t volvo_c1_fsm_checksum(const CANPacket_t *msg) {
  const unsigned int angle_raw = (((unsigned int)msg->data[4] & 0x3FU) << 8U) | (unsigned int)msg->data[5];
  const unsigned int direction = (unsigned int)msg->data[7] & 0x3U;
  const unsigned int checksum_sum = ((unsigned int)msg->data[3] + direction + angle_raw + (angle_raw >> 8U)) & 0xFFU;
  const unsigned int inverted_checksum = checksum_sum ^ 0xFFU;
  return (uint8_t)inverted_checksum;
}

// Stock ACC braking, and its standstill hold, trip the same pressure bits in 0x55 as the driver's
// pedal. 0x1B0 byte 2 tells them apart: 0xA5 (BrakeStatus 5) is ACC braking and never appears for the
// pedal; 0xA1 (status 1) is just before/after a braking event and 0xE0/0xE1 (bit 6, the hold flag) is
// ACC holding the car at standstill, both only while stock ACC is active (0x30 byte 7 bit 2). The
// pressure bits are ignored while ACC owns the brake. When the driver taps the pedal, ACC cancels and the
// state clears within about a frame, which brings the pedal back at once. 0x1B0 is in the rx checks, so
// losing it invalidates controls instead of leaving the mask stuck. Mirrors volvo_c1_acc_owns_brake().
static bool volvo_c1_pressure_bits = false;
static unsigned int volvo_c1_brake_status = 0U;
static bool volvo_c1_brake_hold = false;
static bool volvo_c1_acc_active = false;

static void volvo_c1_update_brake_pressed(void) {
  const bool acc_owns_brake = (volvo_c1_brake_status == 5U) ||
                              (volvo_c1_acc_active && ((volvo_c1_brake_status == 1U) || volvo_c1_brake_hold));
  brake_pressed = volvo_c1_pressure_bits && !acc_owns_brake;
}

// Resume button (0x10 byte 7 = 0x20) for stop and go. Stock ACC holds the car at standstill (0x1B0 hold flag)
// and waits for the driver's Resume once the lead has been gone a while. A Resume frame is only let through
// while stock ACC is active and holding, the car is stationary, controls are allowed and the driver is not on
// the pedals, and at most VOLVO_C1_RESUME_MAX_FRAMES per hold. The budget is restored when the hold ends, the
// car moves or ACC drops, so one hold gets one burst.
#define VOLVO_C1_RESUME_MAX_FRAMES 40U  // 0.4 s at 100 Hz, the car side sends 0.3 s
static unsigned int volvo_c1_resume_budget = 0U;

static void volvo_c1_resume_rearm(void) {
  if (!volvo_c1_brake_hold || !volvo_c1_acc_active || vehicle_moving) {
    volvo_c1_resume_budget = VOLVO_C1_RESUME_MAX_FRAMES;
  }
}

static bool volvo_c1_resume_allowed(void) {
  const bool ok = controls_allowed && volvo_c1_acc_active && volvo_c1_brake_hold && !vehicle_moving &&
                  !gas_pressed && !brake_pressed && (volvo_c1_resume_budget > 0U);
  if (ok) {
    volvo_c1_resume_budget--;
  }
  return ok;
}

// The last stock CCButtons frame the CEM sent on the car side. A Resume built from it looks like a real button
// press on the bus (the stream keeps its own bytes and only byte 7 changes). Bytes 1-6 must equal this frame when the
// Resume carries stock bytes; byte 0 toggles a bit every few frames and byte 7 is the button itself. Read through the
// optional RX hook, so it is not an rx check and cannot invalidate controls.
static uint8_t volvo_c1_buttons_stock[8] = {0};
static bool volvo_c1_buttons_stock_valid = false;

static bool volvo_c1_buttons_match_stock(const CANPacket_t *msg) {
  bool match = volvo_c1_buttons_stock_valid;
  for (unsigned int i = 1U; i < 7U; i++) {
    match = match && (msg->data[i] == volvo_c1_buttons_stock[i]);
  }
  return match;
}

// Optional V40 always on lateral profile: lateral without ACC engaged, with the driver's LKA switch on and stock ACC
// switched on (standby or active), in drive. Host requests never create permission: they only narrow it.
#define VOLVO_C1_AOL_GEAR_MAX_AGE_US 300000U
static bool volvo_c1_aol_enabled = false;
static bool volvo_c1_aol_lka_on = false;
static bool volvo_c1_aol_drive = false;
static bool volvo_c1_aol_gear_seen = false;
static uint32_t volvo_c1_aol_gear_ts = 0U;

static void volvo_c1_aol_reset(void) {
  volvo_c1_aol_lka_on = false;
  volvo_c1_aol_drive = false;
  volvo_c1_aol_gear_seen = false;
  volvo_c1_aol_gear_ts = 0U;
}

static uint8_t volvo_c1_aol_request(void) {
  uint8_t request = 0U;
  if (heartbeat_engaged && aol_rx_healthy() && !safety_rx_checks_invalid && !relay_malfunction &&
      (safety_get_ts_elapsed(microsecond_timer_get(), aol_host_request_ts) <= AOL_HOST_REQUEST_TIMEOUT_US)) {
    request = aol_host_axis_mask;
  }
  return request;
}

static uint8_t volvo_c1_aol_permission(void) {
  const uint8_t request = volvo_c1_aol_request();
  uint8_t permission = 0U;
  const bool drive = volvo_c1_aol_gear_seen && volvo_c1_aol_drive &&
    (safety_get_ts_elapsed(microsecond_timer_get(), volvo_c1_aol_gear_ts) <= VOLVO_C1_AOL_GEAR_MAX_AGE_US);
  // Engaged stock ACC keeps steering as before; without it the LKA switch and the ACC main are both needed.
  // Always on lateral never grants the longitudinal axis; OP longitudinal needs engaged stock ACC (volvo_c1_long_allowed).
  if (drive && (controls_allowed || (acc_main_on && volvo_c1_aol_lka_on))) {
    permission = request & 1U;
  }
  return permission;
}

// Experimental OP longitudinal (param flag 16), drive and coast only. Stock ACC stays engaged and keeps the brakes.
// OP's 0x160/0x280 must be one of the camera's frames from the last 100 ms with only the request (0x160 byte 7 bits
// 6-0) or the mode and traction (0x280 byte 6 bits 1-0, byte 7) replaced, so the hydraulic brake flags and the masked
// copies always come from the camera. While the camera brakes (0x160 brake flags, 0x280 brake mode) only its frame,
// unchanged, may go out. The camera's frames stop at the panda only while OP may send its own.
#define VOLVO_C1_CAM_FRAMES 5U
#define VOLVO_C1_CAM_FRAME_MAX_AGE_US 100000U
#define VOLVO_C1_LONG_REQUEST_MIN (-7)  // counts from 64 at ~0.08 m/s^2: -0.56, short of the camera's brake mode
#define VOLVO_C1_LONG_REQUEST_MAX 25    // +2.0 m/s^2
#define VOLVO_C1_LONG_TRACTION_MAX 186U  // drive traction for +2.0 m/s^2 (93 per m/s^2)
typedef struct {
  uint8_t data[VOLVO_C1_CAM_FRAMES][8];
  uint32_t ts[VOLVO_C1_CAM_FRAMES];
  bool valid[VOLVO_C1_CAM_FRAMES];
  uint8_t index;
} VolvoC1CamFrames;
static bool volvo_c1_long = false;
static VolvoC1CamFrames volvo_c1_cam_request;
static VolvoC1CamFrames volvo_c1_cam_mode;

static void volvo_c1_cam_frames_clear(VolvoC1CamFrames *frames) {
  frames->index = 0U;
  for (uint8_t i = 0U; i < VOLVO_C1_CAM_FRAMES; i++) {
    frames->valid[i] = false;
  }
}

static void volvo_c1_cam_frames_store(VolvoC1CamFrames *frames, const CANPacket_t *msg) {
  for (uint8_t byte = 0U; byte < 8U; byte++) {
    frames->data[frames->index][byte] = msg->data[byte];
  }
  frames->ts[frames->index] = microsecond_timer_get();
  frames->valid[frames->index] = true;
  frames->index = (frames->index + 1U) % VOLVO_C1_CAM_FRAMES;
}

static bool volvo_c1_cam_frame_fresh(const VolvoC1CamFrames *frames, uint8_t i) {
  return frames->valid[i] &&
         (safety_get_ts_elapsed(microsecond_timer_get(), frames->ts[i]) <= VOLVO_C1_CAM_FRAME_MAX_AGE_US);
}

static bool volvo_c1_long_allowed(void) {
  return volvo_c1_long && controls_allowed && !gas_pressed_prev;
}

static bool volvo_c1_acc_request_tx_ok(const CANPacket_t *msg) {
  const int request = (int)(msg->data[7] & 0x7FU) - 64;
  const bool in_limits = (request >= VOLVO_C1_LONG_REQUEST_MIN) && (request <= VOLVO_C1_LONG_REQUEST_MAX);
  bool ok = false;
  for (uint8_t i = 0U; i < VOLVO_C1_CAM_FRAMES; i++) {
    if (volvo_c1_cam_frame_fresh(&volvo_c1_cam_request, i)) {
      const uint8_t *cam = volvo_c1_cam_request.data[i];
      bool same = (msg->data[7] & 0x80U) == (cam[7] & 0x80U);
      for (uint8_t byte = 0U; byte < 7U; byte++) {
        same = same && (msg->data[byte] == cam[byte]);
      }
      // Signals: ACC_BrakeRequest (byte 2 bit 5), ACC_BrakeRequest_B (byte 5 bit 5)
      const bool cam_brakes = ((cam[2] & 0x20U) != 0U) || ((cam[5] & 0x20U) != 0U);
      const bool unchanged = same && (msg->data[7] == cam[7]);
      ok = ok || unchanged || (same && in_limits && !cam_brakes);
    }
  }
  return ok;
}

static bool volvo_c1_acc_mode_tx_ok(const CANPacket_t *msg) {
  // Signals: ACC_Mode (byte 6 bits 1-0: 0 brake, 1 coast, 2 drive), ACC_TractionRequest (byte 7, signed in coast)
  const unsigned int mode = (unsigned int)msg->data[6] & 0x3U;
  const unsigned int traction = msg->data[7];
  const bool coast_ok = (mode == 1U) && (traction >= 0x81U);  // -127 to -1: engine drag only
  const bool drive_ok = (mode == 2U) && (traction <= VOLVO_C1_LONG_TRACTION_MAX);
  bool ok = false;
  for (uint8_t i = 0U; i < VOLVO_C1_CAM_FRAMES; i++) {
    if (volvo_c1_cam_frame_fresh(&volvo_c1_cam_mode, i)) {
      const uint8_t *cam = volvo_c1_cam_mode.data[i];
      bool same = (msg->data[6] & 0xFCU) == (cam[6] & 0xFCU);
      for (uint8_t byte = 0U; byte < 6U; byte++) {
        same = same && (msg->data[byte] == cam[byte]);
      }
      const bool cam_brakes = (cam[6] & 0x3U) == 0U;
      const bool unchanged = same && (msg->data[6] == cam[6]) && (msg->data[7] == cam[7]);
      ok = ok || unchanged || (same && !cam_brakes && (coast_ok || drive_ok));
    }
  }
  return ok;
}

static void volvo_c1_rx_hook(const CANPacket_t *msg) {
  if (msg->bus == VOLVO_MAIN_BUS) {
    if (msg->addr == VOLVO_C1_PSCM_1) {
      update_sample(&angle_meas, volvo_c1_pscm_angle(msg));
    }

    if (msg->addr == VOLVO_C1_CEM_0) {
      // Signal: LKA_State (CEM0 byte 0 bit 2), the driver's LKA on/off
      volvo_c1_aol_lka_on = GET_BIT(msg, 2U);
    }

    if (msg->addr == VOLVO_C1_SPEED) {
      const uint16_t speed_raw = ((uint16_t)msg->data[6] << 8U) | msg->data[7];
      const float speed = ((float)speed_raw * 0.01f) / 3.6f;
      vehicle_moving = speed > 0.1f;
      UPDATE_VEHICLE_SPEED(speed);
      volvo_c1_resume_rearm();
    }

    if (msg->addr == VOLVO_C1_PEDAL_AND_BRAKE) {
      const unsigned int gas_raw = (((unsigned int)msg->data[1] & 0x3U) << 8U) | (unsigned int)msg->data[2];
      gas_pressed = gas_raw > 50U;  // DBC factor 0.1: greater than 5 percent
      volvo_c1_pressure_bits = GET_BIT(msg, 24U) || GET_BIT(msg, 38U);
      volvo_c1_update_brake_pressed();
    }

    if (msg->addr == VOLVO_C1_BRAKE_MESSAGES) {
      volvo_c1_brake_status = (unsigned int)msg->data[2] & 0x7U;
      volvo_c1_brake_hold = GET_BIT(msg, 22U);
      volvo_c1_update_brake_pressed();
      volvo_c1_resume_rearm();
    }
  }

  if ((msg->bus == VOLVO_PARTY_BUS) && (msg->addr == VOLVO_C1_FSM_0)) {
    // Signals: ACCStatusOnOff (bit 57), ACCStatusActive (bit 58)
    acc_main_on = GET_BIT(msg, 57U);
    volvo_c1_acc_active = GET_BIT(msg, 58U);
    volvo_c1_update_brake_pressed();
    volvo_c1_resume_rearm();
    pcm_cruise_check(volvo_c1_acc_active);
  }
}

static void volvo_c1_optional_rx(const CANPacket_t *msg) {
  if (volvo_c1_long && (msg->bus == VOLVO_PARTY_BUS) && (GET_LEN(msg) == 8U)) {
    if (msg->addr == VOLVO_C1_ACC_REQUEST) {
      volvo_c1_cam_frames_store(&volvo_c1_cam_request, msg);
    }
    if (msg->addr == VOLVO_C1_ACC_MODE) {
      volvo_c1_cam_frames_store(&volvo_c1_cam_mode, msg);
    }
  }

  if (msg->bus == VOLVO_MAIN_BUS) {
    if ((msg->addr == VOLVO_C1_BUTTONS) && (GET_LEN(msg) == 8U)) {
      for (unsigned int i = 0U; i < 8U; i++) {
        volvo_c1_buttons_stock[i] = msg->data[i];
      }
      volvo_c1_buttons_stock_valid = true;
    }

    if (volvo_c1_aol_enabled && (msg->addr == VOLVO_C1_TCM_0)) {
      volvo_c1_aol_gear_seen = GET_LEN(msg) == 8U;
      if (volvo_c1_aol_gear_seen) {
        volvo_c1_aol_gear_ts = microsecond_timer_get();
        // Signal: GearShifter (byte 5 bits 6:5: 0 P, 1 R, 2 N, 3 D)
        volvo_c1_aol_drive = (((unsigned int)msg->data[5] >> 5) & 0x3U) == 0x3U;
      }
    }
  }
}
// C1 retains its measured-angle EPS envelope; the shared angle API no longer supplies it.
static bool volvo_c1_angle_cmd_checks(int desired_angle, bool steer_control_enabled, const AngleSteeringLimits limits) {
  bool violation = false;

  if (lateral_controls_allowed() && steer_control_enabled) {
    // convert floating point angle rate limits to integers in the scale of the desired angle on CAN,
    // add 1 to not false trigger the violation. also fudge the speed by 1 m/s so rate limits are
    // always slightly above openpilot's in case we read an updated speed in between angle commands
    // TODO: this speed fudge can be much lower, look at data to determine the lowest reasonable offset
    const float fudged_speed = (vehicle_speed.min / VEHICLE_SPEED_FACTOR) - 1.;
    int delta_angle_up = (safety_interpolate(limits.angle_rate_up_lookup, fudged_speed) * limits.angle_deg_to_can) + 1.;
    int delta_angle_down = (safety_interpolate(limits.angle_rate_down_lookup, fudged_speed) * limits.angle_deg_to_can) + 1.;

    // allow down limits at zero since small floats from openpilot will be rounded to 0
    // TODO: openpilot should be cognizant of this and not send small floats
    int highest_desired_angle = desired_angle_last + ((desired_angle_last > 0) ? delta_angle_up : delta_angle_down);
    int lowest_desired_angle = desired_angle_last - ((desired_angle_last >= 0) ? delta_angle_down : delta_angle_up);

    // check that commanded angle value isn't too far from measured, used to limit torque for some safety modes
    // ensure we start moving in direction of meas while respecting relaxed rate limits if error is exceeded
    if (((vehicle_speed.values[0] / VEHICLE_SPEED_FACTOR) > 0.0F)) {
      // flipped fudge to avoid false positives
      const float fudged_speed_error = (vehicle_speed.max / VEHICLE_SPEED_FACTOR) + 1.;
      const int delta_angle_up_relaxed = (safety_interpolate(limits.angle_rate_up_lookup, fudged_speed_error) * limits.angle_deg_to_can) - 1.;
      const int delta_angle_down_relaxed = (safety_interpolate(limits.angle_rate_down_lookup, fudged_speed_error) * limits.angle_deg_to_can) - 1.;

      // the minimum and maximum angle allowed based on the measured angle
      const int lowest_desired_angle_error = angle_meas.min - 455 - 1;
      const int highest_desired_angle_error = angle_meas.max + 455 + 1;

      // the MAX is to allow the desired angle to hit the edge of the bounds and not require going under it
      if (desired_angle_last > highest_desired_angle_error) {
        const int delta = (desired_angle_last >= 0) ? delta_angle_down_relaxed : delta_angle_up_relaxed;
        highest_desired_angle = SAFETY_MAX(desired_angle_last - delta, highest_desired_angle_error);

      } else if (desired_angle_last < lowest_desired_angle_error) {
        const int delta = (desired_angle_last <= 0) ? delta_angle_down_relaxed : delta_angle_up_relaxed;
        lowest_desired_angle = SAFETY_MIN(desired_angle_last + delta, lowest_desired_angle_error);

      } else {
        // already inside error boundary, don't allow commanding outside it
        highest_desired_angle = SAFETY_MIN(highest_desired_angle, highest_desired_angle_error);
        lowest_desired_angle = SAFETY_MAX(lowest_desired_angle, lowest_desired_angle_error);
      }

      // don't enforce above the max steer
      // TODO: this should always be done
      lowest_desired_angle = SAFETY_CLAMP(lowest_desired_angle, -limits.max_angle, limits.max_angle);
      highest_desired_angle = SAFETY_CLAMP(highest_desired_angle, -limits.max_angle, limits.max_angle);
    }

    // check for violation;
    violation |= safety_max_limit_check(desired_angle, highest_desired_angle, lowest_desired_angle);
  }
  desired_angle_last = desired_angle;

  // Angle should be close to current angle while not steering
  if (!steer_control_enabled) {
    violation |= steer_angle_cmd_inactive_check(desired_angle, limits.max_angle);
  }

  // No angle control allowed when controls are not allowed
  if (!lateral_controls_allowed()) {
    violation |= steer_control_enabled;
  }

  // reset to current angle if either controls is not allowed or there's a violation
  if (violation || !lateral_controls_allowed()) {
    desired_angle_last = SAFETY_CLAMP(angle_meas.values[0], -limits.max_angle, limits.max_angle);
  }

  return violation;
}

static bool volvo_c1_tx_hook(const CANPacket_t *msg) {
  static const AngleSteeringLimits VOLVO_C1_ANGLE_STEERING_LIMITS = {
    .max_angle = VOLVO_C1_MAX_ANGLE_CAN,
    .angle_deg_to_can = VOLVO_C1_ANGLE_DEG_TO_CAN,
    .angle_rate_up_lookup = {
      {7.0f, 17.0f, 36.0f},
      {2.0f, 0.25f, 0.1f},
    },
    .angle_rate_down_lookup = {
      {7.0f, 17.0f, 36.0f},
      {2.0f, 0.25f, 0.1f},
    },
    .frequency = 50U,
  };
  bool tx = true;
  if (msg->addr == VOLVO_C1_FSM_1) {
    const int desired_angle = volvo_c1_fsm_angle(msg);
    const uint8_t direction = msg->data[7] & 0x3U;
    const bool steer_control_enabled = direction != 0U;
    tx &= SAFETY_ABS(desired_angle) <= VOLVO_C1_MAX_ANGLE_CAN;
    tx &= !volvo_c1_angle_cmd_checks(desired_angle, steer_control_enabled, VOLVO_C1_ANGLE_STEERING_LIMITS);
    tx &= (direction == 0U) || (direction == 3U);
    tx &= (msg->data[0] == 0xE3U) && (msg->data[1] == 0xB4U) && (msg->data[2] == 0x08U);
    tx &= (msg->data[3] == 0x80U) && ((msg->data[4] & 0xC0U) == 0x80U) && ((msg->data[7] & 0xFCU) == 0x94U);
    tx &= msg->data[6] == volvo_c1_fsm_checksum(msg);
  }

  if (msg->addr == VOLVO_C1_PSCM_1) {
    const int relayed_angle = volvo_c1_pscm_angle(msg);
    const uint32_t torque_raw = (((uint32_t)msg->data[1] & 0x0FU) << 8U) | (uint32_t)msg->data[2];
    tx &= torque_raw == 2000U;
    tx &= (msg->data[1] & 0x20U) == 0U;
    const int measured_max = angle_meas.max + VOLVO_C1_RELAY_ANGLE_TOLERANCE;
    const int measured_min = angle_meas.min - VOLVO_C1_RELAY_ANGLE_TOLERANCE;
    tx &= !safety_max_limit_check(relayed_angle, measured_max, measured_min);
  }

  // Only ACC cancel (byte 7 = 0x10) and, at a held standstill, Resume (byte 7 = 0x20) may be synthesized.
  // Other bytes: all zero, or, for Resume only, the stock frame's own bytes 1-6 (what a real press carries).
  if (msg->addr == VOLVO_C1_BUTTONS) {
    bool zero_shape = msg->data[6] == 0U;
    for (unsigned int i = 0U; i < 6U; i++) {
      zero_shape = zero_shape && (msg->data[i] == 0U);
    }
    const bool cancel_or_none = (msg->data[7] & 0xEFU) == 0U;
    bool resume_ok = false;
    if (msg->data[7] == 0x20U) {
      resume_ok = (zero_shape || volvo_c1_buttons_match_stock(msg)) && volvo_c1_resume_allowed();
    }
    tx &= (zero_shape && cancel_or_none) || resume_ok;
  }

  if (msg->addr == VOLVO_C1_ACC_REQUEST) {
    tx &= volvo_c1_long_allowed() && volvo_c1_acc_request_tx_ok(msg);
  }
  if (msg->addr == VOLVO_C1_ACC_MODE) {
    tx &= volvo_c1_long_allowed() && volvo_c1_acc_mode_tx_ok(msg);
  }
  return tx;
}

static bool volvo_c1_fwd_hook(int bus_num, int addr) {
  const bool acc_msg = ((unsigned int)addr == VOLVO_C1_ACC_REQUEST) || ((unsigned int)addr == VOLVO_C1_ACC_MODE);
  return ((unsigned int)bus_num == VOLVO_PARTY_BUS) && acc_msg && volvo_c1_long_allowed();
}
static safety_config volvo_c1_init(uint16_t param) {
  volvo_c1_long = GET_FLAG(param, VOLVO_C1_PARAM_LONG);
  volvo_c1_cam_frames_clear(&volvo_c1_cam_request);
  volvo_c1_cam_frames_clear(&volvo_c1_cam_mode);
  volvo_c1_pressure_bits = false;
  volvo_c1_brake_status = 0U;
  volvo_c1_brake_hold = false;
  volvo_c1_acc_active = false;
  volvo_c1_resume_budget = VOLVO_C1_RESUME_MAX_FRAMES;
  volvo_c1_buttons_stock_valid = false;
  volvo_c1_aol_reset();
  volvo_c1_aol_enabled = (unsigned int)alternative_experience == 32U;
  if (volvo_c1_aol_enabled) {
    static const AolSafetyPolicy policy = {
      .reset = volvo_c1_aol_reset,
      .request_mask = volvo_c1_aol_request,
      .permission_mask = volvo_c1_aol_permission,
      .rx_invalid = volvo_c1_aol_reset,
    };
    aol_policy = &policy;
  }
  static const CanMsg VOLVO_C1_TX_MSGS[] = {
    {VOLVO_C1_FSM_1, VOLVO_MAIN_BUS, 8, .check_relay = true},
    {VOLVO_C1_PSCM_1, VOLVO_PARTY_BUS, 8, .check_relay = true},
    {VOLVO_C1_BUTTONS, VOLVO_MAIN_BUS, 8, .check_relay = false},
  };
  // check_relay off: the camera's own frames reach the car whenever OP may not send
  static const CanMsg VOLVO_C1_LONG_TX_MSGS[] = {
    {VOLVO_C1_FSM_1, VOLVO_MAIN_BUS, 8, .check_relay = true},
    {VOLVO_C1_PSCM_1, VOLVO_PARTY_BUS, 8, .check_relay = true},
    {VOLVO_C1_BUTTONS, VOLVO_MAIN_BUS, 8, .check_relay = false},
    {VOLVO_C1_ACC_REQUEST, VOLVO_MAIN_BUS, 8, .check_relay = false},
    {VOLVO_C1_ACC_MODE, VOLVO_MAIN_BUS, 8, .check_relay = false},
  };
  static RxCheck volvo_c1_rx_checks[] = {
    {.msg = {{VOLVO_C1_PSCM_1, VOLVO_MAIN_BUS, 8, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{VOLVO_C1_FSM_0, VOLVO_PARTY_BUS, 8, 100U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{VOLVO_C1_PEDAL_AND_BRAKE, VOLVO_MAIN_BUS, 8, 100U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{VOLVO_C1_SPEED, VOLVO_MAIN_BUS, 8, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{VOLVO_C1_BRAKE_MESSAGES, VOLVO_MAIN_BUS, 8, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{VOLVO_C1_CEM_0, VOLVO_MAIN_BUS, 8, 40U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
  };
  safety_config ret = BUILD_SAFETY_CFG(volvo_c1_rx_checks, VOLVO_C1_TX_MSGS);
  if (volvo_c1_long) {
    ret = BUILD_SAFETY_CFG(volvo_c1_rx_checks, VOLVO_C1_LONG_TX_MSGS);
  }
  return ret;
}
const safety_hooks volvo_c1_hooks = {.init = volvo_c1_init, .rx = volvo_c1_rx_hook, .tx = volvo_c1_tx_hook};
