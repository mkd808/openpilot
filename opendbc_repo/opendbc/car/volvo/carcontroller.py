import os
from collections import deque
import numpy as np
from opendbc.can import CANPacker
from opendbc.car import DT_CTRL, Bus
from opendbc.car.carlog import carlog
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.lateral import apply_std_steer_angle_limits
from opendbc.car.volvo.eucd_params import LongMode
from opendbc.car.volvo.volvocan import (c1_next_long_mode, create_c1_acc_mode, create_c1_acc_request, create_c1_brake_relay, create_c1_cancel,
                                        create_c1_fsm0_relay,
                                        create_c1_pscm_message, create_c1_resume, create_c1_steering_control)
from opendbc.car.volvo.values import CarControllerParams, VolvoSafetyFlags


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP):
    super().__init__(dbc_names, CP)
    self.packer = CANPacker(dbc_names[Bus.pt])
    self.apply_angle_last = 0.0
    self.c1_torque_samples = deque(maxlen=CarControllerParams.C1_N_ZERO_TORQUE)
    self.c1_recovery_until = -1
    self.c1_resume_frames = 0  # V40 Resume burst, frames left
    self.c1_resume_done = False  # this hold already got its burst
    self.c1_resume_last_frame = None  # frame the last burst started
    self.c1_resume_bursts = 0  # bursts started so far; odd ones use the plain frame, even ones the stock-shaped frame
    self.c1_accel = 0.0  # accel OP sent last, or the camera's while it owns the frames
    self.c1_long_mode = LongMode.COAST
    self.c1_cam_accel = 0.0
    self.c1_cam_mode = LongMode.COAST
    self.c1_long_was_active = False
    self.c1_handoff = False  # OP is cancelling stock ACC to hand a stop it can't finish to the driver
    self.c1_handoff_frame = None  # frame the handoff's first cancel went out
    # brake test (test branch): panda allows it only with the C1_BRAKE_TEST safety flag
    self.c1_bt_capable = any(cfg.safetyParam & VolvoSafetyFlags.C1_BRAKE_TEST for cfg in CP.safetyConfigs)
    self.c1_bt_armed = False  # the flag file is on the device
    self.c1_bt_check_frame = None
    self.c1_bt_lka = None  # LKA switch state last frame; a change is the trigger
    self.c1_bt_active = False
    self.c1_bt_start = 0
    self.c1_bt_v0 = 0.
    self.c1_bt_relay_until = -1  # last frame to relay 0x1B0 to the camera
    self.c1_bt_release = False  # overshoot: flags dropped until the car is back near the request
    self.c1_bt_flags = False  # OP's hydraulic flags went out this frame

  def update(self, CC, CS, now_nanos):
    can_sends = []
    actuators = CC.actuators

    if self.frame % 2 == 0:  # stock FSM1 and PSCM1 messages are 50 Hz
      requested_active = CC.latActive and CS.out.vEgo > self.CP.minSteerSpeed
      recovering = requested_active and self.frame < self.c1_recovery_until

      if not requested_active:
        self.c1_torque_samples.clear()
        self.c1_recovery_until = -1
      elif recovering:
        self.c1_torque_samples.clear()
      else:
        if self.c1_recovery_until >= 0:
          self.c1_recovery_until = -1
          self.c1_torque_samples.clear()
        self.c1_torque_samples.append(CS.c1_lka_torque)
        if len(self.c1_torque_samples) == CarControllerParams.C1_N_ZERO_TORQUE and all(torque == 0 for torque in self.c1_torque_samples):
          self.c1_recovery_until = self.frame + 100
          self.c1_torque_samples.clear()
          recovering = True

      lat_active = requested_active and not recovering
      desired_angle = float(
        np.clip(
          actuators.steeringAngleDeg,
          CS.out.steeringAngleDeg - CarControllerParams.C1_ANGLE_ERROR,
          CS.out.steeringAngleDeg + CarControllerParams.C1_ANGLE_ERROR,
        )
      )
      apply_angle = apply_std_steer_angle_limits(
        desired_angle,
        self.apply_angle_last,
        CS.out.vEgoRaw,
        CS.out.steeringAngleDeg,
        lat_active,
        CarControllerParams.C1_ANGLE_LIMITS,
      )

      can_sends.append(create_c1_pscm_message(self.packer, CS.c1_msg_pscm))
      can_sends.append(create_c1_steering_control(self.packer, apply_angle, lat_active))
      self.apply_angle_last = apply_angle

    if CC.cruiseControl.cancel and self.frame % 10 == 0:
      can_sends.append(create_c1_cancel(self.packer))

    self._c1_resume_buttons(CC, CS, can_sends)
    if self.CP.openpilotLongitudinalControl:
      can_sends.extend(self._c1_longitudinal(CC, CS))

    new_actuators = actuators.as_builder()
    new_actuators.steeringAngleDeg = self.apply_angle_last
    if CC.longActive:
      new_actuators.accel = self.c1_accel
    self.frame += 1
    return new_actuators, can_sends

  def _c1_resume_buttons(self, CC, CS, can_sends):
    """Stop and go: one Resume burst per hold when controlsd asks for it (flag on, lead gone, stock ACC still holding).

    volvo_c1.h lets Resume through only at a held standstill with ACC active and caps the frames per hold."""
    at_hold = (CS.out.cruiseState.enabled and CS.out.standstill and CS.c1_brake_hold and
               not CS.out.gasPressed and not CS.out.brakePressed)
    if not at_hold:
      # A burst in progress stops when the hold ends. The next burst is allowed only once the rearm time has passed,
      # so a brief hold flicker right after a burst cannot start a second one.
      self.c1_resume_frames = 0
      if (self.c1_resume_last_frame is None or
          self.frame - self.c1_resume_last_frame >= CarControllerParams.C1_RESUME_REARM_FRAMES):
        self.c1_resume_done = False
      return

    if CC.cruiseControl.resume and not self.c1_resume_done:
      self.c1_resume_frames = CarControllerParams.C1_RESUME_HOLD_FRAMES
      self.c1_resume_done = True
      self.c1_resume_last_frame = self.frame
      self.c1_resume_bursts += 1

    if self.c1_resume_frames > 0:
      # Alternate the frame shape per burst to get an A/B record: the plain frame (bytes 0-6 zero) released the hold 2 of 3
      # times on older routes, the stock-shaped one 0 of 1 on route 25. volvo_c1.h accepts both.
      stock = CS.c1_buttons_stock if self.c1_resume_bursts % 2 == 0 else None
      can_sends.append(create_c1_resume(self.packer, stock))
      self.c1_resume_frames -= 1

  def _c1_brake_test(self, CC, CS, cam_brakes) -> bool:
    """Button-triggered brake test (see C1_BRAKE_TEST_FLAG): True while OP sends its fixed brake request."""
    CCP = CarControllerParams
    if not self.c1_bt_capable:
      return False
    if self.c1_bt_check_frame is None or self.frame - self.c1_bt_check_frame >= CCP.C1_BRAKE_TEST_FLAG_CHECK_FRAMES:
      self.c1_bt_armed = os.path.exists(CCP.C1_BRAKE_TEST_FLAG)
      self.c1_bt_check_frame = self.frame
    lka = bool(getattr(CS, "c1_lka_on", False))
    flipped = self.c1_bt_lka is not None and lka != self.c1_bt_lka
    self.c1_bt_lka = lka
    clear = (CC.longActive and not CS.out.gasPressed and not CS.out.brakePressed and not cam_brakes and
             not CC.hudControl.leadVisible and self.c1_bt_armed)
    if self.c1_bt_active:
      elapsed = (self.frame - self.c1_bt_start) * DT_CTRL
      reason = ("speed drop" if CS.out.vEgo <= self.c1_bt_v0 - CCP.C1_BRAKE_TEST_DROP else
                "time" if elapsed >= CCP.C1_BRAKE_TEST_MAX_S else None if clear else "aborted")
      if reason is not None:
        self.c1_bt_active = False
        carlog.warning(f"volvo_c1_brake_test end {reason} after {elapsed:.2f}s, {self.c1_bt_v0 * 3.6:.1f} -> {CS.out.vEgo * 3.6:.1f} km/h")
    elif flipped and clear and CS.out.vEgo >= CCP.C1_BRAKE_TEST_MIN_SPEED:
      self.c1_bt_active, self.c1_bt_start, self.c1_bt_v0 = True, self.frame, CS.out.vEgo
      self.c1_bt_release = False
      carlog.warning(f"volvo_c1_brake_test start at {CS.out.vEgo * 3.6:.1f} km/h")
    return self.c1_bt_active

  def _c1_longitudinal(self, CC, CS):
    """V40 OP long: drive and coast, stock ACC engaged and keeping all braking (see C1_LONG_ACCEL_MIN).

    One frame for each camera 0x160/0x280 frame, so the camera's cadence and masked copies carry through; volvo_c1.h
    only takes the camera's recent frames with the request, the hydraulic brake flags or the mode and traction replaced.
    OP never sends brake mode or the hydraulic flags. While the camera brakes (brake flags), asks
    for more decel than OP may, or the car is at a standstill (the camera's hold), its frames go out unchanged. The
    camera's brake mode without the flags is engine braking only: OP keeps the request then, and the
    camera's 0x280 (brake mode, no traction) goes out unchanged, since volvo_c1.h passes only that while the camera is in
    brake mode. A stop OP would have to finish while the camera asks to drive is handed to the driver by cancelling
    stock ACC (C1_LONG_HANDOFF_SPEED); until ACC is off OP's frames keep going out, also at a standstill."""
    CCP = CarControllerParams
    request_frames, mode_frames = CS.c1_stock_acc_request_frames, CS.c1_stock_acc_mode_frames
    if request_frames:
      self.c1_cam_accel = float(request_frames[-1]["ACC_AccelRequest"])
    if mode_frames:
      self.c1_cam_mode = LongMode(int(mode_frames[-1]["ACC_Mode"]))
    cam_hydraulic = any(f["ACC_BrakeRequest"] or f["ACC_BrakeRequest_B"] for f in request_frames)
    cam_brakes = cam_hydraulic or self.c1_cam_accel < CCP.C1_LONG_ACCEL_MIN

    # Stop handoff (see C1_LONG_HANDOFF_SPEED): latched until ACC is off
    if not CC.enabled:
      self.c1_handoff = False
    elif (CC.longActive and CS.out.vEgo < CCP.C1_LONG_HANDOFF_SPEED and CC.actuators.accel < CCP.C1_LONG_HANDOFF_ACCEL and
          not cam_brakes and self.c1_cam_accel > 0.):
      self.c1_handoff = True
    cam_owns = cam_brakes or (CS.out.standstill and not self.c1_handoff)
    brake_test = self._c1_brake_test(CC, CS, cam_brakes)

    if not CC.longActive:
      self.c1_long_was_active = False
      if not CC.enabled:
        return []  # the panda forwards the camera's frames itself
      # Engaged without long (an override): the panda may already block the camera's frames, so they go out unchanged.
      # It only takes them while it blocks, so the car never gets both.
      cam_owns = True
    else:
      if cam_owns or not self.c1_long_was_active:
        # hand back from (or start at) the camera's request, so OP's ramps from where the car is. The camera's brake
        # mode carries on as OP's only with its hydraulic flags; without them it was engine braking, so OP coasts.
        self.c1_accel = self.c1_cam_accel
        self.c1_long_mode = LongMode.COAST if self.c1_cam_mode == LongMode.BRAKE and not cam_hydraulic else self.c1_cam_mode
      self.c1_long_was_active = True

    if not cam_owns and brake_test:
      # fixed request, ramped in at the jerk limit, brake mode with the flags the way the camera brakes
      self.c1_accel = max(CCP.C1_BRAKE_TEST_ACCEL, self.c1_accel - CCP.C1_LONG_JERK_DOWN * DT_CTRL)
      # overshoot limiter: drop the flags while the car brakes much harder than asked
      if CS.out.aEgo < CCP.C1_BRAKE_TEST_ACCEL - CCP.C1_BRAKE_TEST_OVERSHOOT:
        if not self.c1_bt_release:
          carlog.warning(f"volvo_c1_brake_test overshoot release at {CS.out.aEgo:+.2f} m/s^2")
        self.c1_bt_release = True
      elif CS.out.aEgo > CCP.C1_BRAKE_TEST_ACCEL - CCP.C1_BRAKE_TEST_REAPPLY:
        self.c1_bt_release = False
      self.c1_long_mode = LongMode.BRAKE if self.c1_accel < 0. else LongMode.COAST
    elif not cam_owns:
      accel = min(max(CC.actuators.accel, CCP.C1_LONG_ACCEL_MIN), CCP.C1_LONG_ACCEL_MAX)
      if CCP.C1_LONG_CAMERA_CAP:
        accel = min(accel, max(self.c1_cam_accel, CCP.C1_LONG_ACCEL_MIN))
      # ramp from the last request, within OP's range: after the camera's braking that is OP's floor
      last = min(max(self.c1_accel, CCP.C1_LONG_ACCEL_MIN), CCP.C1_LONG_ACCEL_MAX)
      accel = min(max(accel, last - CCP.C1_LONG_JERK_DOWN * DT_CTRL), last + CCP.C1_LONG_JERK_UP * DT_CTRL)
      self.c1_accel = accel
      mode = c1_next_long_mode(self.c1_long_mode, accel)
      self.c1_long_mode = LongMode.COAST if mode == LongMode.BRAKE else mode

    sends = []
    for frame in request_frames:
      if cam_owns:
        sends.append(create_c1_acc_request(self.packer, frame, float(frame["ACC_AccelRequest"])))
      else:
        brake = brake_test and self.c1_long_mode == LongMode.BRAKE and self.c1_accel < 0. and not self.c1_bt_release
        sends.append(create_c1_acc_request(self.packer, frame, self.c1_accel, brake))
        if brake:
          # relay the car's 0x1B0 to the camera from OP's first braking frame on, as volvo_c1.h allows
          self.c1_bt_relay_until = self.frame + CCP.C1_BRAKE_TEST_RELAY_TAIL_FRAMES
    for frame in mode_frames:
      cam_frame = cam_owns or int(frame["ACC_Mode"]) == LongMode.BRAKE
      sends.append(create_c1_acc_mode(self.packer, frame) if cam_frame else create_c1_acc_mode(self.packer, frame, self.c1_accel, self.c1_long_mode))
    self.c1_bt_flags = brake_test and not cam_owns and self.c1_long_mode == LongMode.BRAKE and self.c1_accel < 0. and not self.c1_bt_release
    if self.frame <= self.c1_bt_relay_until:
      for frame in getattr(CS, "c1_brake_msg_frames", []):
        sends.append(create_c1_brake_relay(self.packer, frame))
      for frame in getattr(CS, "c1_fsm0_frames", []):
        sends.append(create_c1_fsm0_relay(self.packer, frame, self.c1_bt_flags))
    if self.c1_handoff and (self.c1_handoff_frame is None or (self.frame - self.c1_handoff_frame) % 10 == 0):
      if self.c1_handoff_frame is None:
        self.c1_handoff_frame = self.frame
      sends.append(create_c1_cancel(self.packer))
    if not self.c1_handoff:
      self.c1_handoff_frame = None
    return sends
