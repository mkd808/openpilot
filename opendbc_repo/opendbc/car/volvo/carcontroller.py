from collections import deque
import numpy as np
from opendbc.can import CANPacker
from opendbc.car import Bus
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.lateral import apply_std_steer_angle_limits
from opendbc.car.volvo.volvocan import create_c1_cancel, create_c1_pscm_message, create_c1_resume, create_c1_steering_control
from opendbc.car.volvo.values import CarControllerParams


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

    new_actuators = actuators.as_builder()
    new_actuators.steeringAngleDeg = self.apply_angle_last
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
