"""Volvo V40 (C1) OP longitudinal: warn when OP needs braking that nobody is giving.

OP's own braking is off on the V40 (it faulted stock ACC), so OP can only coast to C1_LONG_ACCEL_MIN, and the stock
ACC camera ignores stationary cars at speed. Approaching a stopped car the camera keeps asking to accelerate, OP wants
to brake hard but only coasts, and openpilot, thinking it owns longitudinal, raised no alert (route
7c7d3d5e5ff31d51/00000016, 123-129 s: stopped car seen from 91 m, OP asking for down to -3.4, driver braked at 21 m).

The warning fires when, for HOLD_S:
- OP long is active and the car is moving,
- the camera isn't braking: the accel that went out is OP's own, at or above its coast floor (while the camera brakes,
  its own request goes out and it already handles the car ahead),
- OP asks for more than gentle coasting,
- and stopping behind a lead we're closing on needs more than coasting can give.

Replayed over routes 0f, 10, 12, 13 and 16 it fires once: route 16 at 126.0 s, 58 m from the stopped car and about
3.5 s before the driver braked.
"""

WANT_BELOW = -0.8  # m/s^2, OP's desired accel
WANT_MARGIN = 0.3  # m/s^2: OP wants this much more than the accel that went out (step 2b: OP may brake to ~-1.0 itself)
NEED_ABOVE = 1.0  # m/s^2, decel needed to stop behind the closing lead
HOLD_S = 0.3
MIN_SPEED = 3.0  # m/s
# m/s^2: below this the accel that went out is the camera's own braking. OP coasting only: -0.5. With step 2b's OP
# braking on (its flag file), OP's own request may go to ~-1.28, so -1.35. With OP braking off the -0.5 floor still
# tells the camera's early braking from OP's coasting (route 00000013 at 157.7 s).
OP_FLOOR = -0.5
OP_FLOOR_OP_BRAKING = -1.35
STANDOFF = 4.0  # m, stopping gap behind the lead


def stopping_decel(d_rel: float, v_rel: float) -> float:
  """Constant decel needed to match a closing lead's speed before STANDOFF; 0 when not closing."""
  if v_rel >= 0.:
    return 0.
  return v_rel ** 2 / (2. * max(d_rel - STANDOFF, 0.5))


class BrakeGapWarning:
  def __init__(self):
    self.since: float | None = None

  def update(self, t: float, long_active: bool, v_ego: float, desired_accel: float, output_accel: float,
             lead_present: bool, d_rel: float, v_rel: float, op_braking: bool = False) -> bool:
    floor = OP_FLOOR_OP_BRAKING if op_braking else OP_FLOOR
    short = (long_active and v_ego > MIN_SPEED and output_accel >= floor and desired_accel < WANT_BELOW and
             desired_accel < output_accel - WANT_MARGIN and lead_present and stopping_decel(d_rel, v_rel) > NEED_ABOVE)
    if not short:
      self.since = None
      return False
    if self.since is None:
      self.since = t
    return t - self.since >= HOLD_S
