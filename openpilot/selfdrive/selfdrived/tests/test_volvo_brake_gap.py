import pytest

from openpilot.selfdrive.selfdrived.volvo_brake_gap import BrakeGapWarning, HOLD_S, stopping_decel

DT = 0.01
# route 00000016 at 126 s: 40 km/h, OP wants -0.94 but only its -0.48 coast goes out, stopped car 58 m ahead
RED_LIGHT = {'long_active': True, 'v_ego': 11.1, 'desired_accel': -0.94, 'output_accel': -0.48, 'lead_present': True, 'd_rel': 58., 'v_rel': -11.1}


def run(seconds: float, **frame) -> bool:
  warning = BrakeGapWarning()
  fired = False
  for i in range(round(seconds / DT)):
    fired = warning.update(i * DT, **frame)
  return fired


def test_stopping_decel():
  assert stopping_decel(58., -11.1) == pytest.approx(11.1 ** 2 / (2 * 54.))
  assert stopping_decel(20., 0.) == 0. and stopping_decel(20., 2.) == 0.
  assert stopping_decel(3., -2.) == pytest.approx(4.)  # inside the standoff: 0.5 m floor


def test_fires_approaching_stopped_car_after_hold():
  assert not run(HOLD_S - 0.05, **RED_LIGHT)
  assert run(HOLD_S + 0.05, **RED_LIGHT)


@pytest.mark.parametrize("change", [
  {'long_active': False},
  {'v_ego': 2.0},  # walking pace
  {'output_accel': -0.68, 'desired_accel': -1.46},  # OP braking off: the camera's early braking (route 13 at 157.7 s)
  {'output_accel': -1.5, 'op_braking': True},  # the camera is braking: its own request went out
  {'desired_accel': -1.2, 'output_accel': -1.25, 'op_braking': True},  # OP's own braking already gives what it wants
  {'desired_accel': -0.6},  # OP only wants gentle coasting
  {'lead_present': False},
  {'v_rel': 0.5},  # lead pulling away
  {'d_rel': 25., 'v_rel': -2.0},  # closing slowly: coasting is enough (route 16 at 304-309 s)
])
def test_quiet(change):
  assert not run(2.0, **{**RED_LIGHT, **change})


def test_fires_when_op_braking_is_not_enough():
  # step 2b: OP brakes at its ~-1.0 limit (request -1.25) but wants -2.0 with a stopped car close ahead
  assert run(HOLD_S + 0.05, **{**RED_LIGHT, 'desired_accel': -2.0, 'output_accel': -1.25, 'd_rel': 35., 'op_braking': True})


def test_restarts_hold_when_condition_breaks():
  warning = BrakeGapWarning()
  t = 0.
  for _ in range(20):
    assert not warning.update(t, **RED_LIGHT)
    t += DT
  assert not warning.update(t, **{**RED_LIGHT, 'output_accel': -1.5})
  for _ in range(round(HOLD_S / DT) - 1):
    t += DT
    assert not warning.update(t, **RED_LIGHT)
