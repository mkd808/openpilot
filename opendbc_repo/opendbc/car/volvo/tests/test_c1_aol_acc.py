from opendbc.car.volvo.values import CAR, volvo_c1_aol_allowed, volvo_c1_aol_stock_acc_mode


def test_skips_cancel_only_for_v40_aol_when_op_not_engaged():
  f = CAR.VOLVO_V40
  assert volvo_c1_aol_stock_acc_mode(f, True, False, True)
  assert not volvo_c1_aol_stock_acc_mode(f, True, False, False)  # AOL off: stock cancel behaviour
  assert not volvo_c1_aol_stock_acc_mode(f, True, True, True)    # OP engaged: unchanged
  assert not volvo_c1_aol_stock_acc_mode(f, False, False, True)  # not pcmCruise


def test_skips_cancel_only_while_v40_acc_self_braking():
  from opendbc.car.volvo.values import volvo_c1_acc_self_braking_mode
  f = CAR.VOLVO_V40
  assert volvo_c1_acc_self_braking_mode(f, True, False, True)
  assert not volvo_c1_acc_self_braking_mode(f, True, False, False)  # driver brake / no ACC braking: cancel stays
  assert not volvo_c1_acc_self_braking_mode(f, True, True, True)    # OP engaged: unchanged
  assert not volvo_c1_acc_self_braking_mode(f, False, False, True)  # not pcmCruise


def test_v40_stock_acc_owns_cruise():
  from opendbc.car.volvo.values import volvo_c1_stock_acc_owns_cruise
  assert volvo_c1_stock_acc_owns_cruise(CAR.VOLVO_V40, True, False)
  assert not volvo_c1_stock_acc_owns_cruise(CAR.VOLVO_V40, True, True)
  assert not volvo_c1_stock_acc_owns_cruise(CAR.VOLVO_V40, False, False)
  assert not volvo_c1_stock_acc_owns_cruise(CAR.VOLVO_V60, True, False)


def test_aol_needs_lka_switch_and_stock_acc_on():
  assert volvo_c1_aol_allowed(True, True)    # LKA on, ACC standby or active
  assert not volvo_c1_aol_allowed(True, False)   # ignition on, ACC never switched on
  assert not volvo_c1_aol_allowed(False, True)   # LKA off
  assert not volvo_c1_aol_allowed(False, False)
