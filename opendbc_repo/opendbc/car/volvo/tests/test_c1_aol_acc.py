from opendbc.car.volvo.values import CAR, volvo_c1_aol_stock_acc_mode


def test_skips_cancel_only_for_v40_aol_when_op_not_engaged():
  f = CAR.VOLVO_V40
  assert volvo_c1_aol_stock_acc_mode(f, True, False, True)
  assert not volvo_c1_aol_stock_acc_mode(f, True, False, False)  # AOL off: stock cancel behaviour
  assert not volvo_c1_aol_stock_acc_mode(f, True, True, True)    # OP engaged: unchanged
  assert not volvo_c1_aol_stock_acc_mode(f, False, False, True)  # not pcmCruise
