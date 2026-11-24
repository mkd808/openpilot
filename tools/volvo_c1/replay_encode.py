#!/usr/bin/env python3
"""Replay the camera's 0x160/0x280 frames from rlogs through the V40 long encoders.

  replay_encode.py ROUTE_DIR_OR_RLOG [...]

Checks, on bus 2:
- copy: every distinct stock frame re-packs byte-identical (no counter or checksum the encoder misses)
- masked: every masked 0x160 is the frame before it with C1_FSM2_MASK applied
- traction: 0x280 traction from the latest 0x160 request and the stock mode, against stock (ACC active)
- mode: c1_next_long_mode on the request, against the stock mode (ACC active)
"""
import copy
import glob
import os
import sys

import numpy as np

from opendbc.can import CANPacker, CANParser
from opendbc.car.volvo.eucd_params import LongMode
from opendbc.car.volvo.volvocan import C1_FSM2_MASK, c1_next_long_mode, create_c1_acc_mode, create_c1_acc_request
from openpilot.tools.lib.logreader import LogReader

DBC = "volvo_v40_2017_pt"
FSM0, FSM2, FSM4 = 0x30, 0x160, 0x280


def is_masked(dat: bytes) -> bool:
  return (dat[0] & 0x90) == 0


def traction_value(b: int, mode: LongMode) -> int:
  return b - 256 if mode == LongMode.COAST and b > 127 else b


def rlogs(paths):
  for path in paths:
    if os.path.isdir(path):
      yield from sorted(glob.glob(f"{path}/*rlog*"), key=lambda p: int(os.path.basename(p).split("--")[0]))
    else:
      yield path


def read_frames(path):
  for m in LogReader(path, sort_by_time=True):
    if m.which() == "can":
      for c in m.can:
        if c.src == 2 and c.address in (FSM0, FSM2, FSM4):
          yield c.address, bytes(c.dat)


class Decoder:
  def __init__(self, name, addr, freq):
    self.name, self.addr = name, addr
    self.cp = CANParser(DBC, [(name, freq)], 2)
    self.cache = {}

  def __call__(self, dat):
    if dat not in self.cache:
      self.cp.update([(0, [(self.addr, dat, 2)])])
      self.cache[dat] = copy.copy(self.cp.vl[self.name])
    return self.cache[dat]


def main(paths):
  packer = CANPacker(DBC)
  fsm2, fsm4 = Decoder("FSM2", FSM2, 50), Decoder("FSM4", FSM4, 40)
  copy_bad, mask_ok, mask_bad = set(), 0, 0
  trac_err, mode_ok, mode_n = [], 0, 0

  for path in rlogs(paths):
    acc, accel, last_fsm2, mode = False, None, None, None
    for addr, dat in read_frames(path):
      if addr == FSM0:
        acc = bool(dat[7] & 0x4)
        if not acc:
          mode = None
      elif addr == FSM2:
        v = fsm2(dat)
        if create_c1_acc_request(packer, v, v["ACC_AccelRequest"])[1] != dat:
          copy_bad.add((addr, dat))
        if is_masked(dat):
          if last_fsm2 is not None and not is_masked(last_fsm2):
            ok = bytes(a & b for a, b in zip(last_fsm2, C1_FSM2_MASK, strict=True)) == dat
            mask_ok += ok
            mask_bad += not ok
        else:
          accel = v["ACC_AccelRequest"]
        last_fsm2 = dat
      elif addr == FSM4:
        v = fsm4(dat)
        if create_c1_acc_mode(packer, v)[1] != dat:
          copy_bad.add((addr, dat))
        if acc and accel is not None:
          stock_mode = LongMode(int(v["ACC_Mode"]))
          ours = create_c1_acc_mode(packer, v, accel, stock_mode)[1]
          trac_err.append(traction_value(ours[7], stock_mode) - traction_value(dat[7], stock_mode))
          mode = stock_mode if mode is None else c1_next_long_mode(mode, accel)
          mode_ok += mode == stock_mode
          mode_n += 1
    print(f"{path}: done", file=sys.stderr)

  n2, n4 = len(fsm2.cache), len(fsm4.cache)
  print(f"copy: {n2} distinct 0x160, {n4} distinct 0x280, {len(copy_bad)} not byte-identical")
  for addr, dat in sorted(copy_bad)[:10]:
    print(f"  0x{addr:X} {dat.hex()}")
  print(f"masked: {mask_ok} match the previous frame & mask, {mask_bad} don't")
  if trac_err:
    e = np.abs(trac_err)
    print(f"traction: {len(e)} ACC frames, MAE {e.mean():.2f}, exact {np.mean(e == 0):.2f}, within 5 {np.mean(e <= 5):.2f}")
  if mode_n:
    print(f"mode: {mode_ok / mode_n:.4f} agree over {mode_n} ACC frames")
  return 1 if copy_bad or mask_bad else 0


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(2)
  sys.exit(main(sys.argv[1:]))
