#!/usr/bin/env python3
"""Byte-level summary of the camera ACC messages 0x270 (FSM3) and 0x280 (FSM4) on bus 2, plus a timeline.

usage: decode_acc_msgs.py LOG.csv                      # per-byte summary of both messages
       decode_acc_msgs.py LOG.csv --window 655 662     # also print every change in that window
"""
import argparse
import collections

import numpy as np

from canlog import byte, load_csv

MESSAGES = {0x270: "FSM3", 0x280: "FSM4", 0x30: "FSM0"}
TIMELINE_BYTES = {0x30: (7,)}  # FSM0 carries two interleaved variants; only byte 7 (ACC state) is tracked


def summarise(d, addr):
  s = d[(d.a == addr) & (d.bus == 2)]
  if s.empty:
    print(f"{hex(addr)} {MESSAGES[addr]}: no frames on bus 2")
    return
  dt = np.diff(np.sort(s.time.values))
  print(f"{hex(addr)} {MESSAGES[addr]}: {len(s)} frames, median period {np.median(dt) * 1000:.0f} ms")
  for i in range(8):
    b = byte(s.v.values, i)
    u = np.unique(b)
    vals = ", ".join(hex(int(x)) for x in u) if len(u) <= 8 else f"{hex(int(u.min()))}..{hex(int(u.max()))}"
    print(f"  byte {i}: {len(u):3d} values  {vals}")
  # a rolling counter would step by a constant amount nearly every frame
  for i in range(8):
    step = np.diff(byte(s.v.values, i)) % 256
    top, n = collections.Counter(step.tolist()).most_common(1)[0]
    if top != 0 and n / len(step) > 0.5:
      print(f"  byte {i} steps by {top} on {n / len(step):.0%} of frames (counter-like)")


def timeline(d, addr, lo, hi):
  s = d[(d.a == addr) & (d.bus == 2) & (d.time >= lo) & (d.time <= hi)].sort_values("time")
  prev = None
  for t, v in zip(s.time.values, s.v.values, strict=True):
    cols = TIMELINE_BYTES.get(addr, range(8))
    row = tuple(int(byte(np.array([v], dtype=np.uint64), i)[0]) for i in cols)
    if row != prev:
      print(f"{hex(addr)} t={t:8.3f} " + " ".join(f"b{i}={x:02x}" for i, x in zip(cols, row, strict=True)))
      prev = row


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("log")
  ap.add_argument("--window", type=float, nargs=2, metavar=("START", "END"))
  args = ap.parse_args()
  d = load_csv(args.log)
  for addr in (0x270, 0x280):
    summarise(d, addr)
  if args.window:
    for addr in (0x30, 0x270, 0x280):
      timeline(d, addr, *args.window)


if __name__ == "__main__":
  main()
