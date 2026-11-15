#!/usr/bin/env python3
"""Stock camera ACC values (0x270 FSM3, 0x280 FSM4) grouped by the car's brake and ACC state.

The camera sends several values that must agree with each other (decel, traction, brake flags and target). This prints
what stock ACC sends in each state, as the reference for any relay or replacement.

States come from 0x1B0 byte 2 (A0 idle, A5 ACC braking, A1 transition, E0/E1 hold) and 0x30 byte 7 (ACC state).

usage: state_table.py LOG.csv [--step 0.1]
"""
import argparse
import collections

import numpy as np

from canlog import ACC_ACTIVE_STATES, BRAKE_ADDR, byte, load_csv, on_grid, series

BRAKE_NAMES = {0xA0: "idle", 0xA1: "transition", 0xA5: "ACC braking", 0xE0: "hold", 0xE1: "hold"}


def pct(x):
  x = x[np.isfinite(x)]
  if len(x) == 0:
    return "-"
  p = np.percentile(x, [5, 50, 95]).astype(int)
  return f"{p[0]}/{p[1]}/{p[2]}"


def top(x, n=3):
  x = x[np.isfinite(x)].astype(int)
  if len(x) == 0:
    return "-"
  return ", ".join(f"{hex(k)}:{c / len(x):.0%}" for k, c in collections.Counter(x.tolist()).most_common(n))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("log")
  ap.add_argument("--step", type=float, default=0.1)
  args = ap.parse_args()
  d = load_csv(args.log)
  t = np.arange(0.5, d.time.max() - 0.5, args.step)

  def grid(addr, i):
    s = d[(d.a == addr) & (d.bus == 2)]
    return on_grid(s.time.values, byte(s.v.values, i), t)

  b1 = series(d, BRAKE_ADDR, 0, lambda v: byte(v, 2))
  brake = on_grid(b1.index.values, b1.values, t)
  f = series(d, 0x30, 2, lambda v: byte(v, 7))
  fsm = on_grid(f.index.values, f.values, t)
  acc = np.isin(fsm, ACC_ACTIVE_STATES)
  decel, trac = grid(0x270, 2), grid(0x270, 3)
  f4, f5, f6 = grid(0x270, 4), grid(0x270, 5), grid(0x270, 6)
  g0, g5, g6, g7 = grid(0x280, 0), grid(0x280, 5), grid(0x280, 6), grid(0x280, 7)

  print("percentiles are p5/p50/p95; flag bytes show the most common values")
  groups = [(f"ACC active, 0x1B0={BRAKE_NAMES.get(k, hex(k))}", acc & (brake == k)) for k in (0xA0, 0xA5, 0xA1)]
  groups.append(("ACC active, hold (0xE0/0xE1)", acc & np.isin(brake, [0xE0, 0xE1])))
  groups.append(("ACC standby (138/154)", np.isin(fsm, [138, 154])))
  groups.append(("ACC off (136)", fsm == 136))
  for name, m in groups:
    print(f"\n{name}: {int(m.sum())} samples")
    if m.sum() < 5:
      continue
    print(f"  0x270 b2 decel {pct(decel[m])}   b3 traction {pct(trac[m])}   b4 {top(f4[m], 2)}   b5 {top(f5[m], 2)}   b6 {top(f6[m], 2)}")
    print(f"  0x280 b0 {top(g0[m], 2)}   b5 {pct(g5[m])}   b6 {top(g6[m], 4)}   b7 {pct(g7[m])}")
    nr = m & np.isin(g6, [0xDE, 0x06])
    print(f"  0x280 b7 when b6 bit1 set: {pct(g7[nr])} ({int(nr.sum())} samples)")


if __name__ == "__main__":
  main()
