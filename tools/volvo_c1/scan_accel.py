#!/usr/bin/env python3
"""Rank every CAN byte by how well it tracks measured acceleration while stock ACC is active.

Request messages lead the car's response, so the best lag tells you which side of the loop a byte is on:
camera-side bytes (bus 2) peak at a longer lag than the ECM/TCM responses on bus 0.

usage: scan_accel.py LOG.csv [--top 25] [--buses 0 2]
"""
import argparse

import numpy as np

from canlog import acc_active_mask, byte, load_csv, measured_accel, on_grid

LAGS_S = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)


def scan(d, buses, step=0.1, min_samples=200, min_frames=500):
  grid = np.arange(20.0, d.time.max() - 1.0, step)
  accel, _ = measured_accel(d, grid)
  use = acc_active_mask(d, grid) & np.isfinite(accel)
  rows = []
  for (addr, bus), g in d[d.bus.isin(buses)].groupby(["a", "bus"]):
    if addr == 0x150 or len(g) < min_frames:
      continue
    for i in range(8):
      x = byte(g.v.values, i).astype(float)
      if x.std() == 0:
        continue
      xs = on_grid(g.time.values, x, grid)
      best = (0.0, 0.0)
      for lag in LAGS_S:
        y = np.roll(xs, int(round(lag / step)))
        m = use & np.isfinite(y)
        if m.sum() < min_samples or y[m].std() == 0:
          continue
        r = float(np.corrcoef(y[m], accel[m])[0, 1])
        if abs(r) > abs(best[0]):
          best = (r, lag)
      rows.append((abs(best[0]), hex(addr), int(bus), i, round(best[0], 3), best[1], len(g)))
  return sorted(rows, reverse=True), int(use.sum())


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("log")
  ap.add_argument("--top", type=int, default=25)
  ap.add_argument("--buses", type=int, nargs="+", default=[0, 2])
  args = ap.parse_args()
  rows, n = scan(load_csv(args.log), args.buses)
  print(f"ACC-active, gas-off samples: {n}")
  print("addr    bus byte  r      lag_s  frames")
  for _, addr, bus, i, r, lag, nframes in rows[:args.top]:
    print(f"{addr:7} {bus:3} {i:4} {r:6.3f} {lag:5.1f} {nframes}")


if __name__ == "__main__":
  main()
