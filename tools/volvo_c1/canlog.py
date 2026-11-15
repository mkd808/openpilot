"""Helpers for the Volvo C1 (V40) CAN CSV exports (columns: time, addr, bus, data).

Bus numbers in these exports: 0 car, 2 camera, 1 radar, 128/130 mirrored copies of 0/2.
Data bytes are big-endian in the hex string, so byte(v, 0) is the first byte on the wire.
"""
import numpy as np
import pandas as pd

ACC_ACTIVE_STATES = (142, 143, 158, 159)  # 0x30 byte 7 values with ACC active
SPEED_ADDR, SPEED_SCALE = 0x150, 0.01 / 3.6  # bytes 6-7, 0.01 km/h to m/s
FSM0_ADDR = 0x30
BRAKE_ADDR = 0x1B0
PEDAL_ADDR = 0x55


def load_csv(path: str) -> pd.DataFrame:
  """Returns columns time (s), bus, a (address), v (uint64 payload, first byte most significant)."""
  d = pd.read_csv(path, dtype=str)
  d["time"] = d.time.astype(float)
  d["bus"] = d.bus.astype(int)
  d["a"] = d.addr.map(lambda x: int(x, 16))
  parsed = {s: int(s[2:].ljust(16, "0"), 16) for s in d.data.unique()}
  d["v"] = d.data.map(parsed).astype(np.uint64)
  return d[["time", "bus", "a", "v"]]


def byte(v, i: int):
  """Byte i (0 = first on the wire) of payloads v, as ints."""
  return ((np.asarray(v, dtype=np.uint64) >> np.uint64(8 * (7 - i))) & np.uint64(0xFF)).astype(int)


def series(d: pd.DataFrame, addr: int, bus: int, fn) -> pd.Series:
  s = d[(d.a == addr) & (d.bus == bus)]
  return pd.Series(fn(s.v.values), index=s.time.values)


def on_grid(times, values, grid_t):
  """Last value at or before each grid time (NaN before the first sample). Handles duplicate timestamps."""
  times = np.asarray(times)
  values = np.asarray(values, dtype=float)
  order = np.argsort(times, kind="stable")
  t2, v2 = times[order], values[order]
  k = np.searchsorted(t2, grid_t, side="right") - 1
  return np.where(k >= 0, v2[np.clip(k, 0, None)], np.nan)


def measured_accel(d: pd.DataFrame, grid_t, smooth_s: float = 1.1):
  """Ego acceleration from wheel speed (m/s^2) on grid_t, plus the speed itself."""
  speed = series(d, SPEED_ADDR, 0, lambda v: (byte(v, 6) * 256 + byte(v, 7)) * SPEED_SCALE)
  sp = on_grid(speed.index.values, speed.values, grid_t)
  dt = float(np.median(np.diff(grid_t)))
  n = max(int(round(smooth_s / dt)), 1)
  a = np.gradient(np.convolve(sp, np.ones(n) / n, mode="same"), dt)
  a[:n] = np.nan  # the smoothing filter is not valid within one window of either end
  a[-n:] = np.nan
  return a, sp


def acc_active_mask(d: pd.DataFrame, grid_t):
  """Grid samples where stock ACC is active and the accelerator pedal is not pressed."""
  fsm = series(d, FSM0_ADDR, 2, lambda v: byte(v, 7))
  pedal = series(d, PEDAL_ADDR, 0, lambda v: ((byte(v, 1) & 3) << 8 | byte(v, 2)) * 0.1)
  state = on_grid(fsm.index.values, fsm.values, grid_t)
  gas = on_grid(pedal.index.values, pedal.values, grid_t)
  return np.isin(state, ACC_ACTIVE_STATES) & (gas < 1.0)
