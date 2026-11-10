import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from canlog import byte, load_csv, on_grid
from decode_acc_msgs import summarise
from scan_accel import scan


def _write(tmp_path, rows):
  p = tmp_path / "log.csv"
  p.write_text("time,addr,bus,data\n" + "\n".join(rows) + "\n")
  return str(p)


def test_load_csv_and_byte_order(tmp_path):
  d = load_csv(_write(tmp_path, ["0.000,0x270,2,0x0000FF0787000000", "0.027,0x270,2,0x0000FE0887000000"]))
  assert list(d.a) == [0x270, 0x270]
  assert list(byte(d.v.values, 2)) == [0xFF, 0xFE]
  assert list(byte(d.v.values, 3)) == [0x07, 0x08]


def test_short_data_is_padded(tmp_path):
  d = load_csv(_write(tmp_path, ["0.000,0x10,0,0x04"]))
  assert byte(d.v.values, 0)[0] == 0x04
  assert byte(d.v.values, 7)[0] == 0


def test_on_grid_handles_duplicate_times():
  v = on_grid([0.0, 0.1, 0.1, 0.2], [1, 2, 3, 4], np.array([-0.1, 0.05, 0.1, 0.25]))
  assert np.isnan(v[0])
  assert list(v[1:]) == [1, 3, 4]


def test_summarise_runs_on_small_log(tmp_path, capsys):
  d = load_csv(_write(tmp_path, [f"{i * 0.027:.3f},0x270,2,0x0000FF0787000000" for i in range(5)]))
  summarise(d, 0x270)
  out = capsys.readouterr().out
  assert "0x270 FSM3" in out and "byte 2" in out


def test_scan_accel_finds_a_planted_signal(tmp_path):
  rows = []
  for i in range(6000):  # 60 s at 100 Hz
    t = i * 0.01
    speed = 10 + 2 * np.sin(t / 5)
    sp = int(speed * 3.6 / 0.01)
    rows.append(f"{t:.3f},0x150,0,0x{0:012X}{sp:04X}")
    rows.append(f"{t:.3f},0x30,2,0x{0:014X}8F")  # ACC active
    rows.append(f"{t:.3f},0x55,0,0x0000000000000000")
    accel = 2 * np.cos(t / 5) / 5
    code = int(np.clip(128 + accel * 100, 0, 255))
    rows.append(f"{t:.3f},0x270,2,0x0000{code:02X}0000000000")
  d = load_csv(_write(tmp_path, rows))
  out, n = scan(d, [0, 2], min_frames=100)
  assert n > 200
  assert out[0][1] == "0x270" and out[0][3] == 2 and out[0][0] > 0.9
