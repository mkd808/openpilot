# Volvo C1 (V40) log analysis tools

Small scripts for the CAN CSV exports (`time,addr,bus,data`). They need only numpy and pandas and run from this folder.

- `canlog.py`: loader and helpers (byte access, resampling, measured acceleration from wheel speed, ACC-active mask).
- `scan_accel.py LOG.csv`: ranks every byte on buses 0 and 2 by correlation with measured acceleration while stock
  ACC is active and the gas pedal is off, with the best lag. Camera-side request bytes peak at a longer lag than ECM and
  TCM responses.
- `decode_acc_msgs.py LOG.csv [--window START END]`: byte-by-byte summary of the camera ACC messages 0x270 (FSM3) and
  0x280 (FSM4), plus every change in a time window, to line up with an event.

- `state_table.py LOG.csv`: the values stock ACC sends in 0x270 and 0x280 in each brake and ACC state (idle, ACC braking,
  hold, standby, off). This is the reference table for keeping the camera's values consistent with each other.

Signal map (route 19, compared against the V60 port in leomonde/StarPilot; confidence is moderate, from logs only):
- 0x280 byte 6, low 2 bits: mode. 0 = braking (mean accel -0.38 m/s^2), 1 = coasting (-0.16), 2 = traction (+0.24).
- 0x280 byte 7: traction request. About 31 in traction, high (234, so -22 signed) in coasting for engine drag, 0 in braking.
- 0x280 byte 5: behaves like the brake target (lower = more braking): about 35 driving, about 0 braking or holding,
  252 with ACC off.
- 0x270 byte 2: decel request, 0xFF = none, about 0.04 m/s^2 per count.
- Not found on the V40: 0x220, 0x299, 0x31A. Candidates for the ECM confirmations the camera expects are bus 0 bytes
  that change with ACC braking: 0x1E0 b1, 0x2C3 b1, 0x1B0, 0x2B5 b2, 0x80 b6/b7, 0x2C5 b7, 0x130. 0x355 b2 is a camera
  message on bus 2 that also changes when ACC brakes.

Earlier findings (route 19): 0x270 byte 2 is the best candidate for the decel request (0xFF = none, about 0.04 m/s^2
per count), byte 3 looks like a traction request, and neither 0x270 nor 0x280 shows a counter or checksum. These are
observations from logs only; nothing has been sent to the car.

Run the tests with `python3 -m pytest --noconftest -o addopts="" -p no:cacheprovider tools/volvo_c1/tests -q`.
