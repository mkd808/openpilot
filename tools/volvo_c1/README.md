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

Signal map (route 19, checked against the V60 port in leomonde/StarPilot and the existing DBC; moderate confidence,
from stock-ACC logs only):
- 0x280 bytes 6-7, low 10 bits (`ACC_TractionRequest`): the top 2 bits act as a mode (0 = braking, 1 = coasting,
  2 = traction; mean measured accel -0.38, -0.16, +0.24 m/s^2) and the low 8 bits are the value (about 31 in
  traction, 234 = -22 signed in coasting for engine drag, 0 in braking).
- 0x280 byte 5 (`SpeedTarget`): behaves like a brake target, lower = more braking (about 35 driving, about 0 braking
  or holding, 252 with ACC off). Width and meaning unconfirmed.
- 0x270 bytes 2-3 (`ACC_LeadDecelRaw`): one signed 16-bit value, not separate fields. It correlates with measured
  acceleration at r = 0.67 (0.3 s lag); median -2539 while ACC brakes, -236 at idle.
- 0x160 `ACC_BrakeRequest`: set on 99% of ACC-braking samples, 0% at idle and in hold. `ACC_BrakeRequest_B` is the one
  set during a standstill hold.
- Not found on the V40: 0x220, 0x299, 0x31A. Candidates for the ECM confirmations the camera expects are bus 0 bytes
  that change with ACC braking: 0x1E0 b1, 0x2C3 b1, 0x1B0, 0x2B5 b2, 0x80 b6/b7, 0x2C5 b7, 0x130. 0x355 b2 is a camera
  message on bus 2 that also changes when ACC brakes.

No rolling counter or checksum was found on 0x270 or 0x280. These are observations from logs only; nothing has been
sent to the car.

Run the tests with `python3 -m pytest --noconftest -o addopts="" -p no:cacheprovider tools/volvo_c1/tests -q`.
