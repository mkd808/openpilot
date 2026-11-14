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

## Brake-feedback candidates (V40 equivalent of the V60 ECM->FSM check)

The V60 port found the camera (FSM) checks ECM brake feedback against its own request. None of the V60 feedback addresses (0x323, 0x133, 0x220, 0x299, 0x12A, 0x325, 0x1FF, 0x127) exist on the V40. Candidates found by ranking bus-0 bytes against the 0x1B0 byte 2 == 0xA5 (ACC braking) label, then splitting into ACC braking / driver braking / coasting, confirmed on three routes. All also appear on bus 2. Observational only; nothing has been sent.

| Message / byte | ACC braking | Otherwise | Lag vs 0x1B0 A5 edge |
|---|---|---|---|
| 0x2B5 b2 | 0xF2 (1.00) | 0xB2, 0x9E parked; but 0xF2 also 13-37% of ACC-active non-braking samples (necessary, not sufficient) | leads 90-630 ms |
| 0x1E0 b1 | 0x16 (0.92-0.96) | 0x17 | lags 80-180 ms |
| 0x2C3 b1 | 0x5C (0.88-0.93) | 0x54 | lags 100-200 ms |
| 0xB0 b6 | 0xA3 (0.92-0.97) | 0x83, but 0xA3 also 0.35 of ACC-active non-braking | leads >1 s (state, not a confirmation) |
| 0x130 b6 | continuous | tracks decel, r=0.91 driver / 0.67 ACC | not an ACC flag |

Not ACC-specific: 0x80 b6. Next: check what the camera does when these disagree with its request (bench only), and whether 0x1B0 byte 2 itself is validated.

Parked pedal-press route: 0x1E0, 0x2C3 and 0x1B0 byte 2 do not respond to the driver pedal; 0x130 b6 does not either (stationary), so it follows vehicle deceleration, not pedal.
Fourth route (mixed cruise and driving, 20 s ACC braking): 0x1E0/0x2C3 stayed clean (0.00 false in ACC-active non-braking); 0x2B5 b2 did not.

## Resume frame: OP vs a human press (14 Nov stop, route 24)

A stop of 11 s behind a lead that then pulled away. OP sent 12 Resume frames (about 0.12 s) and stock ACC released the hold 0.14 s after the first one. ACC then sent a rising traction request for about 0.2 s (0x280 mode 2, value about 177-198), the car crept 0.66 km/h, and ACC withdrew the request and briefly re-applied the hold. ACC stayed active (FSM 143) and OP sent no cancel. The driver then pressed the gas. The shadow logger now reports this as `launch_failed`.

Burst length is not the explanation: human presses of 0.10 s and 0.20 s (braking and busacc routes) launched fine, and the hold released about 0.1-0.15 s after press start in both.

What differs from a human press (observation, not yet shown to matter):
- Human press: the stock stream itself changes, e.g. `F8B1D02F00008000` becomes `F8B1D02F00008020` (bytes 0-3 vary per drive, byte 6 is 0x80, byte 7 gets 0x20) and stays one frame stream at 100 Hz.
- OP press: `0000000000000020` (bytes 0-6 zero) appears as an extra frame at about 100 Hz on the car side while the stock stream keeps sending its idle frame `F8B1A06F00008000` at the same address. The car sees idle and Resume frames alternating.
- The panda requires byte 6 of 0x10 to be 0 (`volvo.h`), so a stock-identical frame would be rejected today.

If a later test is wanted, the cheap experiment is a Resume frame with bytes 0-5 and 6 copied from the latest stock frame, which needs a panda safety change and tests. Not done.

## Resume at a hold: every button press and OP burst on record (14 Nov, route 25 plus earlier routes)

`0x1B0` byte 2 at a standstill hold is `0xE1` (BrakeStatus 1) or `0xE0` (status 0); `0xA0`/`0xA1` mean the hold has released. Every press or burst that carried a Resume frame (byte 7 = 0x20), with the hold released or not:

| Source | Hold state | Result |
|---|---|---|
| Human (shadow2, 411 s) | E1 | released 0.12 s after the first frame, car launched ~1 s later |
| Human (braking, 631 s) | E0 | released 0.10 s, launched |
| Human (route 25, 151 s) | E0 | released 0.10 s, launched |
| OP plain frame `0000000000000020` (postpanda19, 659 s) | E1 | released 0.11 s, launched |
| OP plain frame (route 24, 162 s) | E1 | released 0.11 s, brief re-hold at +0.57 s, no launch |
| OP plain frame (postpanda19, 560 s) | E1 | no effect |
| OP stock-shaped frame `F8B1A06F00008020` (route 25, 148 s, 29 frames) | E1 | no effect |

- A button works from E1 or E0, so the hold state is not what decides it (a gate on status 0 was tried in `3885170d` and taken out again).
- The plain frame released the hold 2 of 3 times, the stock-shaped frame 0 of 1. Small numbers, but the opposite of the expectation that a stock-identical frame would work better.
- The panda accepts both shapes. The car side now alternates per burst (1st, 3rd... plain; 2nd, 4th... stock-shaped), so the CSV from any drive shows which shape was sent (bus 128) and whether `0x1B0` went `E1/E0` -> `A0/A1` about 0.1 s later.
