# Volvo C1 (V40) log analysis tools

Small scripts for the CAN CSV exports (`time,addr,bus,data`). They need only numpy and pandas and run from this folder.

- `canlog.py`: loader and helpers (byte access, resampling, measured acceleration from wheel speed, ACC-active mask).
- `scan_accel.py LOG.csv`: ranks every byte on buses 0 and 2 by correlation with measured acceleration while stock
  ACC is active and the gas pedal is off, with the best lag. Camera-side request bytes peak at a longer lag than ECM and
  TCM responses.
- `decode_acc_msgs.py LOG.csv [--window START END]`: byte-by-byte summary of the camera ACC messages 0x270 (FSM3) and
  0x280 (FSM4), plus every change in a time window, to line up with an event.

- `replay_encode.py ROUTE_DIR_OR_RLOG...`: replays the camera's 0x160/0x280 frames from rlogs through the V40 long
  encoders in `volvocan.py`. Every distinct frame must re-pack byte-identical and every masked 0x160 must be the frame
  before it with `C1_FSM2_MASK`; it also scores the traction and mode models against stock. Needs the openpilot env.

- `state_table.py LOG.csv`: the values stock ACC sends in 0x270 and 0x280 in each brake and ACC state (idle, ACC braking,
  hold, standby, off). This is the reference table for keeping the camera's values consistent with each other.

Signal map (22 Nov, 5 stock-ACC routes: 7c7d3d5e5ff31d51 19, 1a and 08, 18eb05bab882ec44 1a and 1b, about 1430 s of ACC;
checked against the V60 port in StarPilotHQ/StarPilot#332; from stock-ACC rlogs only). The camera sends only 0x30,
0xD0, 0x160, 0x270, 0x280 and 0x355 on bus 2.
- 0x160 byte 7 bits 6-0 (`ACC_AccelRequest`): the ACC acceleration request, `(raw - 64) * ~0.08` m/s^2. Leads measured
  acceleration by 0.2-0.4 s, holds fixed values with ACC off, and keeps falling through brake mode. The value alone
  sorts the 0x280 mode (below about -0.5 brake, -0.5 to 0 coast, 0 and up drive). Firm braking gives 0.073-0.077 per
  count (pooled 0.083), and 0.08 makes the flat-road traction gains match the V60; the hardest request seen is about
  -2 m/s^2. Byte 7 bit 7 is always set in normal frames.
- 0x160 byte 2 bit 5 (`ACC_BrakeRequest`, copied in byte 5 bit 5): the hydraulic brake request. In brake mode it agrees
  with 0x1B0 brake pressure rising in 84-100% of frames, and pressure only rises with it set. Never set outside brake
  mode or with ACC off. Brake mode without it is engine braking only (the two highway routes never set it).
  `ACC_BrakeRequest_B` is the one set during a standstill hold.
- 0x160 byte 3 bits 2-0 + byte 4 (`ACC_MeasuredAccel`): measured acceleration, `(raw - 1030) / ~225` m/s^2. Follows
  aEgo by 0.1-0.2 s and tracks it with ACC off too. Feedback, not a command.
- 0x160 every 5th frame is a masked copy (byte 0 0x90 -> 0x00, byte 7 bit 7 cleared); skip those when decoding.
- 0x280 byte 6 bits 1-0 (`ACC_Mode`): 0 = brake, 1 = coast, 2 = drive, as the V60.
- 0x280 byte 7 (`ACC_TractionRequest`): unsigned in drive, signed in coast, 0 in brake. Against the 0x160 request:
  drive `86 * a_req + 3` (r = 0.93), coast `54 * a_req - 10` (r = 0.73) pooled. Flat routes (grade within 2-3%):
  drive about `92 * a_req` (r 0.94-0.98), coast up to `93 * a_req`, close to the V60's 97/94. Hilly routes (5-6%) fit
  loosely, and a slope term adds nothing, so the camera does not compensate grade in traction. Against measured aEgo,
  coast matches the V60 `94 * a - 9`, and drive is about `97 * a + 35` (steady cruise needs 25-35 at any speed).
- 0x280 byte 5 (`SpeedTarget`): about 35 driving, about 0 braking or holding, 252 with ACC off. Meaning unconfirmed.
- 0x270 bytes 2-3 (`ACC_LeadDecelRaw`): NOT an ACC command. Not idle in 43-72% of ACC-off samples, ramps down about
  1 s before ACC brakes, and its peak does not match the braking achieved. Lead or warning related.
- Slope: requests track the car's net acceleration; the ECM and brakes compensate for grade themselves, as the
  V60 port assumes.
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

## ACC standby flag: 0x145 byte 0 (14 Nov, all routes with 0x1B0)

Volvo's Queue Assist text: a stop of about 3 s or less resumes by itself; a longer one puts ACC in standby with automatic braking until the driver presses the wheel button or the accelerator. The bus shows it as `0x145` (ECM3) byte 0:

- `0xA1` normally, `0xB9` from about 2.4-3.1 s into a standstill hold. It returns to `0xA1` when the driver restarts ACC: 0.08 s after the first Resume frame for human presses (151.5 s route 25, 411.3 s, 631.3 s) and for the two OP plain-frame bursts that worked (route 24 at 162.4 s, postpanda19 at 659.5 s).
- Seen on every hold of 5.6 s or longer on the V40 routes, never on holds of 2.3 s or less.
- Holds that ended with no button: `bella` 290.5 s and postpanda19 564.8 s were accelerator restarts. `bella` 907.5 s had neither pedal nor button (flag went 0xB9 -> 0xA9), unexplained.
- OP bursts that did nothing (postpanda19 560.1 s plain frame, route 25 148.3 s stock-shaped frame) both landed with the flag at `0xB9`, so ACC was in standby and still ignored them.
- Possible use, not implemented: send the burst only while the flag is `0xB9` (standby, where ACC will not resume by itself), and retry after a failed one.
