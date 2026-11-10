# V40 (C1) longitudinal: passive relay and observation plan

Goal: before any brake or accel frame is sent, prove two things from logs alone:
1. The panda relays the camera/car traffic without changing it (nothing the camera or ECM sees differs from stock).
2. The brake-feedback candidates (see README) behave the same on every drive, including stationary holds.

Nothing here sends a new frame. Every stage is passive. Stages 3+ are listed only so the gate is explicit.

## Stage 0 - setup and abort rules
- Branch with the current Resume code only. Longitudinal control stays off (`openpilotLongitudinalControl` false, no accel or brake frames in the TX allowlist).
- Resume flag file may stay on or off; note which in the log name.
- Log name carries the stage and flag state, e.g. `s1-parked-flagoff`.
- Abort and revert to stock if the dash shows any camera, brake or ACC warning, or ACC drops without a button or pedal cause.

## Stage 1 - relay integrity (parked, engine on, in P, parking brake on, ~3 min)
Do nothing. Log CAN on buses 0 and 2 (and the 128/130 mirrors if enabled).
Check offline, per address that appears on both buses:
- Payload identical on bus 0 and bus 2, except the messages C1 intentionally replaces (FSM_1, PSCM_1).
- Frame count within 1 of each other over the run (no drops).
- Median timestamp offset under 2 ms and no inter-frame gap above 2x the nominal period.
- 0x160 five-frame cycle intact (same sequence of byte-7/byte-2 patterns as stock logs).
Pass: all of the above on every address. Fail: any payload mismatch outside the two replaced messages.

## Stage 2 - parked driver inputs (~10 min total)
- 10 brake-pedal presses, 3 s each, 3 s apart.
- 3 presses each of ACC on/off, SET, RESUME, CANCEL (steering-wheel buttons).
- P, R, N, D shifts with the foot on the brake.
Check: 0x1E0 b1, 0x2C3 b1 and 0x1B0 b2 do not move with the pedal; button codes match README. Already true on the 17:29 parked route, so this stage mostly confirms it again after any firmware change.

## Stage 3 - on-road observation (no new frames)
Needs a real stop-and-go drive. Target: 20-30 min, at least 10 stops of 5 s or more under stock ACC hold, a mix of stops behind a lead and at lights. Do not touch brake or gas during a hold.
Log with the shadow logger enabled so each hold start and Resume is tagged.
For each stop record: 0x1B0 b2 sequence (A5 -> E0/E1 -> release), and 0x2B5 b2, 0x1E0 b1, 0x2C3 b1, 0x130 b6 at the hold start, during the hold, and at release.
Pass: the hold-state values are the same on every stop. Any stop that differs is a finding before going further.

## Stage 4 - calibration drive (still no new frames)
Steady ACC following of a lead at three or four different closing rates, plus one gentle ACC stop. Goal: the scale of the 0x270 bytes 2-3 decel value, 0x280 traction/target bytes, and 0x1B0 pressure against measured acceleration. Needs the lead-car speed from radar for ground truth.

## Gate before any sent frame
Only after stages 1-4 pass: write the panda safety design first (decel limit, rate limit, cutout on driver brake, allowlisted addresses, which confirmation bits are passed or masked), with unit tests, review it, then a closed-road low-speed test with a driver ready to brake. Not part of this plan.

## Log checklist per run
Route id, branch and commit, flag state, stage, weather/road, any dash warnings, and the clock times of deliberate actions (pedal presses, button presses).
