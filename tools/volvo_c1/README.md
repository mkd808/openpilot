# Volvo C1 (V40) log analysis tools

Small scripts for the CAN CSV exports (`time,addr,bus,data`). They need only numpy and pandas and run from this folder.

- `canlog.py`: loader and helpers (byte access, resampling, measured acceleration from wheel speed, ACC-active mask).
- `scan_accel.py LOG.csv`: ranks every byte on buses 0 and 2 by correlation with measured acceleration while stock
  ACC is active and the gas pedal is off, with the best lag. Camera-side request bytes peak at a longer lag than ECM and
  TCM responses.
- `decode_acc_msgs.py LOG.csv [--window START END]`: byte-by-byte summary of the camera ACC messages 0x270 (FSM3) and
  0x280 (FSM4), plus every change in a time window, to line up with an event.

Current findings (route 19): 0x270 byte 2 is the best candidate for the decel request (0xFF = none, about 0.04 m/s^2
per count), byte 3 looks like a traction request, and neither 0x270 nor 0x280 shows a counter or checksum. These are
observations from logs only; nothing has been sent to the car.

Run the tests with `python3 -m pytest --noconftest -o addopts="" -p no:cacheprovider tools/volvo_c1/tests -q`.
