# StarPilot



[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/firestar5683/StarPilot)
[![Discord](https://img.shields.io/discord/1387432184121393333?label=Discord)](https://firestar.link/discord)
[![Last Updated](https://img.shields.io/github/last-commit/firestar5683/StarPilot/StarPilot)](https://github.com/firestar5683/StarPilot)
[![Wiki](https://img.shields.io/badge/Wiki-StarPilot-blue?logo=wiki)](https://wiki.firestar.link)

**StarPilot** is a custom fork of [comma.ai's openpilot](https://comma.ai/openpilot),
an open source driver assistance system.

Openpilot provides
* Automated Lane Centering
* Adaptive Cruise Control
* Lane Change Assist
* Driver Monitoring *without wheel nags*

# Volvo V40 (C1) openpilot: personal dev fork

A personal development branch based on StarPilotHQ 6.7.9. It's a work in progress and comes with **no warranty**. Use at your own risk.

Panda safety is treated as the priority: every change runs through openpilot's tools and test suites before it goes on the car.

## Credits

Big thanks to **Firestar4430** for starting the Volvo V40 II port on the StarPilotHQ stable branch, and to **Leomonde**, **Chiachunli** and **Danielzmod** for community support and the groundwork that goes back about six years to the comma two.

## Current state

- **Lateral only.** openpilot steers; longitudinal stays with the car's own ACC (`pcmCruise`).
- DBC signals are being tested and validated on real drives, including keeping openpilot engaged while factory ACC is braking or coasting.
- Radar tracks have been identified (read-only for now).

Everything below comes from our own drive logs (CAN CSVs and rlogs).

## ACC braking and hold states

0x1B0 `BrakeMessages`, byte 2:

| Byte 2 | Meaning | Source |
|---|---|---|
| `0xA0` | Idle, nothing applied | observed |
| `0xA1` | Transition, e.g. just before the car stops and around hold release | observed |
| `0xA5` | ACC braking by itself (`BrakeStatus` = 5); brake pressure rises with the pedal flag off | confirmed |
| `0xE0` / `0xE1` | ACC holding the car at standstill (`BrakeHold`, bit 6; `BrakeStatus` = 1) | confirmed |

`BrakeStatus` values other than 1 and 5 have not been seen.

Related signals:

- `BrakePress0` / `BrakePress1` rise during ACC braking even though the driver isn't on the pedal.
- FSM0 (0x30, byte 7, bus 2) reports ACC state: `138` / `154` standby, `142` / `143` / `158` / `159` active, `136` off. During a hold, ACC stays active, not in standby.

## Safer brake handling than the dragonpilot 0.8.13 port

dragonpilot 0.8.13 added a basic V40 port, but it hard-codes `brakePressed = False`, so a real foot on the brake is never seen. It only allows the cancel button on TX and has no stop-and-go or resume logic.

This branch:

- Derives `brakePressed` from the pressure bits, but ignores pressure while ACC owns the brake (`0xA5` / `0xE0` / `0xE1`). A driver pressing the pedal while ACC is braking still disengages.
- Mirrors that rule in the panda safety (`volvo.h`) so the firmware agrees with the car code.
- Doesn't cancel stock ACC when it's only braking or holding.
- Keeps cancel limited to the driver-intent case.

## Resume Assist (stop-and-go)

Stock ACC stops resuming by itself if the car has been held for more than a few seconds, and waits for the driver to press Resume. Resume Assist presses it for you when the lead pulls away. Stock ACC still does all the accelerating.

- **Off by default.** Turn it on under **Galaxy → Toggles → Vehicle → Volvo V40 Resume Assist** (Volvo only, parked only). Turning it off works any time.
- The toggle creates or removes `/data/volvo_c1_resume_assist`; you can also `touch` / `rm` it over SSH.
- Resume is sent only at standstill, with ACC active and the brake-hold flag set, once the lead has moved away and stock ACC hasn't resumed by itself.
- Panda allows Resume only under those same conditions, capped at 40 frames per hold and rearmed when the hold ends.

Ford-specific lateral-control and vehicle-support work includes substantial adaptations from
[BluePilot](https://github.com/BluePilotDev/bluepilot/tree/bp-7.0), principally developed by
[Alan Polk](https://github.com/alan-polk) with additional BluePilot contributors. StarPilot's
implementation has since diverged, but that does not erase its lineage. See [CREDITS.md](CREDITS.md)
for the code-level provenance and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for applicable
upstream notices and terms. BluePilot and its contributors do not maintain or endorse StarPilot;
please direct support requests for this adaptation to the StarPilot project.

Hyundai, Kia, and Genesis angle steering and related vehicle support include substantial adaptations
from [sunnypilot](https://github.com/sunnypilot/sunnypilot/tree/hkg-angle-steering-2025) and its
[opendbc angle-steering branch](https://github.com/sunnypilot/opendbc/tree/hkg-angle-steering-2025).
StarPilot's implementation has diverged significantly; the upstream contributors do not maintain
this adaptation. Detailed code lineage is recorded in [CREDITS.md](CREDITS.md#hyundai-kia-and-genesis-support-adapted-from-sunnypilot).

StarPilot has a vibrant, welcoming community [discord](https://firestar.link/discord).
Stop by to chat or ask questions!
If your car isn't driving as well as you'd like, [open a ticket so we can help](https://firestar.link/logs)!

## Documentation

Please see [https://wiki.firestar.link](https://wiki.firestar.link) for hardware lists,
installation guides, and software configuration.

## Features

* **Full support for Comma C3, C3X, and C4**
* Full support for chestnut/eGPU on C3, C3X, and C4. C3/C3X may require special gps handling.
* Model switcher with all of comma's tinygrad driving models
* StarPilot custom torque controller
* StarPilot custom longitudinal planner for both radar and VoACC (visual only, radar-less) vehicles
* **Custom tuning for an expanding list of vehicles**
* Firestar Lateral Method -
Auto tuning system to adapt openpilot to *your* car's unique lateral characteristics.
* Galaxy: StarPilot's portal to configure your comma device using your phone from anywhere.
Download models, change settings, update software, diagnostics, manage dashcam video, and more.
* Navigation - Display routes and give the driving model hints on taking exits and turns.
* Always On Lateral (full time steering assist)
* Lane Centering - Enable to center between detected lane lines,
with graceful fallback to model E2E lateral outputs and safety breakout
* Force Stop 2.0 - Better stops at stop signs and traffic lights.
* Speed Limit Controller - Use dashboard, map data, and/or vision detection for speed limit signs
* Curve Speed Controller - Learns from how you take curves
* Conditional Experimental Mode (CEM)
* Conditional Chill Mode (CCM)
* Driving Profiles and Traffic Mode
* Custom themes
* Alert volume customization
* Comma Pedal Interceptor support
* Bluetooth support - Connect keyboards, controllers, steering wheel remotes, and more.
* Sentry mode - Record when motion is detected while parked.
* V-ASM - Detect vehicles in the adjacent/blind spots using the driver-facing camera.
* Additional car support - Toyota SDSU, ZSS, CAN filter; pre-AP tesla; ford; Polestar 2; and more.
* And more!


## GM-only Features

* Increased LKAS fault resiliency
* ASCM_INT and SASCM support
* Custom lateral torque controller, with special tuning for Bolts
* 50% extra torque on 2017 Chevy Bolt
* Improved lateral and longitudinal tuning
* Dashboard cruise control display speed spoofing for vehicles with pedal interceptor
* Extra steering wheel button functionality for vehicles with pedal interceptor
* Optional toggle to boot comma when remote starting your vehicle

## Developer Features

* Native and cross compilation for Windows, Mac, and Ubuntu
* Custom AGNOS to support C3, C3X, and C4
* Raylib UI on C3, C3X, and C4
* To run UI on PC:
  * `./c3` for large UI
  * `./c4` for small UI
* `./build` to produce cross compiled binaries for comma devices.
Uses your comma's sysroot/toolchain
* Toggle: "Use Precompiled Binaries" to allow switching between fast boot / editable builds
* Custom long maneuver tests, specifically designed for regen-only vehicles

## Third-Party Notices

* Portions of this software include modified versions of the Material Design Icons provided by Google under the Apache License 2.0. A copy of the license is included in the `LICENSE-MDI` file.
* Ford support includes software adapted from BluePilot's `bp-7.0` branch. The source repository contains both a standard MIT notice and a separate custom SUNNYPILOT LLC notice; StarPilot preserves both in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
* Hyundai, Kia, and Genesis support includes software adapted directly from sunnypilot's HKG angle-steering branch and sunnypilot/opendbc. StarPilot preserves the applicable notices in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

> This project uses software from Haibin Wen and SUNNYPILOT LLC and is licensed under a custom license requiring permission for use.
