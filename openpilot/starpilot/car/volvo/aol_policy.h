#pragma once
#include "openpilot/selfdrive/pandad/aol_protocol.h"
inline bool volvo_aol_param(uint16_t param) {
  // EUCD (V60); C1 (V40), alone or with OP long (16) and its braking (32)
  return (param == 8U) || (param == 2U) || (param == 18U) || (param == 50U);
}
inline constexpr AolSafetyProfile VOLVO_AOL_PROFILE{36U, volvo_aol_param, true};
