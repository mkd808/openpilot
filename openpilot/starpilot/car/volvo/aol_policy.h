#pragma once
#include "openpilot/selfdrive/pandad/aol_protocol.h"
inline bool volvo_aol_param(uint16_t param) {
  return (param == 8U) || (param == 2U);  // EUCD (V60), C1 (V40)
}
inline constexpr AolSafetyProfile VOLVO_AOL_PROFILE{36U, volvo_aol_param, true};
