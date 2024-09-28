#pragma once

#include <set>

#include "selfdrive/frogpilot/ui/qt/offroad/frogpilot_settings.h"

class FrogPilotSoundsPanel : public FrogPilotListWidget {
  Q_OBJECT

public:
  explicit FrogPilotSoundsPanel(FrogPilotSettingsWindow *parent);

signals:
  void openParentToggle();

private:
  FrogPilotSettingsWindow *parent;

  void hideToggles();
  void showToggles(const std::set<QString> &keys);
  void updateCarToggles();

  std::set<QString> alertVolumeControlKeys = {"DisengageVolume", "EngageVolume", "PromptDistractedVolume", "PromptVolume", "RefuseVolume", "WarningImmediateVolume", "WarningSoftVolume"};
  std::set<QString> customAlertsKeys = {"GreenLightAlert", "LeadDepartingAlert", "LoudBlindspotAlert"};

  std::map<QString, AbstractControl*> toggles;

  Params params;

  bool hasBSM;
};
