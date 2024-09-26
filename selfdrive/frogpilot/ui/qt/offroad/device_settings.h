#pragma once

#include <set>

#include "selfdrive/frogpilot/ui/qt/offroad/frogpilot_settings.h"

class FrogPilotDevicePanel : public FrogPilotListWidget {
  Q_OBJECT

public:
  explicit FrogPilotDevicePanel(FrogPilotSettingsWindow *parent);

signals:
  void openParentToggle();

private:
  void hideToggles();
  void showToggles(const std::set<QString> &keys);

  std::set<QString> deviceManagementKeys = {"DeviceShutdown", "IncreaseThermalLimits", "LowVoltageShutdown", "NoLogging", "NoUploads", "OfflineMode"};

  std::map<QString, AbstractControl*> toggles;

  Params params;
};
