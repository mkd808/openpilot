#pragma once

#include <set>

#include <QStringList>

#include "selfdrive/ui/qt/offroad/settings.h"
#include "selfdrive/ui/ui.h"

#include "selfdrive/frogpilot/ui/qt/offroad/frogpilot_settings.h"

class FrogPilotVehiclesPanel : public FrogPilotListWidget {
  Q_OBJECT

public:
  explicit FrogPilotVehiclesPanel(FrogPilotSettingsWindow *parent);

private:
  FrogPilotSettingsWindow *parent;

  void hideToggles();
  void setModels();
  void updateCarToggles();
  void updateState(const UIState &s);

  ButtonControl *selectMakeButton;
  ButtonControl *selectModelButton;

  ToggleControl *disableOpenpilotLong;

  QString carMake;
  QString carModel;

  QStringList models;

  QMap<QString, QString> carModels;

  std::set<QString> gmKeys = {"ExperimentalGMTune", "LongPitch", "NewLongAPIGM", "VoltSNG"};
  std::set<QString> hyundaiKeys = {"NewLongAPI"};
  std::set<QString> subaruKeys = {"CrosstrekTorque"};
  std::set<QString> toyotaKeys = {"ClusterOffset", "SNGHack", "ToyotaDoors"};

  std::map<QString, AbstractControl*> toggles;

  Params params;

  bool hasExperimentalOpenpilotLongitudinal;
  bool hasOpenpilotLongitudinal;
  bool hasSNG;
  bool isGMPCMCruise;
  bool isImpreza;
  bool isVolt;
  bool started;
};
