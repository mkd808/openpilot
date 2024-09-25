#pragma once

#include <set>

#include "selfdrive/frogpilot/ui/qt/offroad/frogpilot_settings.h"

class FrogPilotLateralPanel : public FrogPilotListWidget {
  Q_OBJECT

public:
  explicit FrogPilotLateralPanel(FrogPilotSettingsWindow *parent);

signals:
  void openParentToggle();

private:
  void hideToggles();
  void showToggles(std::set<QString> &keys);
  void updateCarToggles();
  void updateMetric();
  void updateState(const UIState &s);

  std::set<QString> aolKeys = {"AlwaysOnLateralLKAS", "AlwaysOnLateralMain", "HideAOLStatusBar", "PauseAOLOnBrake"};
  std::set<QString> laneChangeKeys = {"LaneChangeTime", "LaneDetectionWidth", "MinimumLaneChangeSpeed", "NudgelessLaneChange", "OneLaneChange"};
  std::set<QString> lateralTuneKeys = {"ForceAutoTune", "ForceAutoTuneOff", "NNFF", "NNFFLite", "TacoTune", "TurnDesires"};
  std::set<QString> qolKeys = {"PauseLateralSpeed"};

  std::map<std::string, AbstractControl*> toggles;

  Params params;

  bool hasAutoTune;
  bool hasNNFFLog;
  bool isMetric = params.getBool("IsMetric");
  bool started;
};
