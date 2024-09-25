#pragma once

#include <set>

#include "selfdrive/frogpilot/ui/qt/offroad/frogpilot_settings.h"

class FrogPilotLongitudinalPanel : public FrogPilotListWidget {
  Q_OBJECT

public:
  explicit FrogPilotLongitudinalPanel(FrogPilotSettingsWindow *parent);

signals:
  void openParentToggle();
  void openSubParentToggle();

private:
  void hideSubToggles();
  void hideToggles();
  void showToggles(std::set<QString> &keys);
  void updateCarToggles();
  void updateMetric();

  FrogPilotButtonsControl *curveDetectionBtn;

  std::set<QString> conditionalExperimentalKeys = {"CESpeed", "CESpeedLead", "CECurves", "CELead", "CEModelStopTime", "CENavigation", "CESignal", "HideCEMStatusBar"};
  std::set<QString> curveSpeedKeys = {"CurveDetectionMethod", "CurveSensitivity", "DisableCurveSpeedSmoothing", "MTSCCurvatureCheck", "TurnAggressiveness"};
  std::set<QString> experimentalModeActivationKeys = {"ExperimentalModeViaDistance", "ExperimentalModeViaLKAS", "ExperimentalModeViaTap"};
  std::set<QString> longitudinalTuneKeys = {"AccelerationProfile", "DecelerationProfile", "HumanAcceleration", "HumanFollowing", "StoppingDistance"};
  std::set<QString> qolKeys = {"CustomCruise", "CustomCruiseLong", "ForceStandstill", "MapGears", "OnroadDistanceButton", "ReverseCruise", "SetSpeedOffset"};
  std::set<QString> speedLimitControllerKeys = {"SLCControls", "SLCQOL", "SLCVisuals"};
  std::set<QString> speedLimitControllerControlsKeys = {"Offset1", "Offset2", "Offset3", "Offset4", "SLCFallback", "SLCOverride", "SLCPriority"};
  std::set<QString> speedLimitControllerQOLKeys = {"ForceMPHDashboard", "SetSpeedLimit", "SLCConfirmation", "SLCLookaheadHigher", "SLCLookaheadLower"};
  std::set<QString> speedLimitControllerVisualsKeys = {"ShowSLCOffset", "SpeedLimitChangedAlert", "UseVienna"};

  std::map<std::string, AbstractControl*> toggles;

  Params params;

  bool hasPCMCruise;
  bool hasDashSpeedLimits;
  bool isGM;
  bool isHKGCanFd;
  bool isMetric = params.getBool("IsMetric");
  bool isSubaru;
  bool isToyota;
  bool slcOpen;
};
