#pragma once

#include <set>

#include "selfdrive/frogpilot/ui/qt/offroad/frogpilot_settings.h"

class FrogPilotAdvancedPanel : public FrogPilotListWidget {
  Q_OBJECT

public:
  explicit FrogPilotAdvancedPanel(FrogPilotSettingsWindow *parent);

signals:
  void openParentToggle();
  void openSubParentToggle();
  void openSubSubParentToggle();

private:
  QString processModelName(const QString &modelName);

  void hideSubToggles();
  void hideSubSubToggles();
  void hideToggles();
  void showEvent(QShowEvent *event) override;
  void showToggles(std::set<QString> &keys);
  void startDownloadAllModels();
  void updateCalibrationDescription();
  void updateCarToggles();
  void updateMetric();
  void updateModelLabels();
  void updateState(const UIState &s);

  ButtonControl *deleteModelBtn;
  ButtonControl *downloadAllModelsBtn;
  ButtonControl *downloadModelBtn;
  ButtonControl *selectModelBtn;

  FrogPilotParamValueButtonControl *steerFrictionToggle;
  FrogPilotParamValueButtonControl *steerLatAccelToggle;
  FrogPilotParamValueButtonControl *steerKPToggle;
  FrogPilotParamValueButtonControl *steerRatioToggle;

  std::set<QString> aggressivePersonalityKeys = {"AggressiveFollow", "AggressiveJerkAcceleration", "AggressiveJerkDanger", "AggressiveJerkSpeed", "ResetAggressivePersonality"};
  std::set<QString> customDrivingPersonalityKeys = {"AggressivePersonalityProfile", "RelaxedPersonalityProfile", "StandardPersonalityProfile", "TrafficPersonalityProfile"};
  std::set<QString> developerUIKeys = {"BorderMetrics", "FPSCounter", "LateralMetrics", "LongitudinalMetrics", "NumericalTemp", "SidebarMetrics", "UseSI"};
  std::set<QString> lateralTuneKeys = {"SteerFriction", "SteerLatAccel", "SteerKP", "SteerRatio"};
  std::set<QString> longitudinalTuneKeys = {"LeadDetectionThreshold", "MaxDesiredAcceleration"};
  std::set<QString> modelManagementKeys = {"AutomaticallyUpdateModels", "DeleteModel", "DownloadModel", "DownloadAllModels", "ModelRandomizer", "ResetCalibrations", "SelectModel"};
  std::set<QString> modelRandomizerKeys = {"ManageBlacklistedModels", "ResetScores", "ReviewScores"};
  std::set<QString> modelUIKeys = {"DynamicPathWidth", "HideLeadMarker", "LaneLinesWidth", "PathEdgeWidth", "PathWidth", "RoadEdgesWidth", "UnlimitedLength"};
  std::set<QString> relaxedPersonalityKeys = {"RelaxedFollow", "RelaxedJerkAcceleration", "RelaxedJerkDanger", "RelaxedJerkSpeed", "ResetRelaxedPersonality"};
  std::set<QString> standardPersonalityKeys = {"StandardFollow", "StandardJerkAcceleration", "StandardJerkDanger", "StandardJerkSpeed", "ResetStandardPersonality"};
  std::set<QString> trafficPersonalityKeys = {"TrafficFollow", "TrafficJerkAcceleration", "TrafficJerkDanger", "TrafficJerkSpeed", "ResetTrafficPersonality"};

  std::map<std::string, AbstractControl*> toggles;

  QDir modelDir{"/data/models/"};

  QList<LabelControl*> labelControls;

  Params params;
  Params paramsMemory{"/dev/shm/params"};
  Params paramsStorage{"/persist/params"};

  bool allModelsDownloading;
  bool cancellingDownload;
  bool customPersonalityOpen;
  bool disableOpenpilotLongitudinal;
  bool hasAutoTune;
  bool hasOpenpilotLongitudinal;
  bool haveModelsDownloaded;
  bool isMetric = params.getBool("IsMetric");
  bool liveValid;
  bool modelDeleting;
  bool modelDownloading;
  bool modelManagementOpen;
  bool modelRandomizer;
  bool modelsDownloaded;
  bool started;

  float steerFrictionStock;
  float steerLatAccelStock;
  float steerKPStock;
  float steerRatioStock;

  QStringList availableModelNames;
  QStringList availableModels;
  QStringList experimentalModels;
};
