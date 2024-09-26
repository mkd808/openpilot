#pragma once

#include <set>

#include "selfdrive/frogpilot/ui/qt/offroad/frogpilot_settings.h"

class FrogPilotAdvancedVisualsPanel : public FrogPilotListWidget {
  Q_OBJECT

public:
  explicit FrogPilotAdvancedVisualsPanel(FrogPilotSettingsWindow *parent);

signals:
  void openParentToggle();

private:
  void hideToggles();
  void showEvent(QShowEvent *event) override;
  void showToggles(const std::set<QString> &keys);
  void updateCarToggles();
  void updateMetric();

  FrogPilotButtonToggleControl *borderMetricsBtn;

  std::set<QString> developerUIKeys = {"BorderMetrics", "FPSCounter", "LateralMetrics", "LongitudinalMetrics", "NumericalTemp", "SidebarMetrics", "UseSI"};
  std::set<QString> modelUIKeys = {"DynamicPathWidth", "HideLeadMarker", "LaneLinesWidth", "PathEdgeWidth", "PathWidth", "RoadEdgesWidth", "UnlimitedLength"};

  std::map<QString, AbstractControl*> toggles;

  Params params;

  bool disableOpenpilotLongitudinal;
  bool hasAutoTune;
  bool hasBSM;
  bool hasOpenpilotLongitudinal;
  bool isMetric = params.getBool("IsMetric");
};
