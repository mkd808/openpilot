#include "selfdrive/frogpilot/ui/qt/offroad/advanced_visual_settings.h"

FrogPilotAdvancedVisualsPanel::FrogPilotAdvancedVisualsPanel(FrogPilotSettingsWindow *parent) : FrogPilotListWidget(parent) {
  const std::vector<std::tuple<QString, QString, QString, QString>> advancedToggles {
    {"DeveloperUI", tr("Developer UI"), tr("Show detailed information about openpilot's internal operations."), "../frogpilot/assets/toggle_icons/icon_device.png"},
    {"BorderMetrics", tr("Border Metrics"), tr("Display performance metrics around the edge of the screen while driving."), ""},
    {"FPSCounter", tr("FPS Counter"), tr("Show the 'Frames Per Second' (FPS) at the bottom of the screen while driving."), ""},
    {"LateralMetrics", tr("Lateral Metrics"), tr("Display metrics related to steering control at the top of the screen while driving."), ""},
    {"LongitudinalMetrics", tr("Longitudinal Metrics"), tr("Display metrics related to acceleration, speed, and desired following distance at the top of the screen while driving."), ""},
    {"NumericalTemp", tr("Numerical Temperature Gauge"), tr("Show exact temperature readings instead of general status labels like 'GOOD', 'OK', or 'HIGH' in the sidebar."), ""},
    {"SidebarMetrics", tr("Sidebar"), tr("Display system information like CPU, GPU, RAM usage, IP address, and storage space in the sidebar."), ""},
    {"UseSI", tr("Use International System of Units"), tr("Display measurements using the 'International System of Units' (SI)."), ""},

    {"ModelUI", tr("Model UI"), tr("Customize the model visualizations on the screen."), "../assets/offroad/icon_calibration.png"},
    {"DynamicPathWidth", tr("Dynamic Path Width"), tr("Automatically adjust the width of the driving path display based on openpilot's engagement state."), ""},
    {"HideLeadMarker", tr("Hide Lead Marker"), tr("Do not display the marker for the vehicle ahead on the screen."), ""},
    {"LaneLinesWidth", tr("Lane Lines Width"), tr("Adjust how thick the lane lines appear on the display.\n\nDefault matches the MUTCD standard of 4 inches."), ""},
    {"PathEdgeWidth", tr("Path Edges Width"), tr("Adjust the width of the edges of the driving path to represent different modes and statuses.\n\nDefault is 20% of the total path width.\n\nColor Guide:\n- Blue: Navigation\n- Light Blue: 'Always On Lateral'\n- Green: Default\n- Orange: 'Experimental Mode'\n- Red: 'Traffic Mode'\n- Yellow: 'Conditional Experimental Mode' Overridden"), ""},
    {"PathWidth", tr("Path Width"), tr("Set how wide the driving path appears on your screen.\n\nDefault (6.1 feet / 1.9 meters) matches the width of a 2019 Lexus ES 350."), ""},
    {"RoadEdgesWidth", tr("Road Edges Width"), tr("Adjust how thick the road edges appear on the display.\n\nDefault matches half of the MUTCD standard lane line width of 4 inches."), ""},
    {"UnlimitedLength", tr("'Unlimited' Road UI Length"), tr("Extend the display of the path, lane lines, and road edges as far as the model can see."), ""},
  };

  for (const auto &[param, title, desc, icon] : advancedToggles) {
    AbstractControl *advancedVisualToggle;

    if (param == "DeveloperUI") {
      FrogPilotParamManageControl *developerUIToggle = new FrogPilotParamManageControl(param, title, desc, icon);
      QObject::connect(developerUIToggle, &FrogPilotParamManageControl::manageButtonClicked, [this]() {
        borderMetricsBtn->setEnabledButtons(0, hasBSM);

        std::set<QString> modifiedDeveloperUIKeys = developerUIKeys;

        if (!hasAutoTune) {
          modifiedDeveloperUIKeys.erase("LateralMetrics");
        }

        if (disableOpenpilotLongitudinal || !hasOpenpilotLongitudinal) {
          modifiedDeveloperUIKeys.erase("LongitudinalMetrics");
        }

        showToggles(modifiedDeveloperUIKeys);
      });
      advancedVisualToggle = developerUIToggle;
    } else if (param == "BorderMetrics") {
      std::vector<QString> borderToggles{"BlindSpotMetrics", "ShowSteering", "SignalMetrics"};
      std::vector<QString> borderToggleNames{tr("Blind Spot"), tr("Steering Torque"), tr("Turn Signal")};
      borderMetricsBtn = new FrogPilotButtonToggleControl(param, title, desc, borderToggles, borderToggleNames);
      advancedVisualToggle = borderMetricsBtn;
    } else if (param == "LateralMetrics") {
      std::vector<QString> lateralToggles{"TuningInfo"};
      std::vector<QString> lateralToggleNames{tr("Auto Tune")};
      advancedVisualToggle = new FrogPilotButtonToggleControl(param, title, desc, lateralToggles, lateralToggleNames);
    } else if (param == "LongitudinalMetrics") {
      std::vector<QString> longitudinalToggles{"LeadInfo", "JerkInfo"};
      std::vector<QString> longitudinalToggleNames{tr("Lead Info"), tr("Longitudinal Jerk")};
      advancedVisualToggle = new FrogPilotButtonToggleControl(param, title, desc, longitudinalToggles, longitudinalToggleNames);
    } else if (param == "NumericalTemp") {
      std::vector<QString> temperatureToggles{"Fahrenheit"};
      std::vector<QString> temperatureToggleNames{tr("Fahrenheit")};
      advancedVisualToggle = new FrogPilotButtonToggleControl(param, title, desc, temperatureToggles, temperatureToggleNames);
    } else if (param == "SidebarMetrics") {
      std::vector<QString> sidebarMetricsToggles{"ShowCPU", "ShowGPU", "ShowIP", "ShowMemoryUsage", "ShowStorageLeft", "ShowStorageUsed"};
      std::vector<QString> sidebarMetricsToggleNames{tr("CPU"), tr("GPU"), tr("IP"), tr("RAM"), tr("SSD Left"), tr("SSD Used")};
      FrogPilotButtonToggleControl *sidebarMetricsToggle = new FrogPilotButtonToggleControl(param, title, desc, sidebarMetricsToggles, sidebarMetricsToggleNames, false, 150);
      QObject::connect(sidebarMetricsToggle, &FrogPilotButtonToggleControl::buttonClicked, [this](int index) {
        if (index == 0) {
          params.putBool("ShowGPU", false);
        } else if (index == 1) {
          params.putBool("ShowCPU", false);
        } else if (index == 3) {
          params.putBool("ShowStorageLeft", false);
          params.putBool("ShowStorageUsed", false);
        } else if (index == 4) {
          params.putBool("ShowMemoryUsage", false);
          params.putBool("ShowStorageUsed", false);
        } else if (index == 5) {
          params.putBool("ShowMemoryUsage", false);
          params.putBool("ShowStorageLeft", false);
        }
      });
      advancedVisualToggle = sidebarMetricsToggle;

    } else if (param == "ModelUI") {
      FrogPilotParamManageControl *modelUIToggle = new FrogPilotParamManageControl(param, title, desc, icon);
      QObject::connect(modelUIToggle, &FrogPilotParamManageControl::manageButtonClicked, [this]() {
        std::set<QString> modifiedModelUIKeysKeys = modelUIKeys;

        if (disableOpenpilotLongitudinal || !hasOpenpilotLongitudinal) {
          modifiedModelUIKeysKeys.erase("HideLeadMarker");
        }

        showToggles(modifiedModelUIKeysKeys);
      });
      advancedVisualToggle = modelUIToggle;
    } else if (param == "LaneLinesWidth" || param == "RoadEdgesWidth") {
      advancedVisualToggle = new FrogPilotParamValueControl(param, title, desc, icon, 0, 24, tr(" inches"));
    } else if (param == "PathEdgeWidth") {
      advancedVisualToggle = new FrogPilotParamValueControl(param, title, desc, icon, 0, 100, tr("%"));
    } else if (param == "PathWidth") {
      advancedVisualToggle = new FrogPilotParamValueControl(param, title, desc, icon, 0, 10, tr(" feet"), std::map<int, QString>(), 0.1);

    } else {
      advancedVisualToggle = new ParamControl(param, title, desc, icon);
    }

    addItem(advancedVisualToggle);
    toggles[param] = advancedVisualToggle;

    tryConnect(advancedVisualToggle, &updateFrogPilotToggles);

    if (FrogPilotParamManageControl *frogPilotManageToggle = qobject_cast<FrogPilotParamManageControl*>(advancedVisualToggle)) {
      QObject::connect(frogPilotManageToggle, &FrogPilotParamManageControl::manageButtonClicked, this, &FrogPilotAdvancedVisualsPanel::openParentToggle);
    }

    QObject::connect(advancedVisualToggle, &AbstractControl::showDescriptionEvent, [this]() {
      update();
    });
  }

  QObject::connect(parent, &FrogPilotSettingsWindow::closeParentToggle, this, &FrogPilotAdvancedVisualsPanel::hideToggles);
  QObject::connect(parent, &FrogPilotSettingsWindow::updateMetric, this, &FrogPilotAdvancedVisualsPanel::updateMetric);
  QObject::connect(uiState(), &UIState::offroadTransition, this, &FrogPilotAdvancedVisualsPanel::updateCarToggles);

  hideToggles();
  updateMetric();
}

void FrogPilotAdvancedVisualsPanel::showEvent(QShowEvent *event) {
  disableOpenpilotLongitudinal = params.getBool("DisableOpenpilotLongitudinal");
}

void FrogPilotAdvancedVisualsPanel::updateCarToggles() {
  std::string carParams = params.get("CarParamsPersistent");
  if (!carParams.empty()) {
    AlignedBuffer aligned_buf;
    capnp::FlatArrayMessageReader cmsg(aligned_buf.align(carParams.data(), carParams.size()));
    cereal::CarParams::Reader CP = cmsg.getRoot<cereal::CarParams>();

    std::string carName = CP.getCarName();

    hasAutoTune = (carName == "hyundai" || carName == "toyota") && CP.getLateralTuning().which() == cereal::CarParams::LateralTuning::TORQUE;
    hasBSM = CP.getEnableBsm();
    hasOpenpilotLongitudinal = hasLongitudinalControl(CP);
  } else {
    hasAutoTune = true;
    hasBSM = true;
    hasOpenpilotLongitudinal = true;
  }
}

void FrogPilotAdvancedVisualsPanel::updateMetric() {
  bool previousIsMetric = isMetric;
  isMetric = params.getBool("IsMetric");

  if (isMetric != previousIsMetric) {
    double smallDistanceConversion = isMetric ? INCH_TO_CM : CM_TO_INCH;
    double distanceConversion = isMetric ? FOOT_TO_METER : METER_TO_FOOT;

    params.putFloatNonBlocking("LaneLinesWidth", params.getFloat("LaneLinesWidth") * smallDistanceConversion);
    params.putFloatNonBlocking("RoadEdgesWidth", params.getFloat("RoadEdgesWidth") * smallDistanceConversion);

    params.putFloatNonBlocking("PathWidth", params.getFloat("PathWidth") * distanceConversion);
  }

  FrogPilotParamValueControl *laneLinesWidthToggle = static_cast<FrogPilotParamValueControl*>(toggles["LaneLinesWidth"]);
  FrogPilotParamValueControl *pathWidthToggle = static_cast<FrogPilotParamValueControl*>(toggles["PathWidth"]);
  FrogPilotParamValueControl *roadEdgesWidthToggle = static_cast<FrogPilotParamValueControl*>(toggles["RoadEdgesWidth"]);

  if (isMetric) {
    laneLinesWidthToggle->setDescription(tr("Adjust how thick the lane lines appear on the display.\n\nDefault matches the Vienna standard of 10 centimeters."));
    roadEdgesWidthToggle->setDescription(tr("Adjust how thick the road edges appear on the display.\n\nDefault matches half of the Vienna standard of 10 centimeters."));

    laneLinesWidthToggle->updateControl(0, 60, tr(" centimeters"));
    roadEdgesWidthToggle->updateControl(0, 60, tr(" centimeters"));

    pathWidthToggle->updateControl(0, 3, tr(" meters"));
  } else {
    laneLinesWidthToggle->setDescription(tr("Adjust how thick the lane lines appear on the display.\n\nDefault matches the MUTCD standard of 4 inches."));
    roadEdgesWidthToggle->setDescription(tr("Adjust how thick the road edges appear on the display.\n\nDefault matches half of the MUTCD standard of 4 inches."));

    laneLinesWidthToggle->updateControl(0, 24, tr(" inches"));
    roadEdgesWidthToggle->updateControl(0, 24, tr(" inches"));

    pathWidthToggle->updateControl(0, 10, tr(" feet"));
  }
}

void FrogPilotAdvancedVisualsPanel::showToggles(const std::set<QString> &keys) {
  setUpdatesEnabled(false);

  for (auto &[key, toggle] : toggles) {
    toggle->setVisible(keys.find(key) != keys.end());
  }

  setUpdatesEnabled(true);
  update();
}

void FrogPilotAdvancedVisualsPanel::hideToggles() {
  setUpdatesEnabled(false);

  for (auto &[key, toggle] : toggles) {
    bool subToggles = developerUIKeys.find(key) != developerUIKeys.end() ||
                      modelUIKeys.find(key) != modelUIKeys.end();

    toggle->setVisible(!subToggles);
  }

  setUpdatesEnabled(true);
  update();
}
