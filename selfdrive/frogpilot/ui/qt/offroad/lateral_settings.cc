#include <filesystem>
#include <iostream>

#include "selfdrive/frogpilot/ui/qt/offroad/lateral_settings.h"

bool checkNNFFLogFileExists(const std::string &carFingerprint) {
  std::filesystem::path latModelsPath("../car/torque_data/lat_models");
  if (!std::filesystem::exists(latModelsPath)) {
    std::cerr << "Lat models directory does not exist." << std::endl;
    return false;
  }

  for (const std::filesystem::directory_entry &entry : std::filesystem::directory_iterator(latModelsPath)) {
    std::string filename = entry.path().filename().string();
    if (filename.find(carFingerprint) == 0) {
      std::cout << "NNFF supports fingerprint: " << filename << std::endl;
      return true;
    }
  }
  return false;
}

FrogPilotLateralPanel::FrogPilotLateralPanel(FrogPilotSettingsWindow *parent) : FrogPilotListWidget(parent) {
  const std::vector<std::tuple<QString, QString, QString, QString>> lateralToggles {
    {"AlwaysOnLateral", tr("Always on Lateral"), tr("Maintain openpilot lateral control when the brake or gas pedals are used.\n\nDeactivation occurs only through the 'Cruise Control' button."), "../frogpilot/assets/toggle_icons/icon_always_on_lateral.png"},
    {"AlwaysOnLateralLKAS", tr("Control Via LKAS Button"), tr("Enable or disable 'Always On Lateral' by clicking your 'LKAS' button."), ""},
    {"AlwaysOnLateralMain", tr("Enable On Cruise Main"), tr("Enable 'Always On Lateral' by clicking your 'Cruise Control' button without requiring openpilot to be enabled first."), ""},
    {"PauseAOLOnBrake", tr("Pause On Brake Below"), tr("Pause 'Always On Lateral' when the brake pedal is being pressed below the set speed."), ""},
    {"HideAOLStatusBar", tr("Hide the Status Bar"), tr("Don't use the status bar for 'Always On Lateral'."), ""},

    {"LaneChangeCustomizations", tr("Lane Change Customizations"), tr("Customize the lane change behaviors in openpilot."), "../frogpilot/assets/toggle_icons/icon_lane.png"},
    {"LaneChangeTime", tr("Lane Change Timer"), tr("Set a delay before executing a lane change."), ""},
    {"LaneDetectionWidth", tr("Lane Detection Threshold"), tr("Set the required lane width to be qualified as a lane."), ""},
    {"MinimumLaneChangeSpeed", tr("Minimum Lane Change Speed"), tr("Customize the minimum driving speed to allow openpilot to change lanes."), ""},
    {"NudgelessLaneChange", tr("Nudgeless Lane Change"), tr("Enable lane changes without requiring manual steering input."), ""},
    {"OneLaneChange", tr("One Lane Change Per Signal"), tr("Only allow one lane change per turn signal activation."), ""},

    {"LateralTune", tr("Lateral Tuning"), tr("Modify openpilot's steering behavior."), "../frogpilot/assets/toggle_icons/icon_lateral_tune.png"},
    {"ForceAutoTune", tr("Force Auto Tune"), tr("Forces comma's auto lateral tuning for unsupported vehicles."), ""},
    {"ForceAutoTuneOff", tr("Force Auto Tune Off"), tr("Forces comma's auto lateral tuning off for supported vehicles."), ""},
    {"NNFF", tr("NNFF"), tr("Use Twilsonco's Neural Network Feedforward for enhanced precision in lateral control."), ""},
    {"NNFFLite", tr("Smoother Entry and Exit for Curves"), tr("Uses Twilsonco's steering torque tweak to provide smoother handling when entering and exiting curves."), ""},
    {"TacoTune", tr("Taco Tune"), tr("Use comma's 'Taco Tune' designed for handling left and right turns."), ""},
    {"TurnDesires", tr("Use Turn Desires"), tr("Use turn desires for greater precision in turns below the minimum lane change speed."), ""},

    {"QOLControls", tr("Quality of Life"), tr("Miscellaneous quality of life changes to improve your overall openpilot experience."), "../frogpilot/assets/toggle_icons/quality_of_life.png"},
    {"PauseLateralSpeed", tr("Pause Lateral Below"), tr("Pause lateral control on all speeds below the set speed."), ""},
  };

  for (const auto &[param, title, desc, icon] : lateralToggles) {
    AbstractControl *lateralToggle;

    if (param == "AlwaysOnLateral") {
      FrogPilotParamManageControl *aolToggle = new FrogPilotParamManageControl(param, title, desc, icon);
      QObject::connect(aolToggle, &FrogPilotParamManageControl::manageButtonClicked, [this]() {
        std::set<QString> modifiedAOLKeys = aolKeys;

        if (isSubaru || (params.getBool("ExperimentalModeActivation") && params.getBool("ExperimentalModeViaLKAS"))) {
          modifiedAOLKeys.erase("AlwaysOnLateralLKAS");
        }

        showToggles(modifiedAOLKeys);
      });
      lateralToggle = aolToggle;
    } else if (param == "PauseAOLOnBrake") {
      lateralToggle = new FrogPilotParamValueControl(param, title, desc, icon, 0, 99, tr("mph"));

    } else if (param == "LateralTune") {
      FrogPilotParamManageControl *lateralTuneToggle = new FrogPilotParamManageControl(param, title, desc, icon);
      QObject::connect(lateralTuneToggle, &FrogPilotParamManageControl::manageButtonClicked, [this]() {
        std::set<QString> modifiedLateralTuneKeys = lateralTuneKeys;

        bool usingNNFF = hasNNFFLog && params.getBool("LateralTune") && params.getBool("NNFF");
        if (usingNNFF) {
          modifiedLateralTuneKeys.erase("ForceAutoTune");
          modifiedLateralTuneKeys.erase("ForceAutoTuneOff");
        } else {
          if (hasAutoTune) {
            modifiedLateralTuneKeys.erase("ForceAutoTune");
          } else {
            modifiedLateralTuneKeys.erase("ForceAutoTuneOff");
          }
        }

        if (!hasNNFFLog) {
          modifiedLateralTuneKeys.erase("NNFF");
        } else if (usingNNFF) {
          modifiedLateralTuneKeys.erase("NNFFLite");
        }

        showToggles(modifiedLateralTuneKeys);
      });
      lateralToggle = lateralTuneToggle;

    } else if (param == "QOLControls") {
      FrogPilotParamManageControl *qolToggle = new FrogPilotParamManageControl(param, title, desc, icon);
      QObject::connect(qolToggle, &FrogPilotParamManageControl::manageButtonClicked, [this]() {
        showToggles(qolKeys);
      });
      lateralToggle = qolToggle;
    } else if (param == "PauseLateralSpeed") {
      std::vector<QString> pauseLateralToggles{"PauseLateralOnSignal"};
      std::vector<QString> pauseLateralToggleNames{"Turn Signal Only"};
      lateralToggle = new FrogPilotParamValueButtonControl(param, title, desc, icon, 0, 99, tr("mph"), std::map<int, QString>(), 1, pauseLateralToggles, pauseLateralToggleNames);
    } else if (param == "PauseLateralOnSignal") {
      lateralToggle = new FrogPilotParamValueControl(param, title, desc, icon, 0, 99, tr("mph"));

    } else if (param == "LaneChangeCustomizations") {
      FrogPilotParamManageControl *laneChangeToggle = new FrogPilotParamManageControl(param, title, desc, icon);
      QObject::connect(laneChangeToggle, &FrogPilotParamManageControl::manageButtonClicked, [this]() {
        showToggles(laneChangeKeys);
      });
      lateralToggle = laneChangeToggle;
    } else if (param == "LaneChangeTime") {
      std::map<int, QString> laneChangeTimeLabels;
      for (int i = 0; i <= 10; i++) {
        laneChangeTimeLabels[i] = i == 0 ? "Instant" : QString::number(i / 2.0) + " seconds";
      }
      lateralToggle = new FrogPilotParamValueControl(param, title, desc, icon, 0, 10, QString(), laneChangeTimeLabels);
    } else if (param == "LaneDetectionWidth") {
      lateralToggle = new FrogPilotParamValueControl(param, title, desc, icon, 0, 100, tr(" feet"), std::map<int, QString>(), 0.1);
    } else if (param == "MinimumLaneChangeSpeed") {
      lateralToggle = new FrogPilotParamValueControl(param, title, desc, icon, 0, 99, tr("mph"));

    } else {
      lateralToggle = new ParamControl(param, title, desc, icon);
    }

    addItem(lateralToggle);
    toggles[param.toStdString()] = lateralToggle;

    tryConnect<ToggleControl>(lateralToggle, &ToggleControl::toggleFlipped, this, updateFrogPilotToggles);
    tryConnect<FrogPilotButtonToggleControl>(lateralToggle, &FrogPilotButtonToggleControl::buttonClicked, this, updateFrogPilotToggles);
    tryConnect<FrogPilotParamManageControl>(lateralToggle, &FrogPilotParamManageControl::manageButtonClicked, this, &FrogPilotLateralPanel::openParentToggle);
    tryConnect<FrogPilotParamValueControl>(lateralToggle, &FrogPilotParamValueControl::valueChanged, this, updateFrogPilotToggles);

    QObject::connect(lateralToggle, &AbstractControl::showDescriptionEvent, [this]() {
      update();
    });
  }

  QObject::connect(static_cast<ToggleControl*>(toggles["AlwaysOnLateralLKAS"]), &ToggleControl::toggleFlipped, [this](bool state) {
    if (state && params.getBool("ExperimentalModeViaLKAS")) {
      params.putBoolNonBlocking("ExperimentalModeViaLKAS", false);
    }
  });

  QObject::connect(static_cast<ToggleControl*>(toggles["NNFF"]), &ToggleControl::toggleFlipped, [this](bool state) {
    std::set<QString> modifiedLateralTuneKeys = lateralTuneKeys;

    bool usingNNFF = hasNNFFLog && state;
    if (usingNNFF) {
      modifiedLateralTuneKeys.erase("ForceAutoTune");
      modifiedLateralTuneKeys.erase("ForceAutoTuneOff");
    } else {
      if (hasAutoTune) {
        modifiedLateralTuneKeys.erase("ForceAutoTune");
      } else {
        modifiedLateralTuneKeys.erase("ForceAutoTuneOff");
      }
    }

    if (!hasNNFFLog) {
      modifiedLateralTuneKeys.erase("NNFF");
    } else if (usingNNFF) {
      modifiedLateralTuneKeys.erase("NNFFLite");
    }

    showToggles(modifiedLateralTuneKeys);
  });

  std::set<QString> rebootKeys = {"AlwaysOnLateral", "NNFF", "NNFFLite"};
  for (const QString &key : rebootKeys) {
    QObject::connect(static_cast<ToggleControl*>(toggles[key.toStdString().c_str()]), &ToggleControl::toggleFlipped, [this, key](bool state) {
      if (started) {
        if (key == "AlwaysOnLateral" && state) {
          if (FrogPilotConfirmationDialog::toggle(tr("Reboot required to take effect."), tr("Reboot Now"), this)) {
            Hardware::reboot();
          }
        } else if (key != "AlwaysOnLateral") {
          if (FrogPilotConfirmationDialog::toggle(tr("Reboot required to take effect."), tr("Reboot Now"), this)) {
            Hardware::reboot();
          }
        }
      }
    });
  }

  QObject::connect(parent, &FrogPilotSettingsWindow::closeParentToggle, this, &FrogPilotLateralPanel::hideToggles);
  QObject::connect(parent, &FrogPilotSettingsWindow::updateMetric, this, &FrogPilotLateralPanel::updateMetric);
  QObject::connect(uiState(), &UIState::offroadTransition, this, &FrogPilotLateralPanel::updateCarToggles);
  QObject::connect(uiState(), &UIState::uiUpdate, this, &FrogPilotLateralPanel::updateState);

  updateMetric();
}

void FrogPilotLateralPanel::updateState(const UIState &s) {
  if (!isVisible()) return;

  started = s.scene.started;
}

void FrogPilotLateralPanel::updateCarToggles() {
  auto carParams = params.get("CarParamsPersistent");
  if (!carParams.empty()) {
    AlignedBuffer aligned_buf;
    capnp::FlatArrayMessageReader cmsg(aligned_buf.align(carParams.data(), carParams.size()));
    cereal::CarParams::Reader CP = cmsg.getRoot<cereal::CarParams>();

    auto carFingerprint = CP.getCarFingerprint();
    auto carName = CP.getCarName();

    hasAutoTune = (carName == "hyundai" || carName == "toyota") && CP.getLateralTuning().which() == cereal::CarParams::LateralTuning::TORQUE;
    bool forcingAutoTune = params.getBool("LateralTune") && params.getBool("ForceAutoTune");
    uiState()->scene.has_auto_tune = hasAutoTune || forcingAutoTune;
    hasNNFFLog = checkNNFFLogFileExists(carFingerprint);
    isSubaru = carName == "subaru";
  } else {
    hasAutoTune = false;
    hasNNFFLog = true;
  }

  hideToggles();
}

void FrogPilotLateralPanel::updateMetric() {
  bool previousIsMetric = isMetric;
  isMetric = params.getBool("IsMetric");

  if (isMetric != previousIsMetric) {
    double distanceConversion = isMetric ? FOOT_TO_METER : METER_TO_FOOT;
    double speedConversion = isMetric ? MILE_TO_KM : KM_TO_MILE;

    params.putFloatNonBlocking("LaneDetectionWidth", params.getFloat("LaneDetectionWidth") * distanceConversion);

    params.putFloatNonBlocking("MinimumLaneChangeSpeed", params.getFloat("MinimumLaneChangeSpeed") * speedConversion);
    params.putFloatNonBlocking("PauseAOLOnBrake", params.getFloat("PauseAOLOnBrake") * speedConversion);
    params.putFloatNonBlocking("PauseLateralOnSignal", params.getFloat("PauseLateralOnSignal") * speedConversion);
    params.putFloatNonBlocking("PauseLateralSpeed", params.getFloat("PauseLateralSpeed") * speedConversion);
  }

  FrogPilotParamValueControl *laneWidthToggle = static_cast<FrogPilotParamValueControl*>(toggles["LaneDetectionWidth"]);
  FrogPilotParamValueControl *minimumLaneChangeSpeedToggle = static_cast<FrogPilotParamValueControl*>(toggles["MinimumLaneChangeSpeed"]);
  FrogPilotParamValueControl *pauseAOLOnBrakeToggle = static_cast<FrogPilotParamValueControl*>(toggles["PauseAOLOnBrake"]);
  FrogPilotParamValueControl *pauseLateralToggle = static_cast<FrogPilotParamValueControl*>(toggles["PauseLateralSpeed"]);

  if (isMetric) {
    minimumLaneChangeSpeedToggle->updateControl(0, 150, tr("kph"));
    pauseAOLOnBrakeToggle->updateControl(0, 99, tr("kph"));
    pauseLateralToggle->updateControl(0, 99, tr("kph"));

    laneWidthToggle->updateControl(0, 30, tr(" meters"));
  } else {
    minimumLaneChangeSpeedToggle->updateControl(0, 99, tr("mph"));
    pauseAOLOnBrakeToggle->updateControl(0, 99, tr("mph"));
    pauseLateralToggle->updateControl(0, 99, tr("mph"));

    laneWidthToggle->updateControl(0, 100, tr(" feet"));
  }
}

void FrogPilotLateralPanel::showToggles(std::set<QString> &keys) {
  setUpdatesEnabled(false);

  for (auto &[key, toggle] : toggles) {
    toggle->setVisible(keys.find(key.c_str()) != keys.end());
  }

  setUpdatesEnabled(true);
  update();
}

void FrogPilotLateralPanel::hideToggles() {
  setUpdatesEnabled(false);

  for (auto &[key, toggle] : toggles) {
    bool subToggles = aolKeys.find(key.c_str()) != aolKeys.end() ||
                      laneChangeKeys.find(key.c_str()) != laneChangeKeys.end() ||
                      lateralTuneKeys.find(key.c_str()) != lateralTuneKeys.end() ||
                      qolKeys.find(key.c_str()) != qolKeys.end();

    toggle->setVisible(!subToggles);
  }

  setUpdatesEnabled(true);
  update();
}
