#include <filesystem>
#include <iostream>

#include "selfdrive/frogpilot/navigation/ui/navigation_settings.h"
#include "selfdrive/frogpilot/ui/qt/offroad/advanced_driving_settings.h"
#include "selfdrive/frogpilot/ui/qt/offroad/advanced_visual_settings.h"
#include "selfdrive/frogpilot/ui/qt/offroad/data_settings.h"
#include "selfdrive/frogpilot/ui/qt/offroad/device_settings.h"
#include "selfdrive/frogpilot/ui/qt/offroad/frogpilot_settings.h"
#include "selfdrive/frogpilot/ui/qt/offroad/lateral_settings.h"
#include "selfdrive/frogpilot/ui/qt/offroad/longitudinal_settings.h"
#include "selfdrive/frogpilot/ui/qt/offroad/sounds_settings.h"
#include "selfdrive/frogpilot/ui/qt/offroad/utilities.h"
#include "selfdrive/frogpilot/ui/qt/offroad/vehicle_settings.h"
#include "selfdrive/frogpilot/ui/qt/offroad/visual_settings.h"

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

FrogPilotSettingsWindow::FrogPilotSettingsWindow(SettingsWindow *parent) : QFrame(parent) {
  mainLayout = new QStackedLayout(this);

  frogpilotSettingsWidget = new QWidget(this);
  QVBoxLayout *frogpilotSettingsLayout = new QVBoxLayout(frogpilotSettingsWidget);
  frogpilotSettingsLayout->setContentsMargins(50, 25, 50, 25);

  FrogPilotListWidget *list = new FrogPilotListWidget(frogpilotSettingsWidget);

  FrogPilotAdvancedDrivingPanel *frogpilotAdvancedDrivingPanel = new FrogPilotAdvancedDrivingPanel(this);
  QObject::connect(frogpilotAdvancedDrivingPanel, &FrogPilotAdvancedDrivingPanel::openParentToggle, this, &FrogPilotSettingsWindow::openParentToggle);
  QObject::connect(frogpilotAdvancedDrivingPanel, &FrogPilotAdvancedDrivingPanel::openSubParentToggle, this, &FrogPilotSettingsWindow::openSubParentToggle);
  QObject::connect(frogpilotAdvancedDrivingPanel, &FrogPilotAdvancedDrivingPanel::openSubSubParentToggle, this, &FrogPilotSettingsWindow::openSubSubParentToggle);

  FrogPilotAdvancedVisualsPanel *frogpilotAdvancedVisualsPanel = new FrogPilotAdvancedVisualsPanel(this);
  QObject::connect(frogpilotAdvancedVisualsPanel, &FrogPilotAdvancedVisualsPanel::openParentToggle, this, &FrogPilotSettingsWindow::openParentToggle);

  FrogPilotDevicePanel *frogpilotDevicePanel = new FrogPilotDevicePanel(this);
  QObject::connect(frogpilotDevicePanel, &FrogPilotDevicePanel::openParentToggle, this, &FrogPilotSettingsWindow::openParentToggle);

  FrogPilotLateralPanel *frogpilotLateralPanel = new FrogPilotLateralPanel(this);
  QObject::connect(frogpilotLateralPanel, &FrogPilotLateralPanel::openParentToggle, this, &FrogPilotSettingsWindow::openParentToggle);

  FrogPilotLongitudinalPanel *frogpilotLongitudinalPanel = new FrogPilotLongitudinalPanel(this);
  QObject::connect(frogpilotLongitudinalPanel, &FrogPilotLongitudinalPanel::openParentToggle, this, &FrogPilotSettingsWindow::openParentToggle);
  QObject::connect(frogpilotLongitudinalPanel, &FrogPilotLongitudinalPanel::openSubParentToggle, this, &FrogPilotSettingsWindow::openSubParentToggle);

  FrogPilotSoundsPanel *frogpilotSoundsPanel = new FrogPilotSoundsPanel(this);
  QObject::connect(frogpilotSoundsPanel, &FrogPilotSoundsPanel::openParentToggle, this, &FrogPilotSettingsWindow::openParentToggle);

  FrogPilotVisualsPanel *frogpilotVisualsPanel = new FrogPilotVisualsPanel(this);
  QObject::connect(frogpilotVisualsPanel, &FrogPilotVisualsPanel::openParentToggle, this, &FrogPilotSettingsWindow::openParentToggle);
  QObject::connect(frogpilotVisualsPanel, &FrogPilotVisualsPanel::openSubParentToggle, this, &FrogPilotSettingsWindow::openSubParentToggle);

  std::vector<std::pair<QString, std::vector<QWidget*>>> panels = {
    {tr("Advanced Settings"), {frogpilotAdvancedDrivingPanel, frogpilotAdvancedVisualsPanel}},
    {tr("Alerts and Sounds"), {frogpilotSoundsPanel}},
    {tr("Display Settings"), {frogpilotVisualsPanel}},
    {tr("Driving Controls"), {frogpilotLongitudinalPanel, frogpilotLateralPanel}},
    {tr("Navigation"), {new FrogPilotNavigationPanel(this)}},
    {tr("System Management"), {new FrogPilotDataPanel(this), frogpilotDevicePanel, new UtilitiesPanel(this)}},
    {tr("Vehicle Controls"), {new FrogPilotVehiclesPanel(this)}}
  };

  std::vector<QString> icons = {
    "../frogpilot/assets/toggle_icons/icon_advanced.png",
    "../frogpilot/assets/toggle_icons/icon_sound.png",
    "../frogpilot/assets/toggle_icons/icon_screen.png",
    "../frogpilot/assets/toggle_icons/icon_steering.png",
    "../frogpilot/assets/toggle_icons/icon_map.png",
    "../frogpilot/assets/toggle_icons/icon_system.png",
    "../frogpilot/assets/toggle_icons/icon_vehicle.png",
  };

  std::vector<QString> descriptions = {
    tr("Advanced FrogPilot features for experienced users."),
    tr("Control FrogPilot's sounds and alerts."),
    tr("Modify the appearance of FrogPilot."),
    tr("Control how FrogPilot handles your vehicle's acceleration, braking, and steering."),
    tr("Download offline maps and manage 'Navigate On openpilot (NOO)' settings."),
    tr("Tools and system utilities for maintaining and troubleshooting FrogPilot."),
    tr("Configure settings specific to your vehicle's make and model.")
  };

  std::vector<std::vector<QString>> buttonLabels = {
    {tr("DRIVING"), tr("VISUALS")},
    {tr("MANAGE")},
    {tr("MANAGE")},
    {tr("GAS / BRAKE"), tr("STEERING")},
    {tr("MANAGE")},
    {tr("DATA"), tr("DEVICE"), tr("UTILITIES")},
    {tr("MANAGE")}
  };

  for (size_t i = 0; i < panels.size(); ++i) {
    addPanelControl(list, panels[i].first, descriptions[i], buttonLabels[i], icons[i], panels[i].second);
  }

  frogpilotSettingsLayout->addWidget(new ScrollView(list, frogpilotSettingsWidget));
  frogpilotSettingsLayout->addStretch(1);

  mainLayout->addWidget(frogpilotSettingsWidget);
  mainLayout->setCurrentWidget(frogpilotSettingsWidget);

  QObject::connect(parent, &SettingsWindow::closePanel, [this]() {mainLayout->setCurrentWidget(frogpilotSettingsWidget);});
  QObject::connect(parent, &SettingsWindow::closeParentToggle, this, &FrogPilotSettingsWindow::closeParentToggle);
  QObject::connect(parent, &SettingsWindow::closeSubParentToggle, this, &FrogPilotSettingsWindow::closeSubParentToggle);
  QObject::connect(parent, &SettingsWindow::closeSubSubParentToggle, this, &FrogPilotSettingsWindow::closeSubSubParentToggle);
  QObject::connect(parent, &SettingsWindow::updateMetric, this, &FrogPilotSettingsWindow::updateMetric);
  QObject::connect(uiState(), &UIState::offroadTransition, this, &FrogPilotSettingsWindow::updateCarVariables);

  updateCarVariables();
}

void FrogPilotSettingsWindow::showEvent(QShowEvent *event) {
  disableOpenpilotLongitudinal = params.getBool("DisableOpenpilotLongitudinal");
}

void FrogPilotSettingsWindow::updateCarVariables() {
  float currentFrictionStock = params.getFloat("SteerFrictionStock");
  float currentKPStock = params.getFloat("SteerKPStock");
  float currentLatAccelStock = params.getFloat("SteerLatAccelStock");
  float currentRatioStock = params.getFloat("SteerRatioStock");

  std::string carParams = params.get("CarParamsPersistent");
  if (!carParams.empty()) {
    AlignedBuffer aligned_buf;
    capnp::FlatArrayMessageReader cmsg(aligned_buf.align(carParams.data(), carParams.size()));
    cereal::CarParams::Reader CP = cmsg.getRoot<cereal::CarParams>();
    cereal::CarParams::SafetyModel safetyModel = CP.getSafetyConfigs()[0].getSafetyModel();

    std::string carFingerprint = CP.getCarFingerprint();
    std::string carName = CP.getCarName();

    hasAutoTune = (carName == "hyundai" || carName == "toyota") && CP.getLateralTuning().which() == cereal::CarParams::LateralTuning::TORQUE;
    hasBSM = CP.getEnableBsm();
    hasDashSpeedLimits = carName == "hyundai" || carName == "toyota";
    hasExperimentalOpenpilotLongitudinal = CP.getExperimentalLongitudinalAvailable();
    hasNNFFLog = checkNNFFLogFileExists(carFingerprint);
    hasOpenpilotLongitudinal = hasLongitudinalControl(CP);
    hasPCMCruise = CP.getPcmCruise();
    hasSNG = CP.getMinEnableSpeed() <= 0;
    isGM = carName == "gm";
    isGMPCMCruise = CP.getCarName() == "gm" && CP.getPcmCruise();
    isHKGCanFd = carName == "hyundai" && safetyModel == cereal::CarParams::SafetyModel::HYUNDAI_CANFD;
    isImpreza = carFingerprint == "SUBARU_IMPREZA";
    isSubaru = carName == "subaru";
    isToyota = carName == "toyota";
    isVolt = carFingerprint == "CHEVROLET_VOLT";
    forcingAutoTune = params.getBool("LateralTune") && params.getBool("ForceAutoTune");
    steerFrictionStock = CP.getLateralTuning().getTorque().getFriction();
    steerKPStock = CP.getLateralTuning().getTorque().getKp();
    steerLatAccelStock = CP.getLateralTuning().getTorque().getLatAccelFactor();
    steerRatioStock = CP.getSteerRatio();

    if (currentFrictionStock != steerFrictionStock) {
      if (params.getFloat("SteerFriction") == currentFrictionStock) {
        params.putFloatNonBlocking("SteerFriction", steerFrictionStock);
      }
      params.putFloatNonBlocking("SteerFrictionStock", steerFrictionStock);
    }

    if (currentKPStock != steerKPStock) {
      if (params.getFloat("SteerKP") == currentKPStock) {
        params.putFloatNonBlocking("SteerKP", steerKPStock);
      }
      params.putFloatNonBlocking("SteerKPStock", steerKPStock);
    }

    if (currentLatAccelStock != steerLatAccelStock) {
      if (params.getFloat("SteerLatAccel") == steerLatAccelStock) {
        params.putFloatNonBlocking("SteerLatAccel", steerLatAccelStock);
      }
      params.putFloatNonBlocking("SteerLatAccelStock", steerLatAccelStock);
    }

    if (currentRatioStock != steerRatioStock) {
      if (params.getFloat("SteerRatio") == steerRatioStock) {
        params.putFloatNonBlocking("SteerRatio", steerRatioStock);
      }
      params.putFloatNonBlocking("SteerRatioStock", steerRatioStock);
    }

    uiState()->scene.has_auto_tune = hasAutoTune || forcingAutoTune;
  } else {
    hasAutoTune = true;
    hasBSM = true;
    hasDashSpeedLimits = true;
    hasExperimentalOpenpilotLongitudinal = false;
    hasNNFFLog = true;
    hasOpenpilotLongitudinal = true;
    hasPCMCruise = true;
    hasSNG = false;
    isGMPCMCruise = false;
    isHKGCanFd = true;
    isImpreza = true;
    isSubaru = false;
    isToyota = true;
    isVolt = true;
  }

  std::string liveTorqueParamsKey;
  if (params.getBool("ModelManagement")) {
    QString model = QString::fromStdString(params.get("ModelName"));
    QString part_model_param = processModelName(model);
    liveTorqueParamsKey = part_model_param.toStdString() + "LiveTorqueParameters";
  } else {
    liveTorqueParamsKey = "LiveTorqueParameters";
  }

  if (params.checkKey(liveTorqueParamsKey)) {
    auto torqueParams = params.get(liveTorqueParamsKey);
    if (!torqueParams.empty()) {
      AlignedBuffer aligned_buf;
      capnp::FlatArrayMessageReader cmsg(aligned_buf.align(torqueParams.data(), torqueParams.size()));
      cereal::Event::Reader LTP = cmsg.getRoot<cereal::Event>();

      auto liveTorqueParams = LTP.getLiveTorqueParameters();

      liveValid = liveTorqueParams.getLiveValid();
    } else {
      liveValid = false;
    }
  } else {
    liveValid = false;
  }

  emit updateCarToggles();
}

void FrogPilotSettingsWindow::addPanelControl(FrogPilotListWidget *list, const QString &title, const QString &desc, const std::vector<QString> &button_labels, const QString &icon, const std::vector<QWidget*> &panels) {
  std::vector<QWidget*> panelContainers;
  panelContainers.reserve(panels.size());

  for (QWidget *panel : panels) {
    QWidget *panelContainer = new QWidget(this);
    QVBoxLayout *panelLayout = new QVBoxLayout(panelContainer);
    panelLayout->setContentsMargins(50, 25, 50, 25);
    panelLayout->addWidget(panel);
    panelContainers.push_back(panelContainer);
  }

  FrogPilotButtonsControl *button = new FrogPilotButtonsControl(title, desc, button_labels, false, true, icon);
  QObject::connect(button, &FrogPilotButtonsControl::buttonClicked, [this, panelContainers](int buttonId) {
    if (buttonId < panelContainers.size()) {
      QWidget *selectedPanel = panelContainers[buttonId];
      if (mainLayout->indexOf(selectedPanel) == -1) {
        mainLayout->addWidget(selectedPanel);
      }
      mainLayout->setCurrentWidget(selectedPanel);
    }
    openPanel();
  });

  list->addItem(button);
}
