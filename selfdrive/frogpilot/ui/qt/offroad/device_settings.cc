#include "selfdrive/frogpilot/ui/qt/offroad/device_settings.h"

FrogPilotDevicePanel::FrogPilotDevicePanel(FrogPilotSettingsWindow *parent) : FrogPilotListWidget(parent) {
  const std::vector<std::tuple<QString, QString, QString, QString>> deviceToggles {
    {"DeviceManagement", tr("Device Management"), tr("Tweak your device's behaviors to your personal preferences."), "../frogpilot/assets/toggle_icons/icon_device.png"},
    {"DeviceShutdown", tr("Device Shutdown Timer"), tr("Configure how quickly the device shuts down after going offroad."), ""},
    {"NoLogging", tr("Disable Logging"), tr("Turn off all data tracking to enhance privacy or reduce thermal load."), ""},
    {"NoUploads", tr("Disable Uploads"), tr("Turn off all data uploads to comma's servers."), ""},
    {"IncreaseThermalLimits", tr("Increase Thermal Safety Limit"), tr("Allow the device to run at a temperature above comma's recommended thermal limits."), ""},
    {"LowVoltageShutdown", tr("Low Voltage Shutdown Threshold"), tr("Automatically shut the device down when your battery reaches a specific voltage level to prevent killing your battery."), ""},
    {"OfflineMode", tr("Offline Mode"), tr("Allow the device to be offline indefinitely."), ""},
  };

  for (const auto &[param, title, desc, icon] : deviceToggles) {
    AbstractControl *deviceToggle;

    if (param == "DeviceManagement") {
      FrogPilotParamManageControl *deviceManagementToggle = new FrogPilotParamManageControl(param, title, desc, icon);
      QObject::connect(deviceManagementToggle, &FrogPilotParamManageControl::manageButtonClicked, [this]() {
        showToggles(deviceManagementKeys);
      });
      deviceToggle = deviceManagementToggle;
    } else if (param == "DeviceShutdown") {
      std::map<int, QString> shutdownLabels;
      for (int i = 0; i <= 33; i++) {
        shutdownLabels[i] = i == 0 ? tr("5 mins") : i <= 3 ? QString::number(i * 15) + tr(" mins") : QString::number(i - 3) + (i == 4 ? tr(" hour") : tr(" hours"));
      }
      deviceToggle = new FrogPilotParamValueControl(param, title, desc, icon, 0, 33, QString(), shutdownLabels);
    } else if (param == "NoUploads") {
      std::vector<QString> uploadsToggles{"DisableOnroadUploads"};
      std::vector<QString> uploadsToggleNames{tr("Only Onroad")};
      deviceToggle = new FrogPilotButtonToggleControl(param, title, desc, uploadsToggles, uploadsToggleNames);
    } else if (param == "LowVoltageShutdown") {
      deviceToggle = new FrogPilotParamValueControl(param, title, desc, icon, 11.8, 12.5, tr(" volts"), std::map<int, QString>(), 0.01);

    } else {
      deviceToggle = new ParamControl(param, title, desc, icon);
    }

    addItem(deviceToggle);
    toggles[param.toStdString()] = deviceToggle;

    tryConnect<ToggleControl>(deviceToggle, &ToggleControl::toggleFlipped, this, updateFrogPilotToggles);
    tryConnect<FrogPilotButtonToggleControl>(deviceToggle, &FrogPilotButtonToggleControl::buttonClicked, this, updateFrogPilotToggles);
    tryConnect<FrogPilotParamManageControl>(deviceToggle, &FrogPilotParamManageControl::manageButtonClicked, this, &FrogPilotDevicePanel::openParentToggle);
    tryConnect<FrogPilotParamValueControl>(deviceToggle, &FrogPilotParamValueControl::valueChanged, this, updateFrogPilotToggles);

    QObject::connect(deviceToggle, &AbstractControl::showDescriptionEvent, [this]() {
      update();
    });
  }

  QObject::connect(static_cast<ToggleControl*>(toggles["IncreaseThermalLimits"]), &ToggleControl::toggleFlipped, [this](bool state) {
    if (state) {
      FrogPilotConfirmationDialog::toggleAlert(
        tr("WARNING: This can cause premature wear or damage by running the device over comma's recommended temperature limits!"),
        tr("I understand the risks."), this);
    }
  });

  QObject::connect(static_cast<ToggleControl*>(toggles["NoLogging"]), &ToggleControl::toggleFlipped, [this](bool state) {
    if (state) {
      FrogPilotConfirmationDialog::toggleAlert(
        tr("WARNING: This will prevent your drives from being recorded and the data will be unobtainable!"),
        tr("I understand the risks."), this);
    }
  });

  QObject::connect(static_cast<ToggleControl*>(toggles["NoUploads"]), &ToggleControl::toggleFlipped, [this](bool state) {
    if (state) {
      FrogPilotConfirmationDialog::toggleAlert(
        tr("WARNING: This will prevent your drives from appearing on comma connect which may impact debugging and support!"),
        tr("I understand the risks."), this);
    }
  });

  QObject::connect(parent, &FrogPilotSettingsWindow::closeParentToggle, this, &FrogPilotDevicePanel::hideToggles);

  hideToggles();
}

void FrogPilotDevicePanel::showToggles(std::set<QString> &keys) {
  setUpdatesEnabled(false);

  for (auto &[key, toggle] : toggles) {
    if (keys.find(key.c_str()) != keys.end()) {
      toggle->setVisible(keys.find(key.c_str()) != keys.end());
    }
  }

  setUpdatesEnabled(true);
  update();
}

void FrogPilotDevicePanel::hideToggles() {
  setUpdatesEnabled(false);

  for (auto &[key, toggle] : toggles) {
    bool subToggles = deviceManagementKeys.find(key.c_str()) != deviceManagementKeys.end();
    toggle->setVisible(!subToggles);
  }

  setUpdatesEnabled(true);
  update();
}
