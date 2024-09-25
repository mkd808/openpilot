#include "selfdrive/frogpilot/navigation/ui/navigation_settings.h"
#include "selfdrive/frogpilot/ui/qt/offroad/advanced_settings.h"
#include "selfdrive/frogpilot/ui/qt/offroad/data_settings.h"
#include "selfdrive/frogpilot/ui/qt/offroad/device_settings.h"
#include "selfdrive/frogpilot/ui/qt/offroad/frogpilot_settings.h"
#include "selfdrive/frogpilot/ui/qt/offroad/lateral_settings.h"
#include "selfdrive/frogpilot/ui/qt/offroad/longitudinal_settings.h"
#include "selfdrive/frogpilot/ui/qt/offroad/sounds_settings.h"
#include "selfdrive/frogpilot/ui/qt/offroad/utilities.h"
#include "selfdrive/frogpilot/ui/qt/offroad/vehicle_settings.h"
#include "selfdrive/frogpilot/ui/qt/offroad/visual_settings.h"

FrogPilotSettingsWindow::FrogPilotSettingsWindow(SettingsWindow *parent) : QFrame(parent) {
  mainLayout = new QStackedLayout(this);

  frogpilotSettingsWidget = new QWidget(this);
  QVBoxLayout *frogpilotSettingsLayout = new QVBoxLayout(frogpilotSettingsWidget);
  frogpilotSettingsLayout->setContentsMargins(50, 25, 50, 25);

  FrogPilotListWidget *list = new FrogPilotListWidget(frogpilotSettingsWidget);

  FrogPilotAdvancedPanel *frogpilotAdvancedPanel = new FrogPilotAdvancedPanel(this);
  QObject::connect(frogpilotAdvancedPanel, &FrogPilotAdvancedPanel::openParentToggle, this, &FrogPilotSettingsWindow::openParentToggle);
  QObject::connect(frogpilotAdvancedPanel, &FrogPilotAdvancedPanel::openSubParentToggle, this, &FrogPilotSettingsWindow::openSubParentToggle);
  QObject::connect(frogpilotAdvancedPanel, &FrogPilotAdvancedPanel::openSubSubParentToggle, this, &FrogPilotSettingsWindow::openSubSubParentToggle);

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
    {tr("Advanced Settings"), {frogpilotAdvancedPanel}},
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
    {tr("MANAGE")},
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
