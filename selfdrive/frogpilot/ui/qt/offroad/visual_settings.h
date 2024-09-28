#pragma once

#include <set>

#include "selfdrive/frogpilot/ui/qt/offroad/frogpilot_settings.h"

class FrogPilotVisualsPanel : public FrogPilotListWidget {
  Q_OBJECT

public:
  explicit FrogPilotVisualsPanel(FrogPilotSettingsWindow *parent);

signals:
  void openParentToggle();
  void openSubParentToggle();

private:
  FrogPilotSettingsWindow *parent;

  void hideSubToggles();
  void hideToggles();
  void showEvent(QShowEvent *event) override;
  void showToggles(const std::set<QString> &keys);
  void updateCarToggles();
  void updateState(const UIState &s);

  FrogPilotButtonsControl *manageCustomColorsBtn;
  FrogPilotButtonsControl *manageCustomIconsBtn;
  FrogPilotButtonsControl *manageCustomSignalsBtn;
  FrogPilotButtonsControl *manageCustomSoundsBtn;
  FrogPilotButtonsControl *manageWheelIconsBtn;

  LabelControl *downloadStatusLabel;

  std::set<QString> bonusContentKeys = {"GoatScream", "HolidayThemes", "PersonalizeOpenpilot", "RandomEvents"};
  std::set<QString> customOnroadUIKeys = {"Compass", "CustomPaths", "DynamicPathWidth", "PedalsOnUI", "RoadNameUI", "RotatingWheel"};
  std::set<QString> personalizeOpenpilotKeys = {"CustomColors", "CustomIcons", "CustomSignals", "CustomSounds", "DownloadStatusLabel", "StartupAlert", "WheelIcon"};
  std::set<QString> qolKeys = {"BigMap", "DriverCamera", "HideUIElements", "MapStyle", "StandbyMode", "StoppedTimer"};

  std::map<QString, AbstractControl*> toggles;

  Params params;
  Params paramsMemory{"/dev/shm/params"};

  bool cancellingDownload;
  bool colorDownloading;
  bool colorsDownloaded;
  bool disableOpenpilotLongitudinal;
  bool hasAutoTune;
  bool hasOpenpilotLongitudinal;
  bool iconDownloading;
  bool iconsDownloaded;
  bool personalizeOpenpilotOpen;
  bool signalDownloading;
  bool signalsDownloaded;
  bool soundDownloading;
  bool soundsDownloaded;
  bool themeDeleting;
  bool themeDownloading;
  bool wheelDownloading;
  bool wheelsDownloaded;
};
