#include "selfdrive/frogpilot/ui/qt/offroad/advanced_settings.h"

FrogPilotAdvancedPanel::FrogPilotAdvancedPanel(FrogPilotSettingsWindow *parent) : FrogPilotListWidget(parent) {
  const std::vector<std::tuple<QString, QString, QString, QString>> advancedToggles {
    {"AdvancedLateralTune", tr("Advanced Lateral Tuning"), tr("Advanced settings that control how openpilot manages steering."), "../frogpilot/assets/toggle_icons/icon_lateral_tune.png"},
    {"SteerFriction", steerFrictionStock != 0 ? QString(tr("Friction (Default: %1)")).arg(QString::number(steerFrictionStock, 'f', 2)) : tr("Friction"), tr("Adjust the resistance in the steering system. Higher values provide more stable steering but can make it feel heavy, while lower values allow lighter steering but may feel too sensitive."), ""},
    {"SteerKP", steerKPStock != 0 ? QString(tr("Kp Factor (Default: %1)")).arg(QString::number(steerKPStock, 'f', 2)) : tr("Kp Factor"), tr("Control how aggressively the car corrects its steering. Higher values offer quicker corrections but may feel jerky, while lower values make steering smoother but slower to respond."), ""},
    {"SteerLatAccel", steerLatAccelStock != 0 ? QString(tr("Lateral Accel (Default: %1)")).arg(QString::number(steerLatAccelStock, 'f', 2)) : tr("Lateral Accel"), tr("Adjust how fast the car can steer from side to side. Higher values allow quicker lane changes but can feel unstable, while lower values provide smoother steering but may feel sluggish."), ""},
    {"SteerRatio", steerRatioStock != 0 ? QString(tr("Steer Ratio (Default: %1)")).arg(QString::number(steerRatioStock, 'f', 2)) : tr("Steer Ratio"), tr("Control how much the steering wheel must turn to make the car change direction. Higher values make steering less sensitive but more stable, while lower values increase sensitivity, making the car respond more quickly to smaller steering inputs."), ""},

    {"AdvancedLongitudinalTune", tr("Advanced Longitudinal Tuning"), tr("Advanced settings that control how openpilot manages speed and acceleration."), "../frogpilot/assets/toggle_icons/icon_longitudinal_tune.png"},
    {"LeadDetectionThreshold", tr("Lead Detection Threshold"), tr("How sensitive openpilot is to detecting vehicles ahead. A lower value can help detect vehicles sooner and from farther away, but may occasionally mistake other objects for vehicles."), ""},
    {"MaxDesiredAcceleration", tr("Maximum Acceleration Rate"), tr("Set a cap on how fast openpilot can accelerate to prevent high acceleration at low speeds."), ""},

    {"CustomPersonalities", tr("Customize Driving Personalities"), tr("Customize the driving personality profiles."), "../frogpilot/assets/toggle_icons/icon_personality.png"},
    {"TrafficPersonalityProfile", tr("Traffic Personality"), tr("Customize the 'Traffic' personality profile."), "../frogpilot/assets/stock_theme/distance_icons/traffic.png"},
    {"TrafficFollow", tr("Following Distance"), tr("Set the minimum following distance in 'Traffic Mode'. The distance adjusts dynamically between this value and the 'Aggressive' profile's distance based on your speed."), ""},
    {"TrafficJerkAcceleration", tr("Acceleration Change Sensitivity"), tr("Controls the penalty applied for changes in acceleration while in 'Traffic Mode'. Higher values make acceleration and deceleration smoother but slower, while lower values allow quicker changes but may feel jerky."), ""},
    {"TrafficJerkDanger", tr("Hazard Sensitivity"), tr("Adjusts the penalty for getting too close to other vehicles or obstacles while in 'Traffic Mode'. Higher values make openpilot more cautious, maintaining a safer distance and prioritizing safety, while lower values reduce the likelihood of sudden braking but increase the risk of following too closely."), ""},
    {"TrafficJerkSpeed", tr("Speed Control Smoothness"), tr("Controls the penalty on the rate of change of acceleration while in 'Traffic Mode'. Higher values result in smoother but slower speed changes, while lower values make speed adjustments quicker but potentially more abrupt."), ""},
    {"ResetTrafficPersonality", tr("Reset Settings"), tr("Restore the 'Traffic Mode' settings to their default values."), ""},

    {"AggressivePersonalityProfile", tr("Aggressive Personality"), tr("Customize the 'Aggressive' personality profile."), "../frogpilot/assets/stock_theme/distance_icons/aggressive.png"},
    {"AggressiveFollow", tr("Following Distance"), tr("Set the following distance for 'Aggressive' mode. This represents how many seconds you follow behind the car ahead.\n\nDefault: 1.25 seconds."), ""},
    {"AggressiveJerkAcceleration", tr("Acceleration Change Sensitivity"), tr("Controls the penalty applied for changes in acceleration while using the 'Aggressive' profile personality. Higher values make acceleration and deceleration smoother but slower, while lower values allow quicker changes but may feel jerky.\n\nDefault: 0.5."), ""},
    {"AggressiveJerkDanger", tr("Hazard Sensitivity"), tr("Adjusts the penalty for getting too close to other vehicles or obstacles while using the 'Aggressive' personality profile. Higher values make openpilot more cautious, maintaining a safer distance and prioritizing safety, while lower values reduce the likelihood of sudden braking but increase the risk of following too closely.\n\nDefault: 1.0."), ""},
    {"AggressiveJerkSpeed", tr("Speed Control Smoothness"), tr("Controls the penalty on the rate of change of acceleration while using the 'Standard' personality profile. Higher values result in smoother but slower speed changes, while lower values make speed adjustments quicker but potentially more abrupt.\n\nDefault: 0.5."), ""},
    {"ResetAggressivePersonality", tr("Reset Settings"), tr("Restore the 'Aggressive' settings to their default values."), ""},

    {"StandardPersonalityProfile", tr("Standard Personality"), tr("Customize the 'Standard' personality profile."), "../frogpilot/assets/stock_theme/distance_icons/standard.png"},
    {"StandardFollow", tr("Following Distance"), tr("Set the following distance for 'Standard' mode. This represents how many seconds you follow behind the car ahead.\n\nDefault: 1.45 seconds."), ""},
    {"StandardJerkAcceleration", tr("Acceleration Change Sensitivity"), tr("Controls the penalty applied for changes in acceleration while using the 'Standard' profile personality. Higher values make acceleration and deceleration smoother but slower, while lower values allow quicker changes but may feel jerky.\n\nDefault: 1.0."), ""},
    {"StandardJerkDanger", tr("Hazard Sensitivity"), tr("Adjusts the penalty for getting too close to other vehicles or obstacles while using the 'Standard' personality profile. Higher values make openpilot more cautious, maintaining a safer distance and prioritizing safety, while lower values reduce the likelihood of sudden braking but increase the risk of following too closely.\n\nDefault: 1.0."), ""},
    {"StandardJerkSpeed", tr("Speed Control Smoothness"), tr("Controls the penalty on the rate of change of acceleration while using the 'Standard' personality profile. Higher values result in smoother but slower speed changes, while lower values make speed adjustments quicker but potentially more abrupt.\n\nDefault: 1.0."), ""},
    {"ResetStandardPersonality", tr("Reset Settings"), tr("Restore the 'Standard' settings to their default values."), ""},

    {"RelaxedPersonalityProfile", tr("Relaxed Personality"), tr("Customize the 'Relaxed' personality profile."), "../frogpilot/assets/stock_theme/distance_icons/relaxed.png"},
    {"RelaxedFollow", tr("Following Distance"), tr("Set the following distance for 'Relaxed' mode. This represents how many seconds you follow behind the car ahead.\n\nDefault: 1.75 seconds."), ""},
    {"RelaxedJerkAcceleration", tr("Acceleration Change Sensitivity"), tr("Controls the penalty applied for changes in acceleration while using the 'Relaxed' profile personality. Higher values make acceleration and deceleration smoother but slower, while lower values allow quicker changes but may feel jerky.\n\nDefault: 1.0."), ""},
    {"RelaxedJerkDanger", tr("Hazard Sensitivity"), tr("Adjusts the penalty for getting too close to other vehicles or obstacles while using the 'Relaxed' personality profile. Higher values make openpilot more cautious, maintaining a safer distance and prioritizing safety, while lower values reduce the likelihood of sudden braking but increase the risk of following too closely.\n\nDefault: 1.0."), ""},
    {"RelaxedJerkSpeed", tr("Speed Control Smoothness"), tr("Controls the penalty on the rate of change of acceleration while using the 'Relaxed' personality profile. Higher values result in smoother but slower speed changes, while lower values make speed adjustments quicker but potentially more abrupt.\n\nDefault: 1.0."), ""},
    {"ResetRelaxedPersonality", tr("Reset Settings"), tr("Restore the 'Relaxed' settings to their default values."), ""},

    {"DeveloperUI", tr("Developer UI"), tr("Show detailed information about openpilot's internal operations."), "../frogpilot/assets/toggle_icons/icon_device.png"},
    {"BorderMetrics", tr("Border Metrics"), tr("Display performance metrics around the edge of the screen while driving."), ""},
    {"FPSCounter", tr("FPS Counter"), tr("Show the 'Frames Per Second' (FPS) at the bottom of the screen while driving."), ""},
    {"LateralMetrics", tr("Lateral Metrics"), tr("Display metrics related to steering control at the top of the screen while driving."), ""},
    {"LongitudinalMetrics", tr("Longitudinal Metrics"), tr("Display metrics related to acceleration, speed, and desired following distance at the top of the screen while driving."), ""},
    {"NumericalTemp", tr("Numerical Temperature Gauge"), tr("Show exact temperature readings instead of general status labels like 'GOOD', 'OK', or 'HIGH' in the sidebar."), ""},
    {"SidebarMetrics", tr("Sidebar"), tr("Display system information like CPU, GPU, RAM usage, IP address, and storage space in the sidebar."), ""},
    {"UseSI", tr("Use International System of Units"), tr("Display measurements using the 'International System of Units' (SI)."), ""},

    {"ModelManagement", tr("Model Management"), tr("Manage the driving models used by openpilot."), "../frogpilot/assets/toggle_icons/icon_model.png"},
    {"AutomaticallyUpdateModels", tr("Automatically Update and Download Models"), tr("Automatically download new or updated driving models."), ""},
    {"ModelRandomizer", tr("Model Randomizer"), tr("A random model is selected and can be reviewed at the end of each drive to help find your preferred model."), ""},
    {"ManageBlacklistedModels", tr("Manage Model Blacklist"), tr("Control which models are blacklisted and won't be used for future drives."), ""},
    {"ResetScores", tr("Reset Model Scores"), tr("Clear the ratings you've given to the driving models."), ""},
    {"ReviewScores", tr("Review Model Scores"), tr("View the ratings you've assigned to the driving models."), ""},
    {"DeleteModel", tr("Delete Model"), tr("Remove the selected driving model from your device."), ""},
    {"DownloadModel", tr("Download Model"), tr("Download the selected driving model."), ""},
    {"DownloadAllModels", tr("Download All Models"), tr("Download all undownloaded driving models."), ""},
    {"SelectModel", tr("Select Model"), tr("Select which model openpilot uses to drive."), ""},
    {"ResetCalibrations", tr("Reset Model Calibrations"), tr("Reset calibration settings for the driving models."), ""},

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
    AbstractControl *advancedToggle;

    if (param == "AdvancedLateralTune") {
      FrogPilotParamManageControl *lateralTuneToggle = new FrogPilotParamManageControl(param, title, desc, icon);
      QObject::connect(lateralTuneToggle, &FrogPilotParamManageControl::manageButtonClicked, [this]() {
        std::set<QString> modifiedLateralTuneKeys = lateralTuneKeys;

        if (!liveValid || params.getBool("NNFF")) {
          modifiedLateralTuneKeys.erase("SteerFriction");
          modifiedLateralTuneKeys.erase("SteerLatAccel");
        }

        showToggles(modifiedLateralTuneKeys);
      });
      advancedToggle = lateralTuneToggle;
    } else if (param == "SteerFriction") {
      std::vector<QString> steerFrictionToggles{"ResetSteerFriction"};
      std::vector<QString> steerFrictionToggleNames{"Reset"};
      advancedToggle = new FrogPilotParamValueButtonControl(param, title, desc, icon, 0, 0.25, QString(), std::map<int, QString>(), 0.01, steerFrictionToggles, steerFrictionToggleNames, false);
    } else if (param == "SteerKP") {
      std::vector<QString> steerKPToggles{"ResetSteerKP"};
      std::vector<QString> steerKPToggleNames{"Reset"};
      advancedToggle = new FrogPilotParamValueButtonControl(param, title, desc, icon, steerKPStock * 0.50, steerKPStock * 1.50, QString(), std::map<int, QString>(), 0.01, steerKPToggles, steerKPToggleNames, false);
    } else if (param == "SteerLatAccel") {
      std::vector<QString> steerLatAccelToggles{"ResetSteerLatAccel"};
      std::vector<QString> steerLatAccelToggleNames{"Reset"};
      advancedToggle = new FrogPilotParamValueButtonControl(param, title, desc, icon, steerLatAccelStock * 0.25, steerLatAccelStock * 1.25, QString(), std::map<int, QString>(), 0.01, steerLatAccelToggles, steerLatAccelToggleNames, false);
    } else if (param == "SteerRatio") {
      std::vector<QString> steerRatioToggles{"ResetSteerRatio"};
      std::vector<QString> steerRatioToggleNames{"Reset"};
      advancedToggle = new FrogPilotParamValueButtonControl(param, title, desc, icon, steerRatioStock * 0.75, steerRatioStock * 1.25, QString(), std::map<int, QString>(), 0.01, steerRatioToggles, steerRatioToggleNames, false);

    } else if (param == "AdvancedLongitudinalTune") {
      FrogPilotParamManageControl *longitudinalTuneToggle = new FrogPilotParamManageControl(param, title, desc, icon);
      QObject::connect(longitudinalTuneToggle, &FrogPilotParamManageControl::manageButtonClicked, [this]() {
        std::set<QString> modifiedLongitudinalTuneKeys = longitudinalTuneKeys;

        bool radarlessModel = QString::fromStdString(params.get("RadarlessModels")).split(",").contains(QString::fromStdString(params.get("Model")));
        if (radarlessModel) {
          modifiedLongitudinalTuneKeys.erase("LeadDetectionThreshold");
        }

        showToggles(modifiedLongitudinalTuneKeys);
      });
      advancedToggle = longitudinalTuneToggle;
    } else if (param == "LeadDetectionThreshold") {
      advancedToggle = new FrogPilotParamValueControl(param, title, desc, icon, 1, 99, "%");
    } else if (param == "MaxDesiredAcceleration") {
      advancedToggle = new FrogPilotParamValueButtonControl(param, title, desc, icon, 0.1, 4.0, "m/s", std::map<int, QString>(), 0.1);

    } else if (param == "CustomPersonalities") {
      FrogPilotParamManageControl *customPersonalitiesToggle = new FrogPilotParamManageControl(param, title, desc, icon);
      QObject::connect(customPersonalitiesToggle, &FrogPilotParamManageControl::manageButtonClicked, [this]() {
        showToggles(customDrivingPersonalityKeys);
      });
      advancedToggle = customPersonalitiesToggle;
    } else if (param == "ResetTrafficPersonality" || param == "ResetAggressivePersonality" || param == "ResetStandardPersonality" || param == "ResetRelaxedPersonality") {
      FrogPilotButtonsControl *profileBtn = new FrogPilotButtonsControl(title, desc, {tr("Reset")});
      advancedToggle = profileBtn;
    } else if (param == "TrafficPersonalityProfile") {
      FrogPilotParamManageControl *trafficPersonalityToggle = new FrogPilotParamManageControl(param, title, desc, icon);
      QObject::connect(trafficPersonalityToggle, &FrogPilotParamManageControl::manageButtonClicked, [this]() {
        customPersonalityOpen = true;
        openSubParentToggle();
        showToggles(trafficPersonalityKeys);
      });
      advancedToggle = trafficPersonalityToggle;
    } else if (param == "AggressivePersonalityProfile") {
      FrogPilotParamManageControl *aggressivePersonalityToggle = new FrogPilotParamManageControl(param, title, desc, icon);
      QObject::connect(aggressivePersonalityToggle, &FrogPilotParamManageControl::manageButtonClicked, [this]() {
        customPersonalityOpen = true;
        openSubParentToggle();
        showToggles(aggressivePersonalityKeys);
      });
      advancedToggle = aggressivePersonalityToggle;
    } else if (param == "StandardPersonalityProfile") {
      FrogPilotParamManageControl *standardPersonalityToggle = new FrogPilotParamManageControl(param, title, desc, icon);
      QObject::connect(standardPersonalityToggle, &FrogPilotParamManageControl::manageButtonClicked, [this]() {
        customPersonalityOpen = true;
        openSubParentToggle();
        showToggles(standardPersonalityKeys);
      });
      advancedToggle = standardPersonalityToggle;
    } else if (param == "RelaxedPersonalityProfile") {
      FrogPilotParamManageControl *relaxedPersonalityToggle = new FrogPilotParamManageControl(param, title, desc, icon);
      QObject::connect(relaxedPersonalityToggle, &FrogPilotParamManageControl::manageButtonClicked, [this]() {
        customPersonalityOpen = true;
        openSubParentToggle();
        showToggles(relaxedPersonalityKeys);
      });
      advancedToggle = relaxedPersonalityToggle;
    } else if (trafficPersonalityKeys.find(param) != trafficPersonalityKeys.end() ||
               aggressivePersonalityKeys.find(param) != aggressivePersonalityKeys.end() ||
               standardPersonalityKeys.find(param) != standardPersonalityKeys.end() ||
               relaxedPersonalityKeys.find(param) != relaxedPersonalityKeys.end()) {
      if (param == "TrafficFollow" || param == "AggressiveFollow" || param == "StandardFollow" || param == "RelaxedFollow") {
        if (param == "TrafficFollow") {
          advancedToggle = new FrogPilotParamValueControl(param, title, desc, icon, 0.5, 5, tr(" seconds"), std::map<int, QString>(), 0.01);
        } else {
          advancedToggle = new FrogPilotParamValueControl(param, title, desc, icon, 1, 5, tr(" seconds"), std::map<int, QString>(), 0.01);
        }
      } else {
        advancedToggle = new FrogPilotParamValueControl(param, title, desc, icon, 1, 500, "%");
      }

    } else if (param == "DeveloperUI") {
      FrogPilotParamManageControl *developerUIToggle = new FrogPilotParamManageControl(param, title, desc, icon);
      QObject::connect(developerUIToggle, &FrogPilotParamManageControl::manageButtonClicked, [this]() {
        std::set<QString> modifiedDeveloperUIKeys = developerUIKeys;

        if (!hasAutoTune) {
          modifiedDeveloperUIKeys.erase("LateralMetrics");
        }

        if (disableOpenpilotLongitudinal || !hasOpenpilotLongitudinal) {
          modifiedDeveloperUIKeys.erase("LongitudinalMetrics");
        }

        showToggles(modifiedDeveloperUIKeys);
      });
      advancedToggle = developerUIToggle;
    } else if (param == "BorderMetrics") {
      std::vector<QString> borderToggles{"BlindSpotMetrics", "ShowSteering", "SignalMetrics"};
      std::vector<QString> borderToggleNames{tr("Blind Spot"), tr("Steering Torque"), tr("Turn Signal")};
      advancedToggle = new FrogPilotButtonToggleControl(param, title, desc, borderToggles, borderToggleNames);
    } else if (param == "LateralMetrics") {
      std::vector<QString> lateralToggles{"TuningInfo"};
      std::vector<QString> lateralToggleNames{tr("Auto Tune")};
      advancedToggle = new FrogPilotButtonToggleControl(param, title, desc, lateralToggles, lateralToggleNames);
    } else if (param == "LongitudinalMetrics") {
      std::vector<QString> longitudinalToggles{"LeadInfo", "JerkInfo"};
      std::vector<QString> longitudinalToggleNames{tr("Lead Info"), tr("Longitudinal Jerk")};
      advancedToggle = new FrogPilotButtonToggleControl(param, title, desc, longitudinalToggles, longitudinalToggleNames);
    } else if (param == "NumericalTemp") {
      std::vector<QString> temperatureToggles{"Fahrenheit"};
      std::vector<QString> temperatureToggleNames{tr("Fahrenheit")};
      advancedToggle = new FrogPilotButtonToggleControl(param, title, desc, temperatureToggles, temperatureToggleNames);
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
      advancedToggle = sidebarMetricsToggle;

    } else if (param == "ModelManagement") {
      FrogPilotParamManageControl *modelManagementToggle = new FrogPilotParamManageControl(param, title, desc, icon);
      QObject::connect(modelManagementToggle, &FrogPilotParamManageControl::manageButtonClicked, [this]() {
        availableModelNames = QString::fromStdString(params.get("AvailableModelsNames")).split(",");
        availableModels = QString::fromStdString(params.get("AvailableModels")).split(",");
        experimentalModels = QString::fromStdString(params.get("ExperimentalModels")).split(",");

        modelManagementOpen = true;

        QString currentModel = QString::fromStdString(params.get("Model")) + ".thneed";
        QStringList modelFiles = modelDir.entryList({"*.thneed"}, QDir::Files);
        modelFiles.removeAll(currentModel);
        haveModelsDownloaded = modelFiles.size() > 1;
        modelsDownloaded = params.getBool("ModelsDownloaded");

        showToggles(modelManagementKeys);
      });
      advancedToggle = modelManagementToggle;
    } else if (param == "ModelRandomizer") {
      FrogPilotParamManageControl *modelRandomizerToggle = new FrogPilotParamManageControl(param, title, desc, icon);
      QObject::connect(modelRandomizerToggle, &FrogPilotParamManageControl::manageButtonClicked, [this]() {
        openSubParentToggle();
        showToggles(modelRandomizerKeys);
      });
      advancedToggle = modelRandomizerToggle;
    } else if (param == "ManageBlacklistedModels") {
      FrogPilotButtonsControl *blacklistBtn = new FrogPilotButtonsControl(title, desc, {tr("ADD"), tr("REMOVE")});
      QObject::connect(blacklistBtn, &FrogPilotButtonsControl::buttonClicked, [=](int id) {
        QStringList blacklistedModels = QString::fromStdString(params.get("BlacklistedModels")).split(",", QString::SkipEmptyParts);
        QMap<QString, QString> labelToModelMap;
        QStringList selectableModels, deletableModels;

        for (int i = 0; i < availableModels.size(); i++) {
          QString model = availableModels[i];
          if (blacklistedModels.contains(model)) {
            deletableModels.append(availableModelNames[i]);
          } else {
            selectableModels.append(availableModelNames[i]);
          }
          labelToModelMap[availableModelNames[i]] = model;
        }

        if (id == 0) {
          if (selectableModels.size() == 1) {
            FrogPilotConfirmationDialog::toggleAlert(tr("There's no more models to blacklist! The only available model is \"%1\"!").arg(selectableModels.first()), tr("OK"), this);
          } else {
            QString selectedModel = MultiOptionDialog::getSelection(tr("Select a model to add to the blacklist"), selectableModels, "", this);
            if (!selectedModel.isEmpty()) {
              if (ConfirmationDialog::confirm(tr("Are you sure you want to add the '%1' model to the blacklist?").arg(selectedModel), tr("Add"), this)) {
                QString modelToAdd = labelToModelMap[selectedModel];
                if (!blacklistedModels.contains(modelToAdd)) {
                  blacklistedModels.append(modelToAdd);
                  params.putNonBlocking("BlacklistedModels", blacklistedModels.join(",").toStdString());
                }
              }
            }
          }
        } else if (id == 1) {
          QString selectedModel = MultiOptionDialog::getSelection(tr("Select a model to remove from the blacklist"), deletableModels, "", this);
          if (!selectedModel.isEmpty()) {
            if (ConfirmationDialog::confirm(tr("Are you sure you want to remove the '%1' model from the blacklist?").arg(selectedModel), tr("Remove"), this)) {
              QString modelToRemove = labelToModelMap[selectedModel];
              if (blacklistedModels.contains(modelToRemove)) {
                blacklistedModels.removeAll(modelToRemove);
                params.putNonBlocking("BlacklistedModels", blacklistedModels.join(",").toStdString());
                paramsStorage.put("BlacklistedModels", blacklistedModels.join(",").toStdString());
              }
            }
          }
        }
      });
      advancedToggle = blacklistBtn;
    } else if (param == "ResetScores") {
      ButtonControl *resetCalibrationsBtn = new ButtonControl(title, tr("RESET"), desc);
      QObject::connect(resetCalibrationsBtn, &ButtonControl::clicked, [this]() {
        if (FrogPilotConfirmationDialog::yesorno(tr("Reset all model scores?"), this)) {
          for (const QString &model : availableModelNames) {
            QString cleanedModel = processModelName(model);
            params.remove(QString("%1Drives").arg(cleanedModel).toStdString());
            paramsStorage.remove(QString("%1Drives").arg(cleanedModel).toStdString());
            params.remove(QString("%1Score").arg(cleanedModel).toStdString());
            paramsStorage.remove(QString("%1Score").arg(cleanedModel).toStdString());
          }
          updateModelLabels();
        }
      });
      advancedToggle = reinterpret_cast<AbstractControl*>(resetCalibrationsBtn);
    } else if (param == "ReviewScores") {
      ButtonControl *reviewScoresBtn = new ButtonControl(title, tr("VIEW"), desc);
      QObject::connect(reviewScoresBtn, &ButtonControl::clicked, [this]() {
        openSubSubParentToggle();

        for (LabelControl *label : labelControls) {
          label->setVisible(true);
        }

        for (auto &[key, toggle] : toggles) {
          toggle->setVisible(false);
        }
      });
      advancedToggle = reinterpret_cast<AbstractControl*>(reviewScoresBtn);
    } else if (param == "DeleteModel") {
      deleteModelBtn = new ButtonControl(title, tr("DELETE"), desc);
      QObject::connect(deleteModelBtn, &ButtonControl::clicked, [this]() {
        QStringList deletableModels, existingModels = modelDir.entryList({"*.thneed"}, QDir::Files);
        QMap<QString, QString> labelToFileMap;
        QString currentModel = QString::fromStdString(params.get("Model")) + ".thneed";

        for (int i = 0; i < availableModels.size(); i++) {
          QString modelFile = availableModels[i] + ".thneed";
          if (existingModels.contains(modelFile) && modelFile != currentModel && !availableModelNames[i].contains("(Default)")) {
            deletableModels.append(availableModelNames[i]);
            labelToFileMap[availableModelNames[i]] = modelFile;
          }
        }

        QString selectedModel = MultiOptionDialog::getSelection(tr("Select a model to delete"), deletableModels, "", this);
        if (!selectedModel.isEmpty()) {
          if (ConfirmationDialog::confirm(tr("Are you sure you want to delete the '%1' model?").arg(selectedModel), tr("Delete"), this)) {
            std::thread([=]() {
              modelDeleting = true;
              modelsDownloaded = false;
              update();

              params.putBoolNonBlocking("ModelsDownloaded", false);
              deleteModelBtn->setValue(tr("Deleting..."));

              QFile::remove(modelDir.absoluteFilePath(labelToFileMap[selectedModel]));
              deleteModelBtn->setValue(tr("Deleted!"));

              util::sleep_for(1000);
              deleteModelBtn->setValue("");
              modelDeleting = false;

              QStringList modelFiles = modelDir.entryList({"*.thneed"}, QDir::Files);
              modelFiles.removeAll(currentModel);

              haveModelsDownloaded = modelFiles.size() > 1;
              update();
            }).detach();
          }
        }
      });
      advancedToggle = reinterpret_cast<AbstractControl*>(deleteModelBtn);
    } else if (param == "DownloadModel") {
      downloadModelBtn = new ButtonControl(title, tr("DOWNLOAD"), desc);
      QObject::connect(downloadModelBtn, &ButtonControl::clicked, [this]() {
        if (downloadModelBtn->text() == tr("CANCEL")) {
          paramsMemory.remove("ModelToDownload");
          paramsMemory.putBool("CancelModelDownload", true);
          cancellingDownload = true;

          device()->resetInteractiveTimeout(30);
        } else {
          QMap<QString, QString> labelToModelMap;
          QStringList existingModels = modelDir.entryList({"*.thneed"}, QDir::Files);
          QStringList downloadableModels;

          for (int i = 0; i < availableModels.size(); i++) {
            QString modelFile = availableModels[i] + ".thneed";
            if (!existingModels.contains(modelFile) && !availableModelNames[i].contains("(Default)")) {
              downloadableModels.append(availableModelNames[i]);
              labelToModelMap.insert(availableModelNames[i], availableModels[i]);
            }
          }

          QString modelToDownload = MultiOptionDialog::getSelection(tr("Select a driving model to download"), downloadableModels, "", this);
          if (!modelToDownload.isEmpty()) {
            device()->resetInteractiveTimeout(300);

            modelDownloading = true;
            paramsMemory.put("ModelToDownload", labelToModelMap.value(modelToDownload).toStdString());
            paramsMemory.put("ModelDownloadProgress", "0%");

            downloadModelBtn->setValue(tr("Downloading %1...").arg(modelToDownload.remove(QRegularExpression("[🗺️👀📡]")).trimmed()));

            QTimer *progressTimer = new QTimer(this);
            progressTimer->setInterval(100);

            QObject::connect(progressTimer, &QTimer::timeout, this, [=]() {
              QString progress = QString::fromStdString(paramsMemory.get("ModelDownloadProgress"));
              bool downloadComplete = progress.contains(QRegularExpression("downloaded", QRegularExpression::CaseInsensitiveOption));
              bool downloadFailed = progress.contains(QRegularExpression("cancelled|exists|failed|offline", QRegularExpression::CaseInsensitiveOption));

              if (!progress.isEmpty() && progress != "0%") {
                downloadModelBtn->setValue(progress);
              }

              if (downloadComplete || downloadFailed) {
                bool lastModelDownloaded = downloadComplete;

                if (downloadComplete) {
                  haveModelsDownloaded = true;
                  update();
                }

                if (downloadComplete) {
                  for (const QString &model : availableModels) {
                    if (!QFile::exists(modelDir.filePath(model + ".thneed"))) {
                      lastModelDownloaded = false;
                      break;
                    }
                  }
                }

                downloadModelBtn->setValue(progress);

                paramsMemory.remove("CancelModelDownload");
                paramsMemory.remove("ModelDownloadProgress");

                progressTimer->stop();
                progressTimer->deleteLater();

                QTimer::singleShot(2000, this, [=]() {
                  cancellingDownload = false;
                  modelDownloading = false;

                  downloadModelBtn->setValue("");

                  if (lastModelDownloaded) {
                    modelsDownloaded = true;
                    update();

                    params.putBoolNonBlocking("ModelsDownloaded", modelsDownloaded);
                  }

                  device()->resetInteractiveTimeout(30);
                });
              }
            });
            progressTimer->start();
          }
        }
      });
      advancedToggle = reinterpret_cast<AbstractControl*>(downloadModelBtn);
    } else if (param == "DownloadAllModels") {
      downloadAllModelsBtn = new ButtonControl(title, tr("DOWNLOAD"), desc);
      QObject::connect(downloadAllModelsBtn, &ButtonControl::clicked, [this]() {
        if (downloadAllModelsBtn->text() == tr("CANCEL")) {
          paramsMemory.remove("DownloadAllModels");
          paramsMemory.putBool("CancelModelDownload", true);
          cancellingDownload = true;

          device()->resetInteractiveTimeout(30);
        } else {
          device()->resetInteractiveTimeout(300);

          startDownloadAllModels();
        }
      });
      advancedToggle = reinterpret_cast<AbstractControl*>(downloadAllModelsBtn);
    } else if (param == "SelectModel") {
      selectModelBtn = new ButtonControl(title, tr("SELECT"), desc);
      QObject::connect(selectModelBtn, &ButtonControl::clicked, [this]() {
        QSet<QString> modelFilesBaseNames = QSet<QString>::fromList(modelDir.entryList({"*.thneed"}, QDir::Files).replaceInStrings(QRegExp("\\.thneed$"), ""));
        QStringList selectableModels;

        for (int i = 0; i < availableModels.size(); i++) {
          if (modelFilesBaseNames.contains(availableModels[i]) || availableModelNames[i].contains("(Default)")) {
            selectableModels.append(availableModelNames[i]);
          }
        }

        QString modelToSelect = MultiOptionDialog::getSelection(tr("Select a model - 🗺️ = Navigation | 📡 = Radar | 👀 = VOACC"), selectableModels, "", this);
        if (!modelToSelect.isEmpty()) {
          selectModelBtn->setValue(modelToSelect);
          int modelIndex = availableModelNames.indexOf(modelToSelect);

          params.putNonBlocking("Model", availableModels.at(modelIndex).toStdString());
          params.putNonBlocking("ModelName", modelToSelect.toStdString());

          if (experimentalModels.contains(availableModels.at(modelIndex))) {
            FrogPilotConfirmationDialog::toggleAlert(tr("WARNING: This is a very experimental model and may drive dangerously!"), tr("I understand the risks."), this);
          }

          QString model = availableModelNames.at(modelIndex);
          QString part_model_param = processModelName(model);

          if (!params.checkKey(part_model_param.toStdString() + "CalibrationParams") || !params.checkKey(part_model_param.toStdString() + "LiveTorqueParameters")) {
            if (FrogPilotConfirmationDialog::yesorno(tr("Start with a fresh calibration for the newly selected model?"), this)) {
              params.remove("CalibrationParams");
              params.remove("LiveTorqueParameters");
            }
          }

          if (started) {
            if (FrogPilotConfirmationDialog::toggle(tr("Reboot required to take effect."), tr("Reboot Now"), this)) {
              Hardware::reboot();
            }
          }
        }
      });
      selectModelBtn->setValue(QString::fromStdString(params.get("ModelName")));
      advancedToggle = reinterpret_cast<AbstractControl*>(selectModelBtn);
    } else if (param == "ResetCalibrations") {
      FrogPilotButtonsControl *resetCalibrationsBtn = new FrogPilotButtonsControl(title, desc, {tr("RESET ALL"), tr("RESET ONE")});
      QObject::connect(resetCalibrationsBtn, &FrogPilotButtonsControl::showDescriptionEvent, this, &FrogPilotAdvancedPanel::updateCalibrationDescription);
      QObject::connect(resetCalibrationsBtn, &FrogPilotButtonsControl::buttonClicked, [=](int id) {
        if (id == 0) {
          if (FrogPilotConfirmationDialog::yesorno(tr("Are you sure you want to completely reset all of your model calibrations?"), this)) {
            for (const QString &model : availableModelNames) {
              QString cleanedModel = processModelName(model);
              params.remove(QString("%1CalibrationParams").arg(cleanedModel).toStdString());
              paramsStorage.remove(QString("%1CalibrationParams").arg(cleanedModel).toStdString());
              params.remove(QString("%1LiveTorqueParameters").arg(cleanedModel).toStdString());
              paramsStorage.remove(QString("%1LiveTorqueParameters").arg(cleanedModel).toStdString());
            }
          }
        } else if (id == 1) {
          QStringList selectableModelLabels;
          for (int i = 0; i < availableModels.size(); i++) {
            selectableModelLabels.append(availableModelNames[i]);
          }

          QString modelToReset = MultiOptionDialog::getSelection(tr("Select a model to reset"), selectableModelLabels, "", this);
          if (!modelToReset.isEmpty()) {
            if (FrogPilotConfirmationDialog::yesorno(tr("Are you sure you want to completely reset this model's calibrations?"), this)) {
              QString cleanedModel = processModelName(modelToReset);
              params.remove(QString("%1CalibrationParams").arg(cleanedModel).toStdString());
              paramsStorage.remove(QString("%1CalibrationParams").arg(cleanedModel).toStdString());
              params.remove(QString("%1LiveTorqueParameters").arg(cleanedModel).toStdString());
              paramsStorage.remove(QString("%1LiveTorqueParameters").arg(cleanedModel).toStdString());
            }
          }
        }
      });
      advancedToggle = resetCalibrationsBtn;

    } else if (param == "ModelUI") {
      FrogPilotParamManageControl *modelUIToggle = new FrogPilotParamManageControl(param, title, desc, icon);
      QObject::connect(modelUIToggle, &FrogPilotParamManageControl::manageButtonClicked, [this]() {
        std::set<QString> modifiedModelUIKeysKeys = modelUIKeys;

        if (disableOpenpilotLongitudinal || !hasOpenpilotLongitudinal) {
          modifiedModelUIKeysKeys.erase("HideLeadMarker");
        }

        showToggles(modifiedModelUIKeysKeys);
      });
      advancedToggle = modelUIToggle;
    } else if (param == "LaneLinesWidth" || param == "RoadEdgesWidth") {
      advancedToggle = new FrogPilotParamValueControl(param, title, desc, icon, 0, 24, tr(" inches"));
    } else if (param == "PathEdgeWidth") {
      advancedToggle = new FrogPilotParamValueControl(param, title, desc, icon, 0, 100, tr("%"));
    } else if (param == "PathWidth") {
      advancedToggle = new FrogPilotParamValueControl(param, title, desc, icon, 0, 10, tr(" feet"), std::map<int, QString>(), 0.1);

    } else {
      advancedToggle = new ParamControl(param, title, desc, icon);
    }

    addItem(advancedToggle);
    toggles[param.toStdString()] = advancedToggle;

    tryConnect<ToggleControl>(advancedToggle, &ToggleControl::toggleFlipped, this, updateFrogPilotToggles);
    tryConnect<FrogPilotButtonToggleControl>(advancedToggle, &FrogPilotButtonToggleControl::buttonClicked, this, updateFrogPilotToggles);
    tryConnect<FrogPilotParamManageControl>(advancedToggle, &FrogPilotParamManageControl::manageButtonClicked, this, &FrogPilotAdvancedPanel::openParentToggle);
    tryConnect<FrogPilotParamValueControl>(advancedToggle, &FrogPilotParamValueControl::valueChanged, this, updateFrogPilotToggles);

    QObject::connect(advancedToggle, &AbstractControl::showDescriptionEvent, [this]() {
      update();
    });
  }

  QObject::connect(static_cast<ToggleControl*>(toggles["ModelRandomizer"]), &ToggleControl::toggleFlipped, [this](bool state) {
    modelRandomizer = state;
    if (state && !modelsDownloaded) {
      if (FrogPilotConfirmationDialog::yesorno(tr("The 'Model Randomizer' only works with downloaded models. Do you want to download all the driving models?"), this)) {
        startDownloadAllModels();
      }
    }
  });

  steerFrictionToggle = static_cast<FrogPilotParamValueButtonControl*>(toggles["SteerFriction"]);
  QObject::connect(steerFrictionToggle, &FrogPilotParamValueButtonControl::buttonClicked, [this]() {
    params.putFloat("SteerFriction", steerFrictionStock);
    steerFrictionToggle->refresh();
    updateFrogPilotToggles();
  });

  steerKPToggle = static_cast<FrogPilotParamValueButtonControl*>(toggles["SteerKP"]);
  QObject::connect(steerKPToggle, &FrogPilotParamValueButtonControl::buttonClicked, [this]() {
    params.putFloat("SteerKP", steerKPStock);
    steerKPToggle->refresh();
    updateFrogPilotToggles();
  });

  steerLatAccelToggle = static_cast<FrogPilotParamValueButtonControl*>(toggles["SteerLatAccel"]);
  QObject::connect(steerLatAccelToggle, &FrogPilotParamValueButtonControl::buttonClicked, [this]() {
    params.putFloat("SteerLatAccel", steerLatAccelStock);
    steerLatAccelToggle->refresh();
    updateFrogPilotToggles();
  });

  steerRatioToggle = static_cast<FrogPilotParamValueButtonControl*>(toggles["SteerRatio"]);
  QObject::connect(steerRatioToggle, &FrogPilotParamValueButtonControl::buttonClicked, [this]() {
    params.putFloat("SteerRatio", steerRatioStock);
    steerRatioToggle->refresh();
    updateFrogPilotToggles();
  });

  FrogPilotParamValueControl *trafficFollowToggle = static_cast<FrogPilotParamValueControl*>(toggles["TrafficFollow"]);
  FrogPilotParamValueControl *trafficAccelerationToggle = static_cast<FrogPilotParamValueControl*>(toggles["TrafficJerkAcceleration"]);
  FrogPilotParamValueControl *trafficDangerToggle = static_cast<FrogPilotParamValueControl*>(toggles["TrafficJerkDanger"]);
  FrogPilotParamValueControl *trafficSpeedToggle = static_cast<FrogPilotParamValueControl*>(toggles["TrafficJerkSpeed"]);
  FrogPilotButtonsControl *trafficResetButton = static_cast<FrogPilotButtonsControl*>(toggles["ResetTrafficPersonality"]);
  QObject::connect(trafficResetButton, &FrogPilotButtonsControl::buttonClicked, this, [=]() {
    if (FrogPilotConfirmationDialog::yesorno(tr("Are you sure you want to completely reset your settings for the 'Traffic Mode' personality?"), this)) {
      params.putFloat("TrafficFollow", 0.5);
      params.putFloat("TrafficJerkAcceleration", 50);
      params.putFloat("TrafficJerkDanger", 100);
      params.putFloat("TrafficJerkSpeed", 50);
      trafficFollowToggle->refresh();
      trafficAccelerationToggle->refresh();
      trafficDangerToggle->refresh();
      trafficSpeedToggle->refresh();
      updateFrogPilotToggles();
    }
  });

  FrogPilotParamValueControl *aggressiveFollowToggle = static_cast<FrogPilotParamValueControl*>(toggles["AggressiveFollow"]);
  FrogPilotParamValueControl *aggressiveAccelerationToggle = static_cast<FrogPilotParamValueControl*>(toggles["AggressiveJerkAcceleration"]);
  FrogPilotParamValueControl *aggressiveDangerToggle = static_cast<FrogPilotParamValueControl*>(toggles["AggressiveJerkDanger"]);
  FrogPilotParamValueControl *aggressiveSpeedToggle = static_cast<FrogPilotParamValueControl*>(toggles["AggressiveJerkSpeed"]);
  FrogPilotButtonsControl *aggressiveResetButton = static_cast<FrogPilotButtonsControl*>(toggles["ResetAggressivePersonality"]);
  QObject::connect(aggressiveResetButton, &FrogPilotButtonsControl::buttonClicked, this, [=]() {
    if (FrogPilotConfirmationDialog::yesorno(tr("Are you sure you want to completely reset your settings for the 'Aggressive' personality?"), this)) {
      params.putFloat("AggressiveFollow", 1.25);
      params.putFloat("AggressiveJerkAcceleration", 50);
      params.putFloat("AggressiveJerkDanger", 100);
      params.putFloat("AggressiveJerkSpeed", 50);
      aggressiveFollowToggle->refresh();
      aggressiveAccelerationToggle->refresh();
      aggressiveDangerToggle->refresh();
      aggressiveSpeedToggle->refresh();
      updateFrogPilotToggles();
    }
  });

  FrogPilotParamValueControl *standardFollowToggle = static_cast<FrogPilotParamValueControl*>(toggles["StandardFollow"]);
  FrogPilotParamValueControl *standardAccelerationToggle = static_cast<FrogPilotParamValueControl*>(toggles["StandardJerkAcceleration"]);
  FrogPilotParamValueControl *standardDangerToggle = static_cast<FrogPilotParamValueControl*>(toggles["StandardJerkDanger"]);
  FrogPilotParamValueControl *standardSpeedToggle = static_cast<FrogPilotParamValueControl*>(toggles["StandardJerkSpeed"]);
  FrogPilotButtonsControl *standardResetButton = static_cast<FrogPilotButtonsControl*>(toggles["ResetStandardPersonality"]);
  QObject::connect(standardResetButton, &FrogPilotButtonsControl::buttonClicked, this, [=]() {
    if (FrogPilotConfirmationDialog::yesorno(tr("Are you sure you want to completely reset your settings for the 'Standard' personality?"), this)) {
      params.putFloat("StandardFollow", 1.45);
      params.putFloat("StandardJerkAcceleration", 100);
      params.putFloat("StandardJerkDanger", 100);
      params.putFloat("StandardJerkSpeed", 100);
      standardFollowToggle->refresh();
      standardAccelerationToggle->refresh();
      standardDangerToggle->refresh();
      standardSpeedToggle->refresh();
      updateFrogPilotToggles();
    }
  });

  FrogPilotParamValueControl *relaxedFollowToggle = static_cast<FrogPilotParamValueControl*>(toggles["RelaxedFollow"]);
  FrogPilotParamValueControl *relaxedAccelerationToggle = static_cast<FrogPilotParamValueControl*>(toggles["RelaxedJerkAcceleration"]);
  FrogPilotParamValueControl *relaxedDangerToggle = static_cast<FrogPilotParamValueControl*>(toggles["RelaxedJerkDanger"]);
  FrogPilotParamValueControl *relaxedSpeedToggle = static_cast<FrogPilotParamValueControl*>(toggles["RelaxedJerkSpeed"]);
  FrogPilotButtonsControl *relaxedResetButton = static_cast<FrogPilotButtonsControl*>(toggles["ResetRelaxedPersonality"]);
  QObject::connect(relaxedResetButton, &FrogPilotButtonsControl::buttonClicked, this, [=]() {
    if (FrogPilotConfirmationDialog::yesorno(tr("Are you sure you want to completely reset your settings for the 'Relaxed' personality?"), this)) {
      params.putFloat("RelaxedFollow", 1.75);
      params.putFloat("RelaxedJerkAcceleration", 100);
      params.putFloat("RelaxedJerkDanger", 100);
      params.putFloat("RelaxedJerkSpeed", 100);
      relaxedFollowToggle->refresh();
      relaxedAccelerationToggle->refresh();
      relaxedDangerToggle->refresh();
      relaxedSpeedToggle->refresh();
      updateFrogPilotToggles();
    }
  });

  QObject::connect(parent, &FrogPilotSettingsWindow::closeParentToggle, this, &FrogPilotAdvancedPanel::hideToggles);
  QObject::connect(parent, &FrogPilotSettingsWindow::closeSubParentToggle, this, &FrogPilotAdvancedPanel::hideSubToggles);
  QObject::connect(parent, &FrogPilotSettingsWindow::closeSubSubParentToggle, this, &FrogPilotAdvancedPanel::hideSubSubToggles);
  QObject::connect(parent, &FrogPilotSettingsWindow::updateMetric, this, &FrogPilotAdvancedPanel::updateMetric);
  QObject::connect(uiState(), &UIState::driveRated, this, &FrogPilotAdvancedPanel::updateModelLabels);
  QObject::connect(uiState(), &UIState::offroadTransition, this, &FrogPilotAdvancedPanel::updateCarToggles);

  updateMetric();
  updateModelLabels();
}

void FrogPilotAdvancedPanel::showEvent(QShowEvent *event) {
  disableOpenpilotLongitudinal = params.getBool("DisableOpenpilotLongitudinal");
  modelRandomizer = params.getBool("ModelRandomizer");
}

void FrogPilotAdvancedPanel::updateState(const UIState &s) {
  if (!isVisible()) return;

  if (modelManagementOpen) {
    downloadAllModelsBtn->setText(modelDownloading && allModelsDownloading ? tr("CANCEL") : tr("DOWNLOAD"));
    downloadModelBtn->setText(modelDownloading && !allModelsDownloading ? tr("CANCEL") : tr("DOWNLOAD"));

    deleteModelBtn->setEnabled(!modelDeleting && !modelDownloading);
    downloadAllModelsBtn->setEnabled(s.scene.online && !cancellingDownload && !modelDeleting && (!modelDownloading || allModelsDownloading) && !modelsDownloaded);
    downloadModelBtn->setEnabled(s.scene.online && !cancellingDownload && !modelDeleting && !allModelsDownloading && !modelsDownloaded);
    selectModelBtn->setEnabled(!modelDeleting && !modelDownloading && !modelRandomizer);
  }

  started = s.scene.started;
}

void FrogPilotAdvancedPanel::updateCarToggles() {
  float currentFrictionStock = params.getFloat("SteerFrictionStock");
  float currentKPStock = params.getFloat("SteerKPStock");
  float currentLatAccelStock = params.getFloat("SteerLatAccelStock");
  float currentRatioStock = params.getFloat("SteerRatioStock");

  auto carParams = params.get("CarParamsPersistent");
  if (!carParams.empty()) {
    AlignedBuffer aligned_buf;
    capnp::FlatArrayMessageReader cmsg(aligned_buf.align(carParams.data(), carParams.size()));
    cereal::CarParams::Reader CP = cmsg.getRoot<cereal::CarParams>();

    auto carName = CP.getCarName();

    hasAutoTune = (carName == "hyundai" || carName == "toyota") && CP.getLateralTuning().which() == cereal::CarParams::LateralTuning::TORQUE;
    hasOpenpilotLongitudinal = hasLongitudinalControl(CP);
    steerFrictionStock = CP.getLateralTuning().getTorque().getFriction();
    steerKPStock = CP.getLateralTuning().getTorque().getKp();
    steerLatAccelStock = CP.getLateralTuning().getTorque().getLatAccelFactor();
    steerRatioStock = CP.getSteerRatio();

    steerFrictionToggle->setTitle(QString(tr("Friction (Default: %1)")).arg(QString::number(steerFrictionStock, 'f', 2)));
    if (currentFrictionStock != steerFrictionStock) {
      if (params.getFloat("SteerFriction") == currentFrictionStock) {
        params.putFloat("SteerFriction", steerFrictionStock);
      }
      params.putFloat("SteerFrictionStock", steerFrictionStock);
    }

    steerKPToggle->setTitle(QString(tr("Kp Factor (Default: %1)")).arg(QString::number(steerKPStock, 'f', 2)));
    steerKPToggle->updateControl(steerKPStock * 0.50, currentKPStock * 1.50);
    if (currentKPStock != steerKPStock) {
      if (params.getFloat("SteerKP") == currentKPStock) {
        params.putFloat("SteerKP", steerKPStock);
      }
      params.putFloat("SteerKPStock", steerKPStock);
    }

    steerLatAccelToggle->setTitle(QString(tr("Lateral Accel (Default: %1)")).arg(QString::number(steerLatAccelStock, 'f', 2)));
    steerLatAccelToggle->updateControl(steerLatAccelStock * 0.75, steerLatAccelStock * 1.25);
    if (currentLatAccelStock != steerLatAccelStock) {
      if (params.getFloat("SteerLatAccel") == steerLatAccelStock) {
        params.putFloat("SteerLatAccel", steerLatAccelStock);
      }
      params.putFloat("SteerLatAccelStock", steerLatAccelStock);
    }

    steerRatioToggle->setTitle(QString(tr("Steer Ratio (Default: %1)")).arg(QString::number(steerRatioStock, 'f', 2)));
    steerRatioToggle->updateControl(steerRatioStock * 0.75, steerRatioStock * 1.25);
    if (currentRatioStock != steerRatioStock) {
      if (params.getFloat("SteerRatio") == steerRatioStock) {
        params.putFloat("SteerRatio", steerRatioStock);
      }
      params.putFloat("SteerRatioStock", steerRatioStock);
    }
  } else {
    hasAutoTune = true;
    hasOpenpilotLongitudinal = true;
  }

  auto torqueParams = params.get("LiveTorqueParameters");
  if (!torqueParams.empty()) {
    AlignedBuffer aligned_buf;
    capnp::FlatArrayMessageReader cmsg(aligned_buf.align(torqueParams.data(), torqueParams.size()));
    cereal::Event::Reader LTP = cmsg.getRoot<cereal::Event>();

    auto liveTorqueParams = LTP.getLiveTorqueParameters();

    liveValid = liveTorqueParams.getLiveValid();
  } else {
    liveValid = false;
  }

  hideToggles();
}

void FrogPilotAdvancedPanel::updateMetric() {
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
    laneLinesWidthToggle->setDescription(tr("Customize the lane line width.\n\nDefault matches the Vienna average of 10 centimeters."));
    roadEdgesWidthToggle->setDescription(tr("Customize the road edges width.\n\nDefault is 1/2 of the Vienna average lane line width of 10 centimeters."));

    laneLinesWidthToggle->updateControl(0, 60, tr(" centimeters"));
    roadEdgesWidthToggle->updateControl(0, 60, tr(" centimeters"));

    pathWidthToggle->updateControl(0, 3, tr(" meters"));
  } else {
    laneLinesWidthToggle->setDescription(tr("Customize the lane line width.\n\nDefault matches the MUTCD average of 4 inches."));
    roadEdgesWidthToggle->setDescription(tr("Customize the road edges width.\n\nDefault is 1/2 of the MUTCD average lane line width of 4 inches."));

    laneLinesWidthToggle->updateControl(0, 24, tr(" inches"));
    roadEdgesWidthToggle->updateControl(0, 24, tr(" inches"));

    pathWidthToggle->updateControl(0, 10, tr(" feet"));
  }
}

void FrogPilotAdvancedPanel::startDownloadAllModels() {
  allModelsDownloading = true;
  modelDownloading = true;

  paramsMemory.putBool("DownloadAllModels", true);

  downloadAllModelsBtn->setValue(tr("Downloading models..."));

  QTimer *progressTimer = new QTimer(this);
  progressTimer->setInterval(100);

  QObject::connect(progressTimer, &QTimer::timeout, this, [=]() {
    QString progress = QString::fromStdString(paramsMemory.get("ModelDownloadProgress"));
    bool downloadComplete = progress.contains(QRegularExpression("All models downloaded!", QRegularExpression::CaseInsensitiveOption));
    bool downloadFailed = progress.contains(QRegularExpression("cancelled|exists|failed|offline", QRegularExpression::CaseInsensitiveOption));

    if (!progress.isEmpty() && progress != "0%") {
      downloadAllModelsBtn->setValue(progress);
    }

    if (downloadComplete || downloadFailed) {
      if (downloadComplete) {
        haveModelsDownloaded = true;
        update();
      }

      downloadAllModelsBtn->setValue(progress);

      paramsMemory.remove("CancelModelDownload");
      paramsMemory.remove("ModelDownloadProgress");

      progressTimer->stop();
      progressTimer->deleteLater();

      QTimer::singleShot(2000, this, [=]() {
        cancellingDownload = false;
        modelDownloading = false;

        paramsMemory.remove("DownloadAllModels");

        downloadAllModelsBtn->setValue("");

        device()->resetInteractiveTimeout(30);
      });
    }
  });
  progressTimer->start();
}

QString FrogPilotAdvancedPanel::processModelName(const QString &modelName) {
  QString modelCleaned = modelName;
  modelCleaned = modelCleaned.remove(QRegularExpression("[🗺️👀📡]")).simplified();
  QString scoreParam = modelCleaned.remove(QRegularExpression("[^a-zA-Z0-9()-]")).replace(" ", "").simplified();
  scoreParam = scoreParam.replace("(Default)", "").replace("-", "");
  return scoreParam;
}

void FrogPilotAdvancedPanel::updateCalibrationDescription() {
  QString model = QString::fromStdString(params.get("ModelName"));
  QString part_model_param = processModelName(model);

  QString desc =
      tr("openpilot requires the device to be mounted within 4° left or right and "
         "within 5° up or 9° down. openpilot is continuously calibrating, resetting is rarely required.");
  std::string calib_bytes = params.get(part_model_param.toStdString() + "CalibrationParams");
  if (!calib_bytes.empty()) {
    try {
      AlignedBuffer aligned_buf;
      capnp::FlatArrayMessageReader cmsg(aligned_buf.align(calib_bytes.data(), calib_bytes.size()));
      auto calib = cmsg.getRoot<cereal::Event>().getLiveCalibration();
      if (calib.getCalStatus() != cereal::LiveCalibrationData::Status::UNCALIBRATED) {
        double pitch = calib.getRpyCalib()[1] * (180 / M_PI);
        double yaw = calib.getRpyCalib()[2] * (180 / M_PI);
        desc += tr(" Your device is pointed %1° %2 and %3° %4.")
                    .arg(QString::number(std::abs(pitch), 'g', 1), pitch > 0 ? tr("down") : tr("up"),
                         QString::number(std::abs(yaw), 'g', 1), yaw > 0 ? tr("left") : tr("right"));
      }
    } catch (kj::Exception) {
      qInfo() << "invalid CalibrationParams";
    }
  }
  qobject_cast<FrogPilotButtonsControl *>(sender())->setDescription(desc);
}

void FrogPilotAdvancedPanel::updateModelLabels() {
  QVector<QPair<QString, int>> modelScores;
  availableModelNames = QString::fromStdString(params.get("AvailableModelsNames")).split(",");

  for (const QString &model : availableModelNames) {
    QString cleanedModel = processModelName(model);
    int score = params.getInt((cleanedModel + "Score").toStdString());

    if (model.contains("(Default)")) {
      modelScores.prepend(qMakePair(model, score));
    } else {
      modelScores.append(qMakePair(model, score));
    }
  }

  labelControls.clear();

  for (const auto &pair : modelScores) {
    QString scoreDisplay = pair.second == 0 ? "N/A" : QString::number(pair.second) + "%";
    LabelControl *labelControl = new LabelControl(pair.first, scoreDisplay, "");
    addItem(labelControl);
    labelControls.append(labelControl);
  }

  for (LabelControl *label : labelControls) {
    label->setVisible(false);
  }
}

void FrogPilotAdvancedPanel::showToggles(std::set<QString> &keys) {
  setUpdatesEnabled(false);

  for (auto &[key, toggle] : toggles) {
    toggle->setVisible(keys.find(key.c_str()) != keys.end());
  }

  setUpdatesEnabled(true);
  update();
}

void FrogPilotAdvancedPanel::hideToggles() {
  setUpdatesEnabled(false);

  customPersonalityOpen = false;
  modelManagementOpen = false;

  for (LabelControl *label : labelControls) {
    label->setVisible(false);
  }

  for (auto &[key, toggle] : toggles) {
    bool subToggles = aggressivePersonalityKeys.find(key.c_str()) != aggressivePersonalityKeys.end() ||
                      customDrivingPersonalityKeys.find(key.c_str()) != customDrivingPersonalityKeys.end() ||
                      developerUIKeys.find(key.c_str()) != developerUIKeys.end() ||
                      lateralTuneKeys.find(key.c_str()) != lateralTuneKeys.end() ||
                      longitudinalTuneKeys.find(key.c_str()) != longitudinalTuneKeys.end() ||
                      modelManagementKeys.find(key.c_str()) != modelManagementKeys.end() ||
                      modelRandomizerKeys.find(key.c_str()) != modelRandomizerKeys.end() ||
                      modelUIKeys.find(key.c_str()) != modelUIKeys.end() ||
                      relaxedPersonalityKeys.find(key.c_str()) != relaxedPersonalityKeys.end() ||
                      standardPersonalityKeys.find(key.c_str()) != standardPersonalityKeys.end() ||
                      trafficPersonalityKeys.find(key.c_str()) != trafficPersonalityKeys.end();

    toggle->setVisible(!subToggles);
  }

  setUpdatesEnabled(true);
  update();
}

void FrogPilotAdvancedPanel::hideSubToggles() {
  if (customPersonalityOpen) {
    customPersonalityOpen = false;
    showToggles(customDrivingPersonalityKeys);
  } else if (modelManagementOpen) {
    for (LabelControl *label : labelControls) {
      label->setVisible(false);
    }

    showToggles(modelManagementKeys);
  }
}

void FrogPilotAdvancedPanel::hideSubSubToggles() {
  if (modelManagementOpen) {
    for (LabelControl *label : labelControls) {
      label->setVisible(false);
    }

    showToggles(modelRandomizerKeys);
  }
}
