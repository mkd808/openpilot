#include "selfdrive/frogpilot/ui/qt/offroad/longitudinal_settings.h"

FrogPilotLongitudinalPanel::FrogPilotLongitudinalPanel(FrogPilotSettingsWindow *parent) : FrogPilotListWidget(parent) {
  const std::vector<std::tuple<QString, QString, QString, QString>> longitudinalToggles {
    {"ConditionalExperimental", tr("Conditional Experimental Mode"), tr("Automatically switches to 'Experimental Mode' under predefined conditions."), "../frogpilot/assets/toggle_icons/icon_conditional.png"},
    {"CESpeed", tr("Below"), tr("Switch to 'Experimental Mode' below this speed when not following a lead vehicle."), ""},
    {"CECurves", tr("Curve Detected Ahead"), tr("Switch to 'Experimental Mode' when a curve is detected."), ""},
    {"CELead", tr("Lead Detected Ahead"), tr("Switch to 'Experimental Mode' when a slower or stopped lead vehicle is detected ahead."), ""},
    {"CEModelStopTime", tr("Model Wants To Stop In The Next"), tr("Switch to 'Experimental Mode' when the model wants to stop like when it detects a stop light or stop sign."), ""},
    {"CENavigation", tr("Navigation Based"), tr("Switch to 'Experimental Mode' based on navigation data. (i.e. Intersections, stop signs, upcoming turns, etc.)"), ""},
    {"CESignal", tr("Turn Signal When Below Highway Speeds"), tr("Switch to 'Experimental Mode' when using turn signals below highway speeds to help assist with turns."), ""},
    {"HideCEMStatusBar", tr("Hide the Status Bar"), tr("Don't use the status bar for 'Conditional Experimental Mode'."), ""},

    {"CurveSpeedControl", tr("Curve Speed Control"), tr("Slow down for anticipated curves detected by the downloaded maps."), "../frogpilot/assets/toggle_icons/icon_speed_map.png"},
    {"CurveDetectionMethod", tr("Curve Detection Method"), tr("Choose your preferred curve detection method."), ""},
    {"DisableCurveSpeedSmoothing", tr("Disable UI Smoothing"), tr("Disables the smoothing for the requested speed in the onroad UI to show exactly what speed MTSC/VTSC is currently requesting."), ""},
    {"MTSCCurvatureCheck",  tr("Curve Detection Failsafe"), tr("Only trigger MTSC when the model detects a curve in the road. Purely used as a failsafe to prevent false positives. Leave this off if you never experience false positives."), ""},
    {"CurveSensitivity", tr("Curve Detection Sensitivity"), tr("Set curve detection sensitivity. Higher values prompt earlier responses, lower values lead to smoother but later reactions."), ""},
    {"TurnAggressiveness", tr("Turn Speed Aggressiveness"), tr("Set turn speed aggressiveness. Higher values result in faster turns, lower values yield gentler turns."), ""},

    {"ExperimentalModeActivation", tr("Experimental Mode Activation"), tr("Toggle Experimental Mode with either buttons on the steering wheel or the screen. \n\nOverrides 'Conditional Experimental Mode'."), "../assets/img_experimental_white.svg"},
    {"ExperimentalModeViaLKAS", tr("Click LKAS Button"), tr("Enable/disable 'Experimental Mode' by clicking the 'LKAS' button on your steering wheel."), ""},
    {"ExperimentalModeViaTap", tr("Double Tap the UI"), tr("Enable/disable 'Experimental Mode' by double tapping the onroad UI within a 0.5 second time frame."), ""},
    {"ExperimentalModeViaDistance", tr("Long Press Distance"), tr("Enable/disable 'Experimental Mode' by holding down the 'distance' button on your steering wheel for 0.5 seconds."), ""},

    {"LongitudinalTune", tr("Longitudinal Tuning"), tr("Modify openpilot's acceleration and braking behavior."), "../frogpilot/assets/toggle_icons/icon_longitudinal_tune.png"},
    {"AccelerationProfile", tr("Acceleration Profile"), tr("Change the acceleration rate to be either sporty or eco-friendly."), ""},
    {"DecelerationProfile", tr("Deceleration Profile"), tr("Change the deceleration rate to be either sporty or eco-friendly."), ""},
    {"HumanAcceleration", tr("Human-Like Acceleration"), tr("Tweaks the acceleration behavior to be more 'human-like'."), ""},
    {"HumanFollowing", tr("Human-Like Following Distance"), tr("Tweaks the following distance dynamically to be more 'human-like' when coming up behind slower/stopped leads or following faster leads."), ""},
    {"StoppingDistance", tr("Increase Stop Distance"), tr("Increase the stopping distance for a more comfortable stop from lead vehicles."), ""},

    {"QOLControls", tr("Quality of Life"), tr("Miscellaneous quality of life changes to improve your overall openpilot experience."), "../frogpilot/assets/toggle_icons/quality_of_life.png"},
    {"CustomCruise", tr("Cruise Increase Interval"), tr("Set a custom interval to increase the max set speed by."), ""},
    {"CustomCruiseLong", tr("Cruise Increase Interval (Long Press)"), tr("Set a custom interval to increase the max set speed by when holding down the cruise increase button."), ""},
    {"ForceStandstill", tr("Force Standstill State"), tr("Keeps openpilot in the 'standstill' state until the gas pedal is pressed."), ""},
    {"MapGears", tr("Map Accel/Decel To Gears"), tr("Map your acceleration/deceleration profile to your 'Eco' and/or 'Sport' gears."), ""},
    {"ReverseCruise", tr("Reverse Cruise Increase"), tr("Reverses the 'long press' functionality logic to increase the max set speed by 5 instead of 1. Useful to increase the max speed quickly."), ""},
    {"SetSpeedOffset", tr("Set Speed Offset"), tr("Set an offset for your desired set speed."), ""},

    {"SpeedLimitController", tr("Speed Limit Controller"), tr("Automatically adjust the max speed to match the current speed limit using 'Open Street Maps', 'Navigate On openpilot', or your car's dashboard (Toyotas/Lexus/HKG only)."), "../assets/offroad/icon_speed_limit.png"},
    {"SLCControls", tr("Controls Settings"), tr("Manage toggles related to 'Speed Limit Controller's controls."), ""},
    {"Offset1", tr("Speed Limit Offset (0-34 mph)"), tr("Speed limit offset for speed limits between 0-34 mph."), ""},
    {"Offset2", tr("Speed Limit Offset (35-54 mph)"), tr("Speed limit offset for speed limits between 35-54 mph."), ""},
    {"Offset3", tr("Speed Limit Offset (55-64 mph)"), tr("Speed limit offset for speed limits between 55-64 mph."), ""},
    {"Offset4", tr("Speed Limit Offset (65-99 mph)"), tr("Speed limit offset for speed limits between 65-99 mph."), ""},
    {"SLCFallback", tr("Fallback Method"), tr("Choose your fallback method when there is no speed limit available."), ""},
    {"SLCOverride", tr("Override Method"), tr("Choose your preferred method to override the current speed limit."), ""},
    {"SLCPriority", tr("Priority Order"), tr("Configure the speed limit priority order."), ""},
    {"SLCQOL", tr("Quality of Life"), tr("Manage toggles related to 'Speed Limit Controller's quality of life features."), ""},
    {"SLCConfirmation", tr("Confirm New Speed Limits"), tr("Don't automatically start using the new speed limit until it's been manually confirmed."), ""},
    {"ForceMPHDashboard", tr("Force MPH From Dashboard Readings"), tr("Force MPH readings from the dashboard. Only use this if you live in an area where the speed limits from your dashboard are in KPH, but you use MPH."), ""},
    {"SLCLookaheadHigher", tr("Prepare For Higher Speed Limits"), tr("Set a 'lookahead' value to prepare for upcoming speed limits higher than your current speed limit using the data stored in 'Open Street Maps'."), ""},
    {"SLCLookaheadLower", tr("Prepare For Lower Speed Limits"), tr("Set a 'lookahead' value to prepare for upcoming speed limits lower than your current speed limit using the data stored in 'Open Street Maps'."), ""},
    {"SetSpeedLimit", tr("Use Current Speed Limit As Set Speed"), tr("Sets your max speed to the current speed limit if one is populated when you initially enable openpilot."), ""},
    {"SLCVisuals", tr("Visuals Settings"), tr("Manage toggles related to 'Speed Limit Controller's visuals."), ""},
    {"ShowSLCOffset", tr("Show Speed Limit Offset"), tr("Show the speed limit offset separated from the speed limit in the onroad UI when using 'Speed Limit Controller'."), ""},
    {"SpeedLimitChangedAlert", tr("Speed Limit Changed Alert"), tr("Trigger an alert whenever the speed limit changes."), ""},
    {"UseVienna", tr("Use Vienna Speed Limit Signs"), tr("Use the Vienna (EU) speed limit style signs as opposed to MUTCD (US)."), ""},
  };

  for (const auto &[param, title, desc, icon] : longitudinalToggles) {
    AbstractControl *longitudinalToggle;

    if (param == "ConditionalExperimental") {
      FrogPilotParamManageControl *conditionalExperimentalToggle = new FrogPilotParamManageControl(param, title, desc, icon);
      QObject::connect(conditionalExperimentalToggle, &FrogPilotParamManageControl::manageButtonClicked, [this]() {
        showToggles(conditionalExperimentalKeys);
      });
      longitudinalToggle = conditionalExperimentalToggle;
    } else if (param == "CESpeed") {
      FrogPilotParamValueControl *CESpeed = new FrogPilotParamValueControl(param, title, desc, icon, 0, 99, tr("mph"), std::map<int, QString>(), 1.0, true);
      FrogPilotParamValueControl *CESpeedLead = new FrogPilotParamValueControl("CESpeedLead", tr(" With Lead"), tr("Switch to 'Experimental Mode' below this speed when following a lead vehicle."), icon, 0, 99, tr("mph"), std::map<int, QString>(), 1.0, true);
      FrogPilotDualParamControl *conditionalSpeeds = new FrogPilotDualParamControl(CESpeed, CESpeedLead);
      longitudinalToggle = reinterpret_cast<AbstractControl*>(conditionalSpeeds);
    } else if (param == "CECurves") {
      std::vector<QString> curveToggles{"CECurvesLead"};
      std::vector<QString> curveToggleNames{tr("With Lead")};
      longitudinalToggle = new FrogPilotButtonToggleControl(param, title, desc, curveToggles, curveToggleNames);
    } else if (param == "CELead") {
      std::vector<QString> leadToggles{"CESlowerLead", "CEStoppedLead"};
      std::vector<QString> leadToggleNames{tr("Slower Lead"), tr("Stopped Lead")};
      longitudinalToggle = new FrogPilotButtonToggleControl(param, title, desc, leadToggles, leadToggleNames);
    } else if (param == "CENavigation") {
      std::vector<QString> navigationToggles{"CENavigationIntersections", "CENavigationTurns", "CENavigationLead"};
      std::vector<QString> navigationToggleNames{tr("Intersections"), tr("Turns"), tr("With Lead")};
      longitudinalToggle = new FrogPilotButtonToggleControl(param, title, desc, navigationToggles, navigationToggleNames);
    } else if (param == "CEModelStopTime") {
      std::map<int, QString> modelStopTimeLabels;
      for (int i = 0; i <= 10; i++) {
        modelStopTimeLabels[i] = (i == 0) ? tr("Off") : QString::number(i) + " seconds";
      }
      longitudinalToggle = new FrogPilotParamValueControl(param, title, desc, icon, 0, 10, QString(), modelStopTimeLabels);

    } else if (param == "CurveSpeedControl") {
      FrogPilotParamManageControl *curveControlToggle = new FrogPilotParamManageControl(param, title, desc, icon);
      QObject::connect(curveControlToggle, &FrogPilotParamManageControl::manageButtonClicked, [this]() {
        curveDetectionBtn->setEnabledButtons(0, QDir("/data/media/0/osm/offline").exists());
        curveDetectionBtn->setCheckedButton(0, params.getBool("MTSCEnabled"));
        curveDetectionBtn->setCheckedButton(1, params.getBool("VisionTurnControl"));

        showToggles(curveSpeedKeys);
      });
      longitudinalToggle = curveControlToggle;
    } else if (param == "CurveDetectionMethod") {
      curveDetectionBtn = new FrogPilotButtonsControl(title, desc, {tr("Map Based"), tr("Vision")}, true, false);
      connect(curveDetectionBtn, &FrogPilotButtonsControl::buttonClicked, [=](int id) {
        if (id == 0) {
          bool currentMTSCEnabled = params.getBool("MTSCEnabled");
          params.putBool("MTSCEnabled", !currentMTSCEnabled);
          curveDetectionBtn->setCheckedButton(0, !currentMTSCEnabled);
        } else if (id == 1) {
          bool currentVisionTurnControl = params.getBool("VisionTurnControl");
          params.putBool("VisionTurnControl", !currentVisionTurnControl);
          curveDetectionBtn->setCheckedButton(1, !currentVisionTurnControl);
        }
      });
      longitudinalToggle = curveDetectionBtn;
    } else if (param == "CurveSensitivity" || param == "TurnAggressiveness") {
      longitudinalToggle = new FrogPilotParamValueControl(param, title, desc, icon, 1, 200, "%");

    } else if (param == "ExperimentalModeActivation") {
      FrogPilotParamManageControl *experimentalModeActivationToggle = new FrogPilotParamManageControl(param, title, desc, icon);
      QObject::connect(experimentalModeActivationToggle, &FrogPilotParamManageControl::manageButtonClicked, [this]() {
        std::set<QString> modifiedExperimentalModeActivationKeys = experimentalModeActivationKeys;

        if (isSubaru || (params.getBool("AlwaysOnLateral") && params.getBool("AlwaysOnLateralLKAS"))) {
          modifiedExperimentalModeActivationKeys.erase("ExperimentalModeViaLKAS");
        }

        showToggles(modifiedExperimentalModeActivationKeys);
      });
      longitudinalToggle = experimentalModeActivationToggle;

    } else if (param == "LongitudinalTune") {
      FrogPilotParamManageControl *longitudinalTuneToggle = new FrogPilotParamManageControl(param, title, desc, icon);
      QObject::connect(longitudinalTuneToggle, &FrogPilotParamManageControl::manageButtonClicked, [this]() {
        showToggles(longitudinalTuneKeys);
      });
      longitudinalToggle = longitudinalTuneToggle;
    } else if (param == "AccelerationProfile") {
      std::vector<QString> profileOptions{tr("Standard"), tr("Eco"), tr("Sport"), tr("Sport+")};
      ButtonParamControl *profileSelection = new ButtonParamControl(param, title, desc, icon, profileOptions);
      longitudinalToggle = profileSelection;
    } else if (param == "DecelerationProfile") {
      std::vector<QString> profileOptions{tr("Standard"), tr("Eco"), tr("Sport")};
      ButtonParamControl *profileSelection = new ButtonParamControl(param, title, desc, icon, profileOptions);
      longitudinalToggle = profileSelection;
    } else if (param == "StoppingDistance") {
      longitudinalToggle = new FrogPilotParamValueControl(param, title, desc, icon, 0, 10, tr(" feet"));

    } else if (param == "QOLControls") {
      FrogPilotParamManageControl *qolToggle = new FrogPilotParamManageControl(param, title, desc, icon);
      QObject::connect(qolToggle, &FrogPilotParamManageControl::manageButtonClicked, [this]() {
        std::set<QString> modifiedQolKeys = qolKeys;

        if (!hasPCMCruise) {
          modifiedQolKeys.erase("ReverseCruise");
        } else {
          modifiedQolKeys.erase("CustomCruise");
          modifiedQolKeys.erase("CustomCruiseLong");
          modifiedQolKeys.erase("SetSpeedOffset");
        }

        if (!isToyota && !isGM && !isHKGCanFd) {
          modifiedQolKeys.erase("MapGears");
        }

        showToggles(modifiedQolKeys);
      });
      longitudinalToggle = qolToggle;
    } else if (param == "CustomCruise") {
      longitudinalToggle = new FrogPilotParamValueControl(param, title, desc, icon, 1, 99, tr("mph"));
    } else if (param == "CustomCruiseLong") {
      longitudinalToggle = new FrogPilotParamValueControl(param, title, desc, icon, 1, 99, tr("mph"));
    } else if (param == "ForceStandstill") {
      std::vector<QString> forceStopToggles{"ForceStops"};
      std::vector<QString> forceStopToggleNames{tr("Only For Stop Lights/Stop Signs")};
      longitudinalToggle = new FrogPilotButtonToggleControl(param, title, desc, forceStopToggles, forceStopToggleNames);
    } else if (param == "MapGears") {
      std::vector<QString> mapGearsToggles{"MapAcceleration", "MapDeceleration"};
      std::vector<QString> mapGearsToggleNames{tr("Acceleration"), tr("Deceleration")};
      longitudinalToggle = new FrogPilotButtonToggleControl(param, title, desc, mapGearsToggles, mapGearsToggleNames);
    } else if (param == "SetSpeedOffset") {
      longitudinalToggle = new FrogPilotParamValueControl(param, title, desc, icon, 0, 99, tr("mph"));

    } else if (param == "SpeedLimitController") {
      FrogPilotParamManageControl *speedLimitControllerToggle = new FrogPilotParamManageControl(param, title, desc, icon);
      QObject::connect(speedLimitControllerToggle, &FrogPilotParamManageControl::manageButtonClicked, [this]() {
        slcOpen = true;
        showToggles(speedLimitControllerKeys);
      });
      longitudinalToggle = speedLimitControllerToggle;
    } else if (param == "SLCControls") {
      ButtonControl *manageSLCControlsBtn = new ButtonControl(title, tr("MANAGE"), desc);
      QObject::connect(manageSLCControlsBtn, &ButtonControl::clicked, [this]() {
        openSubParentToggle();
        showToggles(speedLimitControllerControlsKeys);
      });
      longitudinalToggle = reinterpret_cast<AbstractControl*>(manageSLCControlsBtn);
    } else if (param == "SLCQOL") {
      ButtonControl *manageSLCQOLBtn = new ButtonControl(title, tr("MANAGE"), desc);
      QObject::connect(manageSLCQOLBtn, &ButtonControl::clicked, [this]() {
        openSubParentToggle();
        std::set<QString> modifiedSpeedLimitControllerQOLKeys = speedLimitControllerQOLKeys;

        if (hasPCMCruise) {
          modifiedSpeedLimitControllerQOLKeys.erase("SetSpeedLimit");
        }

        if (!isToyota) {
          modifiedSpeedLimitControllerQOLKeys.erase("ForceMPHDashboard");
        }

        showToggles(modifiedSpeedLimitControllerQOLKeys);
      });
      longitudinalToggle = reinterpret_cast<AbstractControl*>(manageSLCQOLBtn);
    } else if (param == "SLCConfirmation") {
      std::vector<QString> slcConfirmationToggles{"SLCConfirmationLower", "SLCConfirmationHigher"};
      std::vector<QString> slcConfirmationNames{tr("Lower Limits"), tr("Higher Limits")};
      longitudinalToggle = new FrogPilotButtonToggleControl(param, title, desc, slcConfirmationToggles, slcConfirmationNames);
    } else if (param == "SLCLookaheadHigher" || param == "SLCLookaheadLower") {
      longitudinalToggle = new FrogPilotParamValueControl(param, title, desc, icon, 0, 60, tr(" seconds"));
    } else if (param == "SLCVisuals") {
      ButtonControl *manageSLCVisualsBtn = new ButtonControl(title, tr("MANAGE"), desc);
      QObject::connect(manageSLCVisualsBtn, &ButtonControl::clicked, [this]() {
        openSubParentToggle();
        showToggles(speedLimitControllerVisualsKeys);
      });
      longitudinalToggle = reinterpret_cast<AbstractControl*>(manageSLCVisualsBtn);
    } else if (param == "Offset1" || param == "Offset2" || param == "Offset3" || param == "Offset4") {
      longitudinalToggle = new FrogPilotParamValueControl(param, title, desc, icon, -99, 99, tr("mph"));
    } else if (param == "ShowSLCOffset") {
      std::vector<QString> slcOffsetToggles{"ShowSLCOffsetUI"};
      std::vector<QString> slcOffsetToggleNames{tr("Control Via UI")};
      longitudinalToggle = new FrogPilotButtonToggleControl(param, title, desc, slcOffsetToggles, slcOffsetToggleNames);
    } else if (param == "SLCFallback") {
      std::vector<QString> fallbackOptions{tr("Set Speed"), tr("Experimental Mode"), tr("Previous Limit")};
      ButtonParamControl *fallbackSelection = new ButtonParamControl(param, title, desc, icon, fallbackOptions);
      longitudinalToggle = fallbackSelection;
    } else if (param == "SLCOverride") {
      std::vector<QString> overrideOptions{tr("None"), tr("Manual Set Speed"), tr("Set Speed")};
      ButtonParamControl *overrideSelection = new ButtonParamControl(param, title, desc, icon, overrideOptions);
      longitudinalToggle = overrideSelection;
    } else if (param == "SLCPriority") {
      ButtonControl *slcPriorityButton = new ButtonControl(title, tr("SELECT"), desc);
      QStringList primaryPriorities = {tr("None"), tr("Dashboard"), tr("Navigation"), tr("Offline Maps"), tr("Highest"), tr("Lowest")};
      QStringList secondaryTertiaryPriorities = {tr("None"), tr("Dashboard"), tr("Navigation"), tr("Offline Maps")};
      QStringList priorityPrompts = {tr("Select your primary priority"), tr("Select your secondary priority"), tr("Select your tertiary priority")};

      QObject::connect(slcPriorityButton, &ButtonControl::clicked, [=]() {
        QStringList selectedPriorities;

        for (int i = 1; i <= 3; i++) {
          QStringList currentPriorities = (i == 1) ? primaryPriorities : secondaryTertiaryPriorities;
          QStringList prioritiesToDisplay = currentPriorities;
          for (const auto &selectedPriority : qAsConst(selectedPriorities)) {
            prioritiesToDisplay.removeAll(selectedPriority);
          }

          if (!hasDashSpeedLimits) {
            prioritiesToDisplay.removeAll(tr("Dashboard"));
          }

          if (prioritiesToDisplay.size() == 1 && prioritiesToDisplay.contains(tr("None"))) {
            break;
          }

          QString priorityKey = QString("SLCPriority%1").arg(i);
          QString selection = MultiOptionDialog::getSelection(priorityPrompts[i - 1], prioritiesToDisplay, "", this);

          if (selection.isEmpty()) break;

          params.putNonBlocking(priorityKey.toStdString(), selection.toStdString());
          selectedPriorities.append(selection);

          if (selection == tr("Lowest") || selection == tr("Highest") || selection == tr("None")) break;

          updateFrogPilotToggles();
        }

        selectedPriorities.removeAll(tr("None"));
        slcPriorityButton->setValue(selectedPriorities.join(", "));
      });

      QStringList initialPriorities;
      for (int i = 1; i <= 3; i++) {
        QString priorityKey = QString("SLCPriority%1").arg(i);
        QString priority = QString::fromStdString(params.get(priorityKey.toStdString()));

        if (!priority.isEmpty() && primaryPriorities.contains(priority) && priority != tr("None")) {
          initialPriorities.append(priority);
        }
      }
      slcPriorityButton->setValue(initialPriorities.join(", "));
      longitudinalToggle = slcPriorityButton;

    } else {
      longitudinalToggle = new ParamControl(param, title, desc, icon);
    }

    addItem(longitudinalToggle);
    toggles[param.toStdString()] = longitudinalToggle;

    tryConnect<ToggleControl>(longitudinalToggle, &ToggleControl::toggleFlipped, this, updateFrogPilotToggles);
    tryConnect<FrogPilotButtonToggleControl>(longitudinalToggle, &FrogPilotButtonToggleControl::buttonClicked, this, updateFrogPilotToggles);
    tryConnect<FrogPilotParamManageControl>(longitudinalToggle, &FrogPilotParamManageControl::manageButtonClicked, this, &FrogPilotLongitudinalPanel::openParentToggle);
    tryConnect<FrogPilotParamValueControl>(longitudinalToggle, &FrogPilotParamValueControl::valueChanged, this, updateFrogPilotToggles);

    QObject::connect(longitudinalToggle, &AbstractControl::showDescriptionEvent, [this]() {
      update();
    });
  }

  QObject::connect(static_cast<ToggleControl*>(toggles["ExperimentalModeViaLKAS"]), &ToggleControl::toggleFlipped, [this](bool state) {
    if (state && params.getBool("AlwaysOnLateralLKAS")) {
      params.putBoolNonBlocking("AlwaysOnLateralLKAS", false);
    }
  });

  QObject::connect(parent, &FrogPilotSettingsWindow::closeParentToggle, this, &FrogPilotLongitudinalPanel::hideToggles);
  QObject::connect(parent, &FrogPilotSettingsWindow::closeSubParentToggle, this, &FrogPilotLongitudinalPanel::hideSubToggles);
  QObject::connect(parent, &FrogPilotSettingsWindow::updateMetric, this, &FrogPilotLongitudinalPanel::updateMetric);
  QObject::connect(uiState(), &UIState::offroadTransition, this, &FrogPilotLongitudinalPanel::updateCarToggles);

  updateMetric();
}

void FrogPilotLongitudinalPanel::updateCarToggles() {
  auto carParams = params.get("CarParamsPersistent");
  if (!carParams.empty()) {
    AlignedBuffer aligned_buf;
    capnp::FlatArrayMessageReader cmsg(aligned_buf.align(carParams.data(), carParams.size()));
    cereal::CarParams::Reader CP = cmsg.getRoot<cereal::CarParams>();

    auto carName = CP.getCarName();
    auto safetyConfigs = CP.getSafetyConfigs();
    auto safetyModel = safetyConfigs[0].getSafetyModel();

    hasDashSpeedLimits = carName == "hyundai" || carName == "toyota";
    hasPCMCruise = CP.getPcmCruise();
    isGM = carName == "gm";
    isHKGCanFd = carName == "hyundai" && safetyModel == cereal::CarParams::SafetyModel::HYUNDAI_CANFD;
    isSubaru = carName == "subaru";
    isToyota = carName == "toyota";
  } else {
    hasDashSpeedLimits = true;
    hasPCMCruise = true;
    isGM = true;
    isHKGCanFd = true;
    isToyota = true;
  }

  hideToggles();
}

void FrogPilotLongitudinalPanel::updateMetric() {
  bool previousIsMetric = isMetric;
  isMetric = params.getBool("IsMetric");

  if (isMetric != previousIsMetric) {
    double distanceConversion = isMetric ? FOOT_TO_METER : METER_TO_FOOT;
    double speedConversion = isMetric ? MILE_TO_KM : KM_TO_MILE;

    params.putFloatNonBlocking("StoppingDistance", params.getFloat("StoppingDistance") * distanceConversion);

    params.putFloatNonBlocking("CESpeed", params.getFloat("CESpeed") * speedConversion);
    params.putFloatNonBlocking("CESpeedLead", params.getFloat("CESpeedLead") * speedConversion);
    params.putFloatNonBlocking("CustomCruise", params.getFloat("CustomCruise") * speedConversion);
    params.putFloatNonBlocking("CustomCruiseLong", params.getFloat("CustomCruiseLong") * speedConversion);
    params.putFloatNonBlocking("Offset1", params.getFloat("Offset1") * speedConversion);
    params.putFloatNonBlocking("Offset2", params.getFloat("Offset2") * speedConversion);
    params.putFloatNonBlocking("Offset3", params.getFloat("Offset3") * speedConversion);
    params.putFloatNonBlocking("Offset4", params.getFloat("Offset4") * speedConversion);
    params.putFloatNonBlocking("SetSpeedOffset", params.getFloat("SetSpeedOffset") * speedConversion);
  }

  FrogPilotDualParamControl *ceSpeedToggle = reinterpret_cast<FrogPilotDualParamControl*>(toggles["CESpeed"]);
  FrogPilotParamValueControl *customCruiseToggle = static_cast<FrogPilotParamValueControl*>(toggles["CustomCruise"]);
  FrogPilotParamValueControl *customCruiseLongToggle = static_cast<FrogPilotParamValueControl*>(toggles["CustomCruiseLong"]);
  FrogPilotParamValueControl *offset1Toggle = static_cast<FrogPilotParamValueControl*>(toggles["Offset1"]);
  FrogPilotParamValueControl *offset2Toggle = static_cast<FrogPilotParamValueControl*>(toggles["Offset2"]);
  FrogPilotParamValueControl *offset3Toggle = static_cast<FrogPilotParamValueControl*>(toggles["Offset3"]);
  FrogPilotParamValueControl *offset4Toggle = static_cast<FrogPilotParamValueControl*>(toggles["Offset4"]);
  FrogPilotParamValueControl *setSpeedOffsetToggle = static_cast<FrogPilotParamValueControl*>(toggles["SetSpeedOffset"]);
  FrogPilotParamValueControl *stoppingDistanceToggle = static_cast<FrogPilotParamValueControl*>(toggles["StoppingDistance"]);

  if (isMetric) {
    offset1Toggle->setTitle(tr("Speed Limit Offset (0-34 kph)"));
    offset2Toggle->setTitle(tr("Speed Limit Offset (35-54 kph)"));
    offset3Toggle->setTitle(tr("Speed Limit Offset (55-64 kph)"));
    offset4Toggle->setTitle(tr("Speed Limit Offset (65-99 kph)"));

    offset1Toggle->setDescription(tr("Set speed limit offset for limits between 0-34 kph."));
    offset2Toggle->setDescription(tr("Set speed limit offset for limits between 35-54 kph."));
    offset3Toggle->setDescription(tr("Set speed limit offset for limits between 55-64 kph."));
    offset4Toggle->setDescription(tr("Set speed limit offset for limits between 65-99 kph."));

    ceSpeedToggle->updateControl(0, 150, tr("kph"));
    customCruiseToggle->updateControl(1, 150, tr("kph"));
    customCruiseLongToggle->updateControl(1, 150, tr("kph"));
    offset1Toggle->updateControl(-99, 99, tr("kph"));
    offset2Toggle->updateControl(-99, 99, tr("kph"));
    offset3Toggle->updateControl(-99, 99, tr("kph"));
    offset4Toggle->updateControl(-99, 99, tr("kph"));
    setSpeedOffsetToggle->updateControl(0, 150, tr("kph"));

    stoppingDistanceToggle->updateControl(0, 5, tr(" meters"));
  } else {
    offset1Toggle->setTitle(tr("Speed Limit Offset (0-34 mph)"));
    offset2Toggle->setTitle(tr("Speed Limit Offset (35-54 mph)"));
    offset3Toggle->setTitle(tr("Speed Limit Offset (55-64 mph)"));
    offset4Toggle->setTitle(tr("Speed Limit Offset (65-99 mph)"));

    offset1Toggle->setDescription(tr("Set speed limit offset for limits between 0-34 mph."));
    offset2Toggle->setDescription(tr("Set speed limit offset for limits between 35-54 mph."));
    offset3Toggle->setDescription(tr("Set speed limit offset for limits between 55-64 mph."));
    offset4Toggle->setDescription(tr("Set speed limit offset for limits between 65-99 mph."));

    ceSpeedToggle->updateControl(0, 99, tr("mph"));
    customCruiseToggle->updateControl(1, 99, tr("mph"));
    customCruiseLongToggle->updateControl(1, 99, tr("mph"));
    offset1Toggle->updateControl(-99, 99, tr("mph"));
    offset2Toggle->updateControl(-99, 99, tr("mph"));
    offset3Toggle->updateControl(-99, 99, tr("mph"));
    offset4Toggle->updateControl(-99, 99, tr("mph"));
    setSpeedOffsetToggle->updateControl(0, 99, tr("mph"));

    stoppingDistanceToggle->updateControl(0, 10, tr(" feet"));
  }
}

void FrogPilotLongitudinalPanel::showToggles(std::set<QString> &keys) {
  setUpdatesEnabled(false);

  for (auto &[key, toggle] : toggles) {
    toggle->setVisible(keys.find(key.c_str()) != keys.end());
  }

  setUpdatesEnabled(true);
  update();
}

void FrogPilotLongitudinalPanel::hideToggles() {
  setUpdatesEnabled(false);

  slcOpen = false;

  for (auto &[key, toggle] : toggles) {
    bool subToggles = conditionalExperimentalKeys.find(key.c_str()) != conditionalExperimentalKeys.end() ||
                      curveSpeedKeys.find(key.c_str()) != curveSpeedKeys.end() ||
                      experimentalModeActivationKeys.find(key.c_str()) != experimentalModeActivationKeys.end() ||
                      longitudinalTuneKeys.find(key.c_str()) != longitudinalTuneKeys.end() ||
                      qolKeys.find(key.c_str()) != qolKeys.end() ||
                      speedLimitControllerKeys.find(key.c_str()) != speedLimitControllerKeys.end() ||
                      speedLimitControllerControlsKeys.find(key.c_str()) != speedLimitControllerControlsKeys.end() ||
                      speedLimitControllerQOLKeys.find(key.c_str()) != speedLimitControllerQOLKeys.end() ||
                      speedLimitControllerVisualsKeys.find(key.c_str()) != speedLimitControllerVisualsKeys.end();

    toggle->setVisible(!subToggles);
  }

  setUpdatesEnabled(true);
  update();
}

void FrogPilotLongitudinalPanel::hideSubToggles() {
  if (slcOpen) {
    showToggles(speedLimitControllerKeys);
  }
}
