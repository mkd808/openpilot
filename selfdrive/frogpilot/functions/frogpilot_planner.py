import cereal.messaging as messaging
import numpy as np

from openpilot.common.conversions import Conversions as CV
from openpilot.common.numpy_fast import clip, interp
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, V_CRUISE_MAX
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.longitudinal_planner import A_CRUISE_MIN, A_CRUISE_MAX_BP, get_max_accel
from openpilot.selfdrive.modeld.constants import ModelConstants

from openpilot.selfdrive.frogpilot.functions.frogpilot_functions import get_min_accel_eco, get_max_accel_eco, get_min_accel_sport, get_max_accel_sport

from openpilot.selfdrive.frogpilot.functions.conditional_experimental_mode import ConditionalExperimentalMode
from openpilot.selfdrive.frogpilot.functions.map_turn_speed_controller import MapTurnSpeedController

class FrogPilotPlanner:
  def __init__(self, params, params_memory):
    self.cem = ConditionalExperimentalMode()
    self.mtsc = MapTurnSpeedController()

    self.mtsc_target = 0
    self.v_cruise = 0

    self.x_desired_trajectory = np.zeros(CONTROL_N)

    self.update_frogpilot_params(params, params_memory)

  def update(self, sm, mpc):
    carState, controlsState, modelData = sm['carState'], sm['controlsState'], sm['modelV2']

    enabled = controlsState.enabled

    v_cruise_kph = min(controlsState.vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    v_ego = carState.vEgo

    # Acceleration profiles
    v_cruise_changed = (self.mtsc_target) + 1 < v_cruise  # Use stock acceleration profiles to handle MTSC more precisely
    if v_cruise_changed:
      self.accel_limits = [A_CRUISE_MIN, get_max_accel(v_ego)]
    elif self.acceleration_profile == 1:
      self.accel_limits = [get_min_accel_eco(v_ego), get_max_accel_eco(v_ego)]
    elif self.acceleration_profile in (2, 3):
      self.accel_limits = [get_min_accel_sport(v_ego), get_max_accel_sport(v_ego)]
    else:
      self.accel_limits = [A_CRUISE_MIN, get_max_accel(v_ego)]

    # Conditional Experimental Mode
    if self.conditional_experimental_mode and enabled:
      self.cem.update(carState, sm['frogpilotNavigation'], modelData, mpc, sm['radarState'], carState.standstill, v_ego)

    if enabled:
      self.v_cruise = self.update_v_cruise(carState, controlsState, modelData, enabled, v_cruise, v_ego)
    else:
      self.mtsc_target = v_cruise
      self.v_cruise = v_cruise

    self.x_desired_trajectory_full = np.interp(ModelConstants.T_IDXS, T_IDXS_MPC, mpc.x_solution)
    self.x_desired_trajectory = self.x_desired_trajectory_full[:CONTROL_N]

  def update_v_cruise(self, carState, controlsState, modelData, enabled, v_cruise, v_ego):
    # Pfeiferj's Map Turn Speed Controller
    if self.map_turn_speed_controller:
      self.mtsc_target = np.clip(self.mtsc.target_speed(v_ego, carState.aEgo), MIN_TARGET_V, v_cruise)
      if self.mtsc_target == MIN_TARGET_V:
        self.mtsc_target = v_cruise
    else:
      self.mtsc_target = v_cruise

    v_ego_diff = max(carState.vEgoRaw - carState.vEgoCluster, 0)
    return min(v_cruise, self.mtsc_target) - v_ego_diff

  def publish_lateral(self, sm, pm, DH):
    frogpilot_lateral_plan_send = messaging.new_message('frogpilotLateralPlan')
    frogpilot_lateral_plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState', 'modelV2'])
    frogpilotLateralPlan = frogpilot_lateral_plan_send.frogpilotLateralPlan

    frogpilotLateralPlan.laneWidthLeft = float(DH.lane_width_left)
    frogpilotLateralPlan.laneWidthRight = float(DH.lane_width_right)

    pm.send('frogpilotLateralPlan', frogpilot_lateral_plan_send)

  def publish_longitudinal(self, sm, pm, mpc):
    frogpilot_longitudinal_plan_send = messaging.new_message('frogpilotLongitudinalPlan')
    frogpilot_longitudinal_plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState'])
    frogpilotLongitudinalPlan = frogpilot_longitudinal_plan_send.frogpilotLongitudinalPlan

    frogpilotLongitudinalPlan.adjustedCruise = float(min(self.mtsc_target) * (CV.MS_TO_KPH if self.is_metric else CV.MS_TO_MPH))
    frogpilotLongitudinalPlan.conditionalExperimental = self.cem.experimental_mode
    frogpilotLongitudinalPlan.distances = self.x_desired_trajectory.tolist()
    frogpilotLongitudinalPlan.redLight = bool(self.cem.red_light_detected)

    frogpilotLongitudinalPlan.desiredFollowDistance = mpc.safe_obstacle_distance - mpc.stopped_equivalence_factor
    frogpilotLongitudinalPlan.safeObstacleDistance = mpc.safe_obstacle_distance
    frogpilotLongitudinalPlan.safeObstacleDistanceStock = mpc.safe_obstacle_distance_stock
    frogpilotLongitudinalPlan.stoppedEquivalenceFactor = mpc.stopped_equivalence_factor

    pm.send('frogpilotLongitudinalPlan', frogpilot_longitudinal_plan_send)

  def update_frogpilot_params(self, params, params_memory):
    self.is_metric = params.get_bool("IsMetric")

    self.conditional_experimental_mode = params.get_bool("ConditionalExperimental")
    if self.conditional_experimental_mode:
      self.cem.update_frogpilot_params(self.is_metric, params)
      params.put_bool("ExperimentalMode", True)

    self.custom_personalities = params.get_bool("CustomPersonalities")
    self.aggressive_follow = params.get_int("AggressiveFollow") / 10
    self.standard_follow = params.get_int("StandardFollow") / 10
    self.relaxed_follow = params.get_int("RelaxedFollow") / 10
    self.aggressive_jerk = params.get_int("AggressiveJerk") / 10
    self.standard_jerk = params.get_int("StandardJerk") / 10
    self.relaxed_jerk = params.get_int("RelaxedJerk") / 10

    custom_ui = params.get_bool("CustomUI")
    self.adjacent_lanes = params.get_bool("AdjacentPath") and custom_ui
    self.blind_spot_path = params.get_bool("BlindSpotPath") and custom_ui

    lateral_tune = params.get_bool("LateralTune")
    self.average_desired_curvature = params.get_bool("AverageCurvature") and lateral_tune

    longitudinal_tune = params.get_bool("LongitudinalTune")
    self.acceleration_profile = params.get_int("AccelerationProfile") if longitudinal_tune else 0
    self.aggressive_acceleration = params.get_bool("AggressiveAcceleration") and longitudinal_tune
    self.increased_stopping_distance = params.get_int("StoppingDistance") * (1 if self.is_metric else CV.FOOT_TO_METER) if longitudinal_tune else 0
    self.smoother_braking = params.get_bool("SmoothBraking") and longitudinal_tune

    self.map_turn_speed_controller = params.get_bool("MTSCEnabled")
    if self.map_turn_speed_controller:
      params_memory.put_int("MapTargetLatA", 2 * (params.get_int("MTSCAggressiveness") / 100))

    self.nudgeless = params.get_bool("NudgelessLaneChange")
    self.lane_change_delay = params.get_int("LaneChangeTime") if self.nudgeless else 0
    self.lane_detection = params.get_bool("LaneDetection") and self.nudgeless
    self.one_lane_change = params.get_bool("OneLaneChange") and self.nudgeless
