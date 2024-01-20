import cereal.messaging as messaging
import numpy as np

from openpilot.common.conversions import Conversions as CV
from openpilot.common.numpy_fast import interp
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, V_CRUISE_MAX
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.longitudinal_planner import A_CRUISE_MIN, A_CRUISE_MAX_BP, get_max_accel
from openpilot.selfdrive.modeld.constants import ModelConstants

from openpilot.selfdrive.frogpilot.functions.frogpilot_functions import get_min_accel_eco, get_max_accel_eco, get_min_accel_sport, get_max_accel_sport

class FrogPilotPlanner:
  def __init__(self, params, params_memory):
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
    if self.acceleration_profile == 1:
      self.accel_limits = [get_min_accel_eco(v_ego), get_max_accel_eco(v_ego)]
    elif self.acceleration_profile in (2, 3):
      self.accel_limits = [get_min_accel_sport(v_ego), get_max_accel_sport(v_ego)]
    else:
      self.accel_limits = [A_CRUISE_MIN, get_max_accel(v_ego)]

    self.v_cruise = self.update_v_cruise(carState, controlsState, modelData, enabled, v_cruise, v_ego)

    self.x_desired_trajectory_full = np.interp(ModelConstants.T_IDXS, T_IDXS_MPC, mpc.x_solution)
    self.x_desired_trajectory = self.x_desired_trajectory_full[:CONTROL_N]

  def update_v_cruise(self, carState, controlsState, modelData, enabled, v_cruise, v_ego):
    v_ego_diff = max(carState.vEgoRaw - carState.vEgoCluster, 0)
    return v_cruise - v_ego_diff

  def publish_lateral(self, sm, pm, DH):
    frogpilot_lateral_plan_send = messaging.new_message('frogpilotLateralPlan')
    frogpilot_lateral_plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState', 'modelV2'])
    frogpilotLateralPlan = frogpilot_lateral_plan_send.frogpilotLateralPlan

    pm.send('frogpilotLateralPlan', frogpilot_lateral_plan_send)

  def publish_longitudinal(self, sm, pm, mpc):
    frogpilot_longitudinal_plan_send = messaging.new_message('frogpilotLongitudinalPlan')
    frogpilot_longitudinal_plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState'])
    frogpilotLongitudinalPlan = frogpilot_longitudinal_plan_send.frogpilotLongitudinalPlan

    frogpilotLongitudinalPlan.distances = self.x_desired_trajectory.tolist()

    pm.send('frogpilotLongitudinalPlan', frogpilot_longitudinal_plan_send)

  def update_frogpilot_params(self, params, params_memory):
    self.is_metric = params.get_bool("IsMetric")

    custom_ui = params.get_bool("CustomUI")
    self.blind_spot_path = params.get_bool("BlindSpotPath") and custom_ui

    lateral_tune = params.get_bool("LateralTune")
    self.average_desired_curvature = params.get_bool("AverageCurvature") and lateral_tune

    longitudinal_tune = params.get_bool("LongitudinalTune")
    self.acceleration_profile = params.get_int("AccelerationProfile") if longitudinal_tune else 0
    self.aggressive_acceleration = params.get_bool("AggressiveAcceleration") and longitudinal_tune
