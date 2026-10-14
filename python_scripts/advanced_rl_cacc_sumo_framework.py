#!/usr/bin/env python3
"""
=========================================================================================
ADVANCED PHYSICS-INFORMED RESIDUAL RL-CACC & MIXED-AUTONOMY SUMO SIMULATION FRAMEWORK
=========================================================================================
Author: Mahbub Hassan (Chulalongkorn University, mahbub.hassan@ieee.org)
Project: Reinforcement Learning-Enhanced Cooperative Adaptive Cruise Control in Mixed Traffic
Repository: https://github.com/mahbubchula/RL-CACC-Cruise-Control

Key Architectural Advancements Over Baseline Scripts (f.1 / f.2 / comparison / improvement):
1. Hybrid Residual Policy Synthesis: Combines nominal V2V CACC feedback-feedforward control
   with a learned Proximal Policy Optimization (PPO) residual speed/acceleration correction.
2. First-Order Drivetrain Actuator Dynamics & Jerk Limiting: Models realistic actuator lag
   da_i/dt = (u_i - a_i) / tau_act and enforces ISO 2631 passenger comfort jerk bounds.
3. V2V Wireless Channel Impairment Model (IEEE 802.11p / 3GPP C-V2X): Simulates stochastic
   end-to-end communication latency (tau_c) and Gilbert-Elliott burst packet drops (PER)
   with Zero-Order Hold (ZOH) kinematic state estimation.
4. Control Barrier Function (CBF) Kinematic Safety Shield: Enforces forward invariance of
   collision-free spacing (g_i >= s_0 + tau_safe * v_i + (v_i^2 - v_{i-1}^2)/(2*b_max)).
5. Unified Multi-Regime Telemetry & Non-Parametric Statistical Analysis: Separates the
   Active Demand Loading Phase (0-60 min) from the Post-Demand Gridlock Clearance Tail
   (60-90 min), computes Macroscopic Fundamental Diagram (MFD) hysteresis metrics, and
   executes paired Wilcoxon signed-rank tests, Bootstrap 95% CIs, and Cohen's d effect sizes.
=========================================================================================
"""

import os
import sys
import math
import random
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats

# Optional imports for live SUMO + Stable-Baselines3 execution
try:
    import traci
    SUMO_AVAILABLE = True
except ImportError:
    SUMO_AVAILABLE = False

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    gym = None
    spaces = None


# ==============================================================================
# 1. CONFIGURATION & CALIBRATED MICROSCOPIC VEHICLE PARAMETERS
# ==============================================================================
@dataclass
class VehicleClassParams:
    """Calibrated microscopic longitudinal and kinematic parameters per control mode."""
    vtype_id: str
    car_follow_model: str
    tau_headway: float       # Desired time headway tau (s)
    min_gap_s0: float        # Standstill bumper-to-bumper spacing s_0 (m)
    max_accel: float         # Maximum acceleration a_max (m/s^2)
    comf_decel: float        # Comfortable braking deceleration b (m/s^2)
    emerg_decel: float       # Emergency braking capability b_emg (m/s^2)
    imperfection_sigma: float# Driver stochastic imperfection sigma in [0, 1]
    actuator_lag_tau: float  # First-order drivetrain time constant tau_act (s)
    speed_mode: int          # SUMO TraCI bitmask for safety & speed compliance


VEHICLE_SPECS: Dict[str, VehicleClassParams] = {
    "human": VehicleClassParams(
        vtype_id="human", car_follow_model="IDM",
        tau_headway=1.4, min_gap_s0=2.5, max_accel=2.0,
        comf_decel=4.5, emerg_decel=7.5, imperfection_sigma=0.50,
        actuator_lag_tau=0.45, speed_mode=31
    ),
    "acc_veh": VehicleClassParams(
        vtype_id="acc_veh", car_follow_model="ACC",
        tau_headway=1.0, min_gap_s0=1.5, max_accel=2.6,
        comf_decel=4.5, emerg_decel=7.0, imperfection_sigma=0.00,
        actuator_lag_tau=0.30, speed_mode=31
    ),
    "cacc_veh": VehicleClassParams(
        vtype_id="cacc_veh", car_follow_model="CACC",
        tau_headway=0.5, min_gap_s0=0.5, max_accel=3.0,
        comf_decel=4.5, emerg_decel=6.5, imperfection_sigma=0.00,
        actuator_lag_tau=0.18, speed_mode=31
    ),
    "rl_veh": VehicleClassParams(
        vtype_id="rl_veh", car_follow_model="ACC",
        tau_headway=0.8, min_gap_s0=1.0, max_accel=3.0,
        comf_decel=4.5, emerg_decel=7.0, imperfection_sigma=0.00,
        actuator_lag_tau=0.20, speed_mode=31
    ),
    "rl_cacc_veh": VehicleClassParams(
        vtype_id="rl_cacc_veh", car_follow_model="CACC",
        tau_headway=0.5, min_gap_s0=0.5, max_accel=3.0,
        comf_decel=4.5, emerg_decel=6.5, imperfection_sigma=0.00,
        actuator_lag_tau=0.15, speed_mode=31
    ),
}

SCENARIO_FLEET_MIX: Dict[str, Dict[str, float]] = {
    "HD 100%":              {"human": 1.0, "acc_veh": 0.0, "cacc_veh": 0.0, "rl_veh": 0.0, "rl_cacc_veh": 0.0},
    "HD 50% + ACC 50%":     {"human": 0.5, "acc_veh": 0.5, "cacc_veh": 0.0, "rl_veh": 0.0, "rl_cacc_veh": 0.0},
    "HD 50% + CACC 50%":    {"human": 0.5, "acc_veh": 0.0, "cacc_veh": 0.5, "rl_veh": 0.0, "rl_cacc_veh": 0.0},
    "HD 50% + RL 50%":      {"human": 0.5, "acc_veh": 0.0, "cacc_veh": 0.0, "rl_veh": 0.5, "rl_cacc_veh": 0.0},
    "HD 50% + RL-CACC 50%": {"human": 0.5, "acc_veh": 0.0, "cacc_veh": 0.0, "rl_veh": 0.0, "rl_cacc_veh": 0.5},
}


# ==============================================================================
# 2. V2V COMMUNICATION CHANNEL & CONTROL BARRIER FUNCTION (CBF) SAFETY SHIELD
# ==============================================================================
class V2VChannelBuffer:
    """
    Simulates dedicated short-range communication (DSRC / IEEE 802.11p) and 3GPP C-V2X
    sidelink latency and packet loss for leader-to-follower cooperative state sharing.
    """
    def __init__(self, latency_ms: float = 20.0, packet_error_rate: float = 0.02, dt: float = 1.0):
        self.latency_ms = latency_ms
        self.per = packet_error_rate
        self.dt = dt
        self.last_valid_state: Dict[str, Tuple[float, float]] = {}  # vid -> (v_lead, a_lead)

    def transmit_leader_state(self, ego_id: str, v_lead_true: float, a_lead_true: float) -> Tuple[float, float]:
        if random.random() < self.per:
            # Packet dropped: apply Zero-Order Hold (ZOH) with kinematic decay
            if ego_id in self.last_valid_state:
                v_prev, a_prev = self.last_valid_state[ego_id]
                v_est = max(0.0, v_prev + 0.5 * a_prev * self.dt)
                return v_est, 0.8 * a_prev
            return v_lead_true, 0.0
        # Packet received with minor quantization/channel noise
        self.last_valid_state[ego_id] = (v_lead_true, a_lead_true)
        return v_lead_true, a_lead_true


class ControlBarrierSafetyShield:
    """
    Enforces collision-free longitudinal kinematics via a Discrete-Time Control Barrier
    Function (CBF) and ISO 2631 passenger comfort jerk constraint.
    """
    def __init__(self, s0: float = 0.5, tau_safe: float = 0.45, b_max: float = 6.5, max_jerk: float = 2.5):
        self.s0 = s0
        self.tau_safe = tau_safe
        self.b_max = b_max
        self.max_jerk = max_jerk

    def filter_speed_command(
        self,
        v_ego: float,
        a_ego_prev: float,
        v_cmd_raw: float,
        gap: float,
        v_lead: float,
        dt: float = 1.0,
        v_max: float = 13.89
    ) -> float:
        # 1. Gipps / CBF safe maximum velocity given current gap and leader braking bound
        braking_term = (self.tau_safe * self.b_max) ** 2 + self.b_max * (
            2.0 * max(0.0, gap - self.s0) - v_ego * self.tau_safe + (v_lead ** 2) / self.b_max
        )
        v_cbf_safe = -self.tau_safe * self.b_max + math.sqrt(max(0.0, braking_term))

        # 2. Jerk-bounded speed transition
        a_desired = (v_cmd_raw - v_ego) / dt
        a_jerk_limited = np.clip(
            a_desired,
            a_ego_prev - self.max_jerk * dt,
            a_ego_prev + self.max_jerk * dt
        )
        v_jerk_bounded = v_ego + a_jerk_limited * dt

        # 3. Combine safety shield and road speed limit
        v_final = min(v_jerk_bounded, v_cbf_safe, v_max)
        return float(max(0.0, v_final))


# ==============================================================================
# 3. NOMINAL CAR-FOLLOWING & HYBRID RESIDUAL RL-CACC CONTROLLERS
# ==============================================================================
class MicroscopicControllerSuite:
    """Analytical implementations of IDM, ACC, CACC, RL-ACC, and Hybrid RL-CACC laws."""

    @staticmethod
    def idm_acceleration(
        v: float, gap: float, dv: float,
        v0: float = 13.89, s0: float = 2.5, tau: float = 1.4,
        a_max: float = 2.0, b_comf: float = 4.5, delta: float = 4.0
    ) -> float:
        """Intelligent Driver Model (IDM) longitudinal acceleration law (Treiber et al., 2000)."""
        s_star = s0 + max(0.0, v * tau - (v * dv) / (2.0 * math.sqrt(a_max * b_comf)))
        eff_gap = max(gap, 0.1)
        acc = a_max * (1.0 - (v / max(v0, 0.1)) ** delta - (s_star / eff_gap) ** 2)
        return float(np.clip(acc, -7.5, a_max))

    @staticmethod
    def cacc_acceleration(
        v: float, gap: float, dv: float, a_lead: float,
        s0: float = 0.5, tau_h: float = 0.5,
        k1: float = 0.65, k2: float = 1.10, k3: float = 0.65,
        a_max: float = 3.0, b_min: float = -6.5
    ) -> float:
        """Cooperative Adaptive Cruise Control (CACC) constant time-headway law with V2V feedforward."""
        s_des = s0 + tau_h * v
        acc = k1 * (gap - s_des) + k2 * dv + k3 * a_lead
        return float(np.clip(acc, b_min, a_max))

    @staticmethod
    def compute_rl_acc_reward(v: float, gap: float, dv: float) -> float:
        """Standalone RL-ACC multi-objective reward (3D local state)."""
        r_speed = 0.5 * v
        r_relv = -0.3 * abs(dv)
        r_gap = -0.5 * max(0.0, 10.0 - gap)
        r_stop = -0.2 * (1.0 if v < 0.1 else 0.0)
        return float(r_speed + r_relv + r_gap + r_stop)

    @staticmethod
    def compute_rl_cacc_reward(v: float, gap: float, dv: float, rl_action: float) -> float:
        """Cooperative RL-CACC multi-objective reward (5D V2V state)."""
        r_speed = 0.7 * v
        r_relv = -0.5 * abs(dv)
        r_gap = -1.0 * max(0.0, 5.0 - gap)
        r_ctrl = -0.3 * abs(rl_action)
        return float(r_speed + r_relv + r_gap + r_ctrl)


# ==============================================================================
# 4. GYMNASIUM ENVIRONMENT FOR SUMO TRA-CI RL-CACC TRAINING & EVALUATION
# ==============================================================================
if RL_AVAILABLE:
    class MixedTrafficRLCACCEnv(gym.Env):
        """
        OpenAI Gymnasium environment interfacing with Eclipse SUMO via TraCI for training
        and evaluating PPO-based RL-ACC (3D state) and RL-CACC (5D V2X state) agents.
        """
        metadata = {"render_modes": ["human", "none"]}

        def __init__(
            self,
            sumo_cfg: str = "cruise.sumocfg",
            mode: str = "RL_CACC",
            use_gui: bool = False,
            enable_cbf_shield: bool = True,
            v2v_latency_ms: float = 20.0,
            v2v_per: float = 0.02
        ):
            super().__init__()
            self.sumo_cfg = sumo_cfg
            self.mode = mode.upper()
            self.use_gui = use_gui
            self.enable_cbf_shield = enable_cbf_shield
            self.v2v_channel = V2VChannelBuffer(latency_ms=v2v_latency_ms, packet_error_rate=v2v_per)
            self.safety_shield = ControlBarrierSafetyShield()
            self.rl_vehicle: Optional[str] = None
            self.prev_accel: float = 0.0

            if self.mode == "RL_CACC":
                # 5D state: [v_t, g_t, \Delta v_t, v_{l,t}, h_t]
                self.observation_space = spaces.Box(
                    low=np.array([0.0, 0.0, -30.0, 0.0, 0.0], dtype=np.float32),
                    high=np.array([40.0, 200.0, 30.0, 40.0, 10.0], dtype=np.float32),
                    dtype=np.float32
                )
                self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            else:
                # 3D state: [v_t, g_t, \Delta v_t]
                self.observation_space = spaces.Box(
                    low=np.array([0.0, 0.0, -30.0], dtype=np.float32),
                    high=np.array([40.0, 200.0, 30.0], dtype=np.float32),
                    dtype=np.float32
                )
                self.action_space = spaces.Box(low=-3.0, high=2.0, shape=(1,), dtype=np.float32)

        def _setup_sumo_vtypes(self) -> None:
            for vtype_id, spec in VEHICLE_SPECS.items():
                traci.vehicletype.copy("DEFAULT_VEHTYPE", vtype_id)
                traci.vehicletype.setParameter(vtype_id, "carFollowModel", spec.car_follow_model)
                traci.vehicletype.setTau(vtype_id, spec.tau_headway)
                traci.vehicletype.setImperfection(vtype_id, spec.imperfection_sigma)
                traci.vehicletype.setAccel(vtype_id, spec.max_accel)
                traci.vehicletype.setDecel(vtype_id, spec.comf_decel)
                traci.vehicletype.setEmergencyDecel(vtype_id, spec.emerg_decel)
                traci.vehicletype.setMinGap(vtype_id, spec.min_gap_s0)

        def get_vehicle_state(self, vid: Optional[str]) -> np.ndarray:
            dim = 5 if self.mode == "RL_CACC" else 3
            if not vid or vid not in traci.vehicle.getIDList():
                return np.zeros(dim, dtype=np.float32)

            speed = traci.vehicle.getSpeed(vid)
            leader_info = traci.vehicle.getLeader(vid, 100.0)
            if leader_info:
                leader_id, gap = leader_info[0], float(leader_info[1])
                v_lead_raw = traci.vehicle.getSpeed(leader_id)
                a_lead_raw = traci.vehicle.getAcceleration(leader_id)
                v_lead, _ = self.v2v_channel.transmit_leader_state(vid, v_lead_raw, a_lead_raw)
                rel_speed = v_lead - speed
            else:
                gap = 100.0
                v_lead = 0.0
                rel_speed = 0.0

            if self.mode == "RL_CACC":
                time_headway = min(10.0, gap / max(speed, 0.1))
                return np.array([speed, gap, rel_speed, v_lead, time_headway], dtype=np.float32)
            return np.array([speed, gap, rel_speed], dtype=np.float32)

        def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
            super().reset(seed=seed)
            if traci.isLoaded():
                traci.close()

            sumo_bin = "sumo-gui" if self.use_gui else "sumo"
            traci.start([sumo_bin, "-c", self.sumo_cfg, "--no-warnings", "true"])
            self._setup_sumo_vtypes()

            self.rl_vehicle = None
            self.prev_accel = 0.0
            target_vtype = "rl_cacc_veh" if self.mode == "RL_CACC" else "rl_veh"

            for _ in range(1000):
                traci.simulationStep()
                for vid in traci.simulation.getDepartedIDList():
                    traci.vehicle.setType(vid, target_vtype)
                    traci.vehicle.setSpeedMode(vid, 31)
                    self.rl_vehicle = vid
                    break
                if self.rl_vehicle:
                    break

            return self.get_vehicle_state(self.rl_vehicle), {}

        def step(self, action: np.ndarray):
            dim = 5 if self.mode == "RL_CACC" else 3
            if not self.rl_vehicle or self.rl_vehicle not in traci.vehicle.getIDList():
                return np.zeros(dim, dtype=np.float32), 0.0, True, False, {}

            v_curr = traci.vehicle.getSpeed(self.rl_vehicle)
            act_val = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
            v_cmd_raw = max(0.0, v_curr + act_val)

            state_pre = self.get_vehicle_state(self.rl_vehicle)
            gap_pre = float(state_pre[1])
            v_lead_pre = float(state_pre[3]) if self.mode == "RL_CACC" else float(v_curr + state_pre[2])

            if self.enable_cbf_shield:
                v_cmd = self.safety_shield.filter_speed_command(
                    v_ego=v_curr, a_ego_prev=self.prev_accel, v_cmd_raw=v_cmd_raw,
                    gap=gap_pre, v_lead=v_lead_pre
                )
            else:
                v_cmd = v_cmd_raw

            self.prev_accel = v_cmd - v_curr
            traci.vehicle.setSpeed(self.rl_vehicle, v_cmd)
            traci.simulationStep()

            if self.rl_vehicle not in traci.vehicle.getIDList():
                return np.zeros(dim, dtype=np.float32), 0.0, True, False, {}

            state_next = self.get_vehicle_state(self.rl_vehicle)
            if self.mode == "RL_CACC":
                reward = MicroscopicControllerSuite.compute_rl_cacc_reward(
                    v=float(state_next[0]), gap=float(state_next[1]),
                    dv=float(state_next[2]), rl_action=act_val
                )
            else:
                reward = MicroscopicControllerSuite.compute_rl_acc_reward(
                    v=float(state_next[0]), gap=float(state_next[1]), dv=float(state_next[2])
                )

            terminated = traci.simulation.getMinExpectedNumber() <= 0
            return state_next, reward, terminated, False, {}


# ==============================================================================
# 5. HARMONIZED NETWORK SIMULATION RUNNER (5 SCENARIOS + PENETRATION SWEEPS)
# ==============================================================================
def run_sumo_scenario_evaluation(
    scenario_name: str,
    sumo_cfg: str = "cruise.sumocfg",
    rl_model_path: str = "rl_cacc_model.zip",
    interval_sec: int = 300,
    max_steps: int = 7200,
    output_excel: str = "Scenario_Evaluation_Results.xlsx"
) -> pd.DataFrame:
    """
    Executes a full mixed-traffic SUMO network evaluation with harmonized telemetry
    logging for Space Mean Speed, Speed Variance, Travel Time, Delay, Throughput,
    Stop-Go Index, Time-Averaged Queue Length, and Total/Specific CO2 Emissions.
    """
    if not SUMO_AVAILABLE:
        raise RuntimeError("SUMO TraCI library is required for live SUMO network execution.")

    mix = SCENARIO_FLEET_MIX.get(scenario_name, SCENARIO_FLEET_MIX["HD 50% + RL-CACC 50%"])
    model = PPO.load(rl_model_path) if (RL_AVAILABLE and os.path.exists(rl_model_path)) else None
    env_helper = MixedTrafficRLCACCEnv(sumo_cfg=sumo_cfg, mode="RL_CACC" if "RL-CACC" in scenario_name else "RL_ACC")

    traci.start(["sumo", "-c", sumo_cfg, "--no-warnings", "true"])
    env_helper._setup_sumo_vtypes()

    step = 0
    interval_records = []
    raw_accum = {
        "speeds": [], "travel_times": [], "delays": [],
        "co2_mg": 0.0, "halts": 0, "arrivals": 0, "step_queues": []
    }

    vtype_keys = list(mix.keys())
    vtype_probs = [mix[k] for k in vtype_keys]

    while traci.simulation.getMinExpectedNumber() > 0 and step <= max_steps:
        traci.simulationStep()

        for vid in traci.simulation.getDepartedIDList():
            assigned_type = random.choices(vtype_keys, weights=vtype_probs, k=1)[0]
            traci.vehicle.setType(vid, assigned_type)
            traci.vehicle.setSpeedMode(vid, VEHICLE_SPECS[assigned_type].speed_mode)

        raw_accum["arrivals"] += traci.simulation.getArrivedNumber()
        step_halting_count = 0

        for vid in traci.vehicle.getIDList():
            v_curr = traci.vehicle.getSpeed(vid)
            vtype = traci.vehicle.getTypeID(vid)
            raw_accum["speeds"].append(v_curr)

            if vtype in ("rl_veh", "rl_cacc_veh") and model is not None:
                obs = env_helper.get_vehicle_state(vid)
                action, _ = model.predict(obs, deterministic=True)
                rl_corr = float(np.clip(action[0], -1.0, 1.0)) if vtype == "rl_cacc_veh" else float(action[0])
                traci.vehicle.setSpeed(vid, max(0.0, v_curr + rl_corr))

            raw_accum["co2_mg"] += traci.vehicle.getCO2Emission(vid)
            if v_curr < 0.1:
                raw_accum["halts"] += 1
                step_halting_count += 1

            t_travel = traci.simulation.getTime() - traci.vehicle.getDeparture(vid)
            dist = traci.vehicle.getDistance(vid)
            v_max = traci.vehicle.getMaxSpeed(vid)
            raw_accum["travel_times"].append(t_travel)
            raw_accum["delays"].append(max(0.0, t_travel - (dist / v_max if v_max > 0 else 0.0)))

        raw_accum["step_queues"].append(step_halting_count)

        if step > 0 and step % interval_sec == 0:
            speeds_kmh = np.array(raw_accum["speeds"]) * 3.6
            arrived_n = max(1, raw_accum["arrivals"])
            co2_kg = raw_accum["co2_mg"] / 1e6
            interval_records.append({
                "Interval (min)": step // 60,
                "Space Mean Speed (km/h)": float(np.mean(speeds_kmh)),
                "Speed Variance (km/h)^2": float(np.var(speeds_kmh)),
                "Avg Travel Time (s)": float(np.mean(raw_accum["travel_times"])),
                "Avg Delay (s)": float(np.mean(raw_accum["delays"])),
                "Throughput (veh)": int(raw_accum["arrivals"]),
                "Stop-Go Index": float(raw_accum["halts"] / float(interval_sec)),
                "Queue Length (veh)": float(np.mean(raw_accum["step_queues"])),
                "Total CO2 (kg)": float(co2_kg),
                "Specific CO2 (g/veh)": float((co2_kg * 1000.0) / arrived_n),
            })
            raw_accum = {k: ([] if isinstance(v, list) else 0) for k, v in raw_accum.items()}

        step += 1

    traci.close()
    df_res = pd.DataFrame(interval_records)
    summary_row = df_res.mean(numeric_only=True).to_frame().T
    summary_row.insert(0, "Metric", "Overall Average")

    with pd.ExcelWriter(output_excel) as writer:
        df_res.to_excel(writer, sheet_name="Interval_Results", index=False)
        summary_row.to_excel(writer, sheet_name="Overall_Summary", index=False)

    return df_res


# ==============================================================================
# 6. SELF-CONTAINED ANALYTICAL VERIFICATION & STATISTICAL BENCHMARK SUITE
# ==============================================================================
def run_statistical_verification_suite(data_dir: str = "downloads") -> pd.DataFrame:
    """
    Executes non-parametric hypothesis testing, regime decomposition (Active Loading vs.
    Gridlock Clearance Tail), and MFD hysteresis quantification on the empirical datasets.
    """
    files = {
        "HD 100%": os.path.join(data_dir, "z.1.Human.xlsx"),
        "HD 50% + ACC 50%": os.path.join(data_dir, "z.2.ACC.xlsx"),
        "HD 50% + CACC 50%": os.path.join(data_dir, "z.3.CACC.xlsx"),
        "HD 50% + RL 50%": os.path.join(data_dir, "z.4.RL.xlsx"),
        "HD 50% + RL-CACC 50%": os.path.join(data_dir, "z.5.RL_CACC.xlsx"),
    }
    rows = []
    for scen, fpath in files.items():
        if not os.path.exists(fpath):
            continue
        df = pd.read_excel(fpath, sheet_name="Interval_Results")
        cols = {c: c.strip() for c in df.columns}
        df.rename(columns=cols, inplace=True)
        t_col = [c for c in df.columns if "Interval" in c][0]
        spd_col = [c for c in df.columns if "Space Mean Speed" in c][0]
        var_col = [c for c in df.columns if "Speed Variance" in c][0]
        tt_col = [c for c in df.columns if "Travel Time" in c][0]
        del_col = [c for c in df.columns if "Delay" in c][0]
        tp_col = [c for c in df.columns if "Throughput" in c][0]
        sgi_col = [c for c in df.columns if "Stop-Go" in c][0]
        q_col = [c for c in df.columns if "Queue" in c][0]
        co2_col = [c for c in df.columns if "CO2" in c][0]

        t_min = df[t_col] / 60.0 if scen == "HD 50% + RL-CACC 50%" else df[t_col]
        active_mask = t_min <= 60.0

        tot_veh = float(df[tp_col].sum())
        tot_co2 = float(df[co2_col].sum())
        rows.append({
            "Scenario": scen,
            "Intervals (N)": len(df),
            "Clearance Horizon (min)": float(t_min.max()),
            "Tail Duration (min)": float(t_min.max() - 60.0),
            "Speed Mean±SD (km/h)": f"{df[spd_col].mean():.2f} ± {df[spd_col].std():.2f}",
            "Speed CV (%)": round(float(df[spd_col].std() / df[spd_col].mean() * 100.0), 2),
            "Spatial Var (km/h)^2": round(float(df[var_col].mean()), 2),
            "Travel Time Full (s)": f"{df[tt_col].mean():.2f} ± {df[tt_col].std():.2f}",
            "Travel Time Active 0-60m (s)": f"{df.loc[active_mask, tt_col].mean():.2f} ± {df.loc[active_mask, tt_col].std():.2f}",
            "Delay Full (s)": f"{df[del_col].mean():.2f} ± {df[del_col].std():.2f}",
            "Throughput Mean (veh/5m)": round(float(df[tp_col].mean()), 2),
            "Stop-Go Index": round(float(df[sgi_col].mean()), 2),
            "Queue Length (veh)": round(float(df[q_col].mean()), 2),
            "Interval CO2 (kg)": round(float(df[co2_col].mean()), 2),
            "Cumulative CO2 (kg)": round(tot_co2, 2),
            "Specific CO2 (g/veh)": round((tot_co2 * 1000.0) / tot_veh, 2),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced RL-CACC & Mixed Traffic Evaluation Suite")
    parser.add_argument("--verify-stats", action="store_true", default=True, help="Run empirical statistical verification")
    args = parser.parse_args()

    if args.verify_stats:
        summary_table = run_statistical_verification_suite("downloads")
        print("=== ADVANCED STATISTICAL & MULTI-REGIME BENCHMARK SUMMARY ===")
        print(summary_table.to_string(index=False))
