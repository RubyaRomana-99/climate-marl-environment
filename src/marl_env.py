from gymnasium import spaces
from ray.rllib.env import MultiAgentEnv

import sys
import os
from pathlib import Path
import numpy as np
import copy
import pandas as pd

sys.path.insert(0,os.path.join(os.getcwd(), '../ciceroscm/', 'src'))
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURR_DIR)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from marl_env_step import CICEROSCMEngine, CICERONetEngine
from src.utils.model_utils import instantiate_model, load_state_dict, numpy_state_to_torch

try:
    from co2_predictor import predict_co2, apply_action, get_starting_state
    _CO2_PREDICTOR_AVAILABLE = True
except ImportError:
    _CO2_PREDICTOR_AVAILABLE = False


class ClimateMARL(MultiAgentEnv):

    def __init__(self, env_config, emission_data, economics_config, actions_config):
        super().__init__()

        self.env_config = env_config

        # --- core sizes/time ---
        self.N = int(env_config["N"])
        self.horizon = int(env_config["horizon"])
        self.hist_end = int(env_config["hist_end"])
        self.future_end = int(env_config["future_end"])
        self.window_size = int(env_config["window_size"])
        self.engine_kind = str(env_config["engine"]).lower()
        self.rollout_length = int(env_config["rollout_length"])

        # --- emissions data ---
        self._hist_emissions_pd = emission_data["historical_emissions"].copy()
        self._hist_emissions_np = np.asarray(self._hist_emissions_pd, dtype=np.float32)
        self.baseline_emissions = np.asarray(emission_data["baseline_emissions"], dtype=np.float32)
        self.baseline_emission_growth = np.asarray(
            emission_data["baseline_emission_growth"], dtype=np.float32
        )

        self.gas_names = list(emission_data["gas_names"])
        self.G = len(self.gas_names)

        name_to_index = {name: idx for idx, name in enumerate(self.gas_names)}
        self.controlled_gases = list(env_config["controlled_gases"])
        self.control_indices = np.array(
            [name_to_index[name] for name in self.controlled_gases], dtype=np.int32
        )
        self._controllable_gases = len(self.control_indices)

        emission_shares = np.asarray(emission_data["emission_shares"], dtype=np.float32)
        if emission_shares.shape != (self.N, self.G):
            raise ValueError(
                f"emission_shares must have shape ({self.N}, {self.G}); got {emission_shares.shape}"
            )
        self.emission_shares = emission_shares
        self.baseline_emissions_agent = (
            self.emission_shares[:, None, :] * self.baseline_emissions[None, :, :]
        )

        baseline_control = self.baseline_emissions[0, self.control_indices]
        baseline_control_mean = float(np.mean(baseline_control))
        self._cum_scale = max(1.0, baseline_control_mean * max(1, self.horizon))
        self._emission_scale = max(1.0, baseline_control_mean)

        self.last_agent_emissions = np.zeros((self.N, self.G), dtype=np.float32)
        self.cumulative_emission_delta = np.zeros(
            (self.N, self._controllable_gases), dtype=np.float32
        )

        # --- actions ---
        self.lever_names = list(actions_config["lever_names"])
        if not self.lever_names:
            raise ValueError("At least one mitigation lever must be specified")

        self.lever_count = len(self.lever_names)
        lever_levels_cfg = actions_config["lever_levels"]
        self.lever_levels = []
        for name in self.lever_names:
            levels = np.asarray(lever_levels_cfg[name], dtype=np.float32)
            if levels.ndim != 1 or levels.size == 0:
                raise ValueError(f"Lever '{name}' levels must be a non-empty 1D array")
            self.lever_levels.append(levels)

        self.policy_matrix = np.asarray(actions_config["policy_matrix"], dtype=np.float32)
        if self.policy_matrix.shape != (self.N, self.lever_count, self._controllable_gases):
            raise ValueError(
                "policy_matrix must have shape (num_agents, num_levers, num_controlled_gases); "
                f"got {self.policy_matrix.shape}, expected ({self.N}, {self.lever_count}, {self._controllable_gases})"
            )

        self.adaptation_levels = np.asarray(
            actions_config["adaptation_levels"], dtype=np.float32
        )
        if self.adaptation_levels.ndim != 1 or self.adaptation_levels.size == 0:
            raise ValueError("Adaptation levels must be a non-empty 1D array")

        self.action_sizes = [len(levels) for levels in self.lever_levels]
        self.action_sizes.append(int(self.adaptation_levels.size))

        # --- heterogeneous economics/impacts ---
        costs_cfg = economics_config["costs"]
        self.climate_damage_costs = np.asarray(
            costs_cfg["climate_damage_costs"], dtype=np.float32
        )
        if self.climate_damage_costs.shape != (self.N,):
            raise ValueError(
                "economics.costs.climate_damage_costs must have length equal to num_agents"
            )

        self.adaptation_costs = np.asarray(
            costs_cfg["adaptation_costs"], dtype=np.float32
        )
        if self.adaptation_costs.shape != (self.N,):
            raise ValueError(
                "economics.costs.adaptation_costs must have length equal to num_agents"
            )

        lever_costs = np.zeros((self.N, self.lever_count), dtype=np.float32)
        for idx, name in enumerate(self.lever_names):
            key = f"{name}_costs"
            if key not in costs_cfg:
                raise ValueError(f"Missing economics cost vector for lever '{name}'")
            arr = np.asarray(costs_cfg[key], dtype=np.float32)
            if arr.shape != (self.N,):
                raise ValueError(
                    f"economics.costs.{key} must have length {self.N}, got {len(arr)}"
                )
            lever_costs[:, idx] = arr
        self.lever_costs = lever_costs

        self.prevention_decay = float(economics_config.get("prevention_decay", 0.95))
        self.max_prevention_benefit = float(
            economics_config.get("max_prevention_benefit", 0.5)
        )

        self.prevention_stock = np.zeros(self.N, dtype=np.float32)
        self._act_space = spaces.MultiDiscrete(self.action_sizes)

        # --- agents ---
        self.agents = [f"country_{i}" for i in range(self.N)]

        # --- observation space ---
        obs_low = np.concatenate([
            np.array([0.0, 0.0], dtype=np.float32),
            np.zeros(self.N * self._controllable_gases, dtype=np.float32),
            np.full(self.N * self._controllable_gases, -100.0, dtype=np.float32),
            np.zeros(self.N, dtype=np.float32),
        ])
        obs_high = np.concatenate([
            np.array([3.0, 1.0], dtype=np.float32),
            np.full(self.N * self._controllable_gases, 100.0, dtype=np.float32),
            np.full(self.N * self._controllable_gases, 100.0, dtype=np.float32),
            np.ones(self.N, dtype=np.float32),
        ])
        self._obs_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.observation_spaces = {a: self._obs_space for a in self.agents}
        self.action_spaces = {a: self._act_space for a in self.agents}

        # --- engine parameters ---
        self.net_params = env_config.get("net_params")
        self.scm_params = env_config.get("scm_params")

        self.country_names = list(env_config.get("country_names", []))
        self.country_states = {}
        self.dynamic_co2 = np.zeros(self.N, dtype=np.float32)

        if _CO2_PREDICTOR_AVAILABLE and self.country_names:
            if len(self.country_names) != self.N:
                raise ValueError(
                    f"country_names length ({len(self.country_names)}) must match "
                    f"num_agents ({self.N})"
                )
            for country in self.country_names:
                try:
                    self.country_states[country] = get_starting_state(country)
                except Exception as e:
                    self.country_states[country] = None

        # internal state
        self.reset()
