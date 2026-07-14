"""
Direct unit test of the negotiation penalty — no GPU, no training.
Run on the cluster login node:
    cd ~/climate-marl-environment
    python src/test_negotiation.py
Confirms the penalty fires when agreed levers are below threshold.
"""
import sys, os
from pathlib import Path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
sys.path.insert(0, os.getcwd())

from src.utils.config_utils import load_yaml_config
from src.utils.marl_utils import load_marl_setup
from marl_env import ClimateMARL
import numpy as np

cfg = load_yaml_config("marl.yaml", "marl")
env_config, emission_data, economics_config, actions_config, training_cfg = load_marl_setup(cfg)

print("use_negotiation:", env_config.get("use_negotiation"))
print("agreements:", env_config.get("agreements"))
print("agreement_penalty:", env_config.get("agreement_penalty"))
print("lever_names:", actions_config["lever_names"])

env = ClimateMARL(env_config, emission_data, economics_config, actions_config)
obs, _ = env.reset()

# ---- ALL ZERO effort: every agreement should be BROKEN, penalty > 0 ----
zero_action = np.zeros(len(actions_config["action_sizes"]), dtype=np.int64)
actions = {a: zero_action.copy() for a in env.agents}
obs, rew, term, trunc, info = env.step(actions)
print("\n--- ZERO EFFORT (agreements broken) ---")
for a in env.agents:
    pen = info[a].get("negotiation_penalty", "NOT LOGGED")
    hon = info[a].get("agreement_honored", "NOT LOGGED")
    print(f"  {a}: penalty={pen}  honored={hon}  reward={rew[a]:.4f}")

# ---- MAX effort on all levers: agreements should be HONORED, penalty = 0 ----
env.reset()
max_action = np.zeros(len(actions_config["action_sizes"]), dtype=np.int64)
for j, name in enumerate(actions_config["lever_names"]):
    max_action[j] = len(actions_config["lever_levels"][name]) - 1
actions = {a: max_action.copy() for a in env.agents}
obs, rew, term, trunc, info = env.step(actions)
print("\n--- MAX EFFORT (agreements honored) ---")
for a in env.agents:
    pen = info[a].get("negotiation_penalty", "NOT LOGGED")
    hon = info[a].get("agreement_honored", "NOT LOGGED")
    print(f"  {a}: penalty={pen}  honored={hon}  reward={rew[a]:.4f}")

print("\nEXPECTED: zero-effort penalties > 0 for countries in agreements;")
print("          max-effort penalties = 0 for all.")
