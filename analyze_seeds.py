"""
Multi-seed analysis: mean +/- standard deviation across 5 random seeds.
This turns a single-seed result into a statistically defensible one.

Run on the cluster from ~/climate-marl-environment:
    python analyze_seeds.py

Reads all folders matching *empirical_seed*/ and *uniform_seed*/
"""
import json
import glob
import numpy as np

COUNTRIES = ["Germany", "Poland", "Spain", "Sweden"]
KEYS = ["country_0", "country_1", "country_2", "country_3"]
NAME = dict(zip(KEYS, COUNTRIES))
LEVERS = ["energy", "methane", "agriculture"]

def load_runs(pattern):
    paths = sorted(glob.glob(pattern))
    runs = []
    for p in paths:
        try:
            runs.append(json.load(open(p)))
        except Exception as e:
            print(f"  ! could not read {p}: {e}")
    return runs, paths

emp_runs, emp_paths = load_runs("data/*/marl_results/*empirical_seed*/marl_experiment_results.json")
uni_runs, uni_paths = load_runs("data/*/marl_results/*uniform_seed*/marl_experiment_results.json")

print(f"Loaded {len(emp_runs)} empirical seeds, {len(uni_runs)} uniform seeds.\n")

def final_reward(d):
    s = sorted(d["train_reward"].keys(), key=lambda x: int(x))[-1]
    return d["train_reward"][s]

def final_temp(d):
    t = d.get("temperature_trajectory")
    if not t:
        return np.nan
    if isinstance(t, dict):
        t = t[sorted(t.keys(), key=lambda x: int(x))[-1]]
    arr = np.asarray(t, float).ravel()
    return float(arr[-1]) if arr.size else np.nan

def last_policy(d):
    s = sorted(d["greedy_policy"].keys(), key=lambda x: int(x))[-1]
    return d["greedy_policy"][s]

def avg_effort(policy, country, lever):
    lef = policy.get(country, {}).get("lever_effort_fraction", {})
    v = lef.get(lever)
    return float(np.mean(v)) if v else np.nan

# ---------- TABLE 1: reward mean +/- std ----------
print("=" * 76)
print("TABLE 1 -- FINAL REWARD PER COUNTRY  (mean +/- std over seeds)")
print("=" * 76)
print(f"{'Country':10s} {'Uniform T4':>22s} {'Empirical':>22s}")
print("-" * 76)
emp_rewards = {k: [] for k in KEYS}
uni_rewards = {k: [] for k in KEYS}
for d in emp_runs:
    fr = final_reward(d)
    for k in KEYS: emp_rewards[k].append(fr[k])
for d in uni_runs:
    fr = final_reward(d)
    for k in KEYS: uni_rewards[k].append(fr[k])
for k in KEYS:
    em, es = np.mean(emp_rewards[k]), np.std(emp_rewards[k])
    um, us = np.mean(uni_rewards[k]), np.std(uni_rewards[k])
    print(f"{NAME[k]:10s} {um:>13.2f} +/- {us:<5.2f} {em:>13.2f} +/- {es:<5.2f}")

# ---------- TABLE 2: spread mean +/- std ----------
print()
print("=" * 76)
print("TABLE 2 -- INTER-COUNTRY REWARD SPREAD (max - min), per seed")
print("=" * 76)
emp_spreads, uni_spreads = [], []
for d in emp_runs:
    v = list(final_reward(d).values()); emp_spreads.append(max(v) - min(v))
for d in uni_runs:
    v = list(final_reward(d).values()); uni_spreads.append(max(v) - min(v))
print(f"Uniform T4  spread: {np.mean(uni_spreads):.2f} +/- {np.std(uni_spreads):.2f}")
print(f"Empirical   spread: {np.mean(emp_spreads):.2f} +/- {np.std(emp_spreads):.2f}")
if np.mean(uni_spreads) > 0:
    ratios = [e/u for e, u in zip(emp_spreads, uni_spreads)] if len(emp_spreads)==len(uni_spreads) else []
    if ratios:
        print(f"Ratio (empirical / uniform): {np.mean(ratios):.0f}x +/- {np.std(ratios):.0f}")

# ---------- TABLE 3: temperature mean +/- std ----------
print()
print("=" * 76)
print("TABLE 3 -- FINAL TEMPERATURE (mean +/- std over seeds)")
print("=" * 76)
et = [final_temp(d) for d in emp_runs]
ut = [final_temp(d) for d in uni_runs]
et = [x for x in et if not np.isnan(x)]
ut = [x for x in ut if not np.isnan(x)]
if ut: print(f"Uniform T4  : {np.mean(ut):.3f} +/- {np.std(ut):.3f} C")
if et: print(f"Empirical   : {np.mean(et):.3f} +/- {np.std(et):.3f} C")

# ---------- TABLE 4: mitigation effort mean +/- std ----------
print()
print("=" * 76)
print("TABLE 4 -- AVERAGE MITIGATION EFFORT (mean +/- std over seeds)")
print("=" * 76)
for lever in LEVERS:
    print(f"\n{lever.upper()}:")
    print(f"  {'Country':10s} {'Uniform T4':>20s} {'Empirical':>20s}")
    for k in KEYS:
        ev = [avg_effort(last_policy(d), k, lever) for d in emp_runs]
        uv = [avg_effort(last_policy(d), k, lever) for d in uni_runs]
        ev = [x for x in ev if not np.isnan(x)]
        uv = [x for x in uv if not np.isnan(x)]
        if not ev or not uv: continue
        print(f"  {NAME[k]:10s} {np.mean(uv):>11.3f} +/- {np.std(uv):<5.3f} "
              f"{np.mean(ev):>11.3f} +/- {np.std(ev):<5.3f}")

print()
print("=" * 76)
print("CONSISTENCY CHECK")
print("=" * 76)
print(f"Seeds used: {len(emp_runs)} empirical, {len(uni_runs)} uniform")
print("Small standard deviations mean the result is consistent across")
print("random initialisations -- the finding is not an artefact of one run.")
