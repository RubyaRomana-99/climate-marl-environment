import sys
import os
import numpy as np
import json

sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from co2_predictor import predict_co2, apply_action, get_starting_state

COUNTRIES = ["Germany", "Poland", "Spain", "Sweden"]

print("\nRunning integration tests...\n")
print("Integration Test - co2_predictor")

print("\nTest 1 — predict_co2() loads and runs")
all_passed = True
for country in COUNTRIES:
    try:
        state = get_starting_state(country)
        co2   = predict_co2(country, state)
        print(f"  {country:<20} CO2 = {co2:.2f} MtCO2eq")
    except Exception as e:
        print(f"  {country:<20} FAILED: {e}")
        all_passed = False

print("\nTest 2 — apply_action() one step")
for country in COUNTRIES:
    try:
        state     = get_starting_state(country)
        co2_2021  = predict_co2(country, state)
        actions   = {"e": 1.0, "m": 1.0, "l": 0.5, "p": 0.08}
        new_state = apply_action(country, state, actions, stochastic=False)
        co2_2022  = predict_co2(country, new_state)
        change    = ((co2_2022 - co2_2021) / co2_2021) * 100
        print(f"  {country:<20} 2021={co2_2021:.1f}  "
              f"2022={co2_2022:.1f}  change={change:+.1f}%")
    except Exception as e:
        print(f"  {country:<20} FAILED: {e}")
        all_passed = False

print("\nTest 3 — Simulating marl_env.step() for 4 years")

country_states = {c: get_starting_state(c) for c in COUNTRIES}
action_map     = {0: "e", 1: "m", 2: "l"}

lever_efforts = np.array([
    [1.0, 1.0, 0.5],
    [0.5, 0.5, 0.0],
    [1.0, 0.0, 0.5],
    [0.0, 0.0, 0.0],
], dtype=np.float32)

print(f"\n  {'Year':<6} {'Germany':>10} {'Poland':>10} "
      f"{'Spain':>10} {'Sweden':>10} {'Total':>10}")
print(f"  {'-'*58}")

for year in range(2021, 2025):
    co2_values = []
    for i, country in enumerate(COUNTRIES):
        if year == 2021:
            co2 = predict_co2(country, country_states[country])
        else:
            actions = {action_map[j]: float(lever_efforts[i, j])
                      for j in range(3)}
            actions["p"] = 0.03
            country_states[country] = apply_action(
                country, country_states[country],
                actions, stochastic=False
            )
            co2 = predict_co2(country, country_states[country])
        co2_values.append(co2)

    total = sum(co2_values)
    vals  = "  ".join(f"{v:>10.1f}" for v in co2_values)
    print(f"  {year:<6} {vals}  {total:>10.1f}")

print("\nTest 4 — Dynamic emission shares")
co2_dict = {c: predict_co2(c, country_states[c]) for c in COUNTRIES}
total    = sum(co2_dict.values())
for country, co2 in co2_dict.items():
    share = co2 / total
    print(f"  {country:<20} CO2={co2:.1f} Mt  share={share:.4f}")
print(f"  Sum of shares: {sum(co2/total for co2 in co2_dict.values()):.4f}")

print("\nTest 5 — Positive coefficient check in rl_parameters.json")
try:
    with open("rl_parameters.json") as f:
        rl = json.load(f)

    positive_found = False
    for country, elasts in rl.get("country_elasticities", {}).items():
        for action in ["e", "m", "l", "p"]:
            if action in elasts:
                val = elasts[action].get("elasticity", 0)
                if val > 0:
                    print(f"  {country} action={action} elasticity={val} is positive")
                    positive_found = True
    if not positive_found:
        print("  All policy elasticities are zero or negative")
except FileNotFoundError:
    print("  rl_parameters.json not found")
    all_passed = False

print()

if all_passed:
    print("Tests completed successfully")
else:
    print("Some tests failed - check errors above")

print()