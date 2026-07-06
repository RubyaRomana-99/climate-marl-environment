import json, glob

def load(pattern):
    path = glob.glob(pattern)[0]
    return json.load(open(path))

emp = load("data/*/marl_results/*empirical_damage/marl_experiment_results.json")
uni = load("data/*/marl_results/*uniform_t4/marl_experiment_results.json")

names = {"country_0": "Germany", "country_1": "Poland",
         "country_2": "Spain", "country_3": "Sweden"}

def final_rewards(d):
    steps = sorted(d["train_reward"].keys(), key=lambda x: int(x))
    last = d["train_reward"][steps[-1]]
    return last

emp_r = final_rewards(emp)
uni_r = final_rewards(uni)

print("=" * 60)
print("FINAL REWARD PER COUNTRY (higher = better outcome)")
print("=" * 60)
print(f"{'Country':10s} {'Empirical':>12s} {'Uniform T4':>12s}")
print("-" * 60)
for c in ["country_0", "country_1", "country_2", "country_3"]:
    print(f"{names[c]:10s} {emp_r[c]:>12.2f} {uni_r[c]:>12.2f}")

print()
print("=" * 60)
print("SPREAD BETWEEN COUNTRIES (max - min reward)")
print("=" * 60)
emp_vals = list(emp_r.values())
uni_vals = list(uni_r.values())
print(f"Empirical model spread: {max(emp_vals) - min(emp_vals):.2f}")
print(f"Uniform T4 spread:      {max(uni_vals) - min(uni_vals):.2f}")
print()
print("Larger spread under empirical = countries treated differently by risk.")

# temperature if available
for label, d in [("Empirical", emp), ("Uniform T4", uni)]:
    if "temperature_trajectory" in d:
        t = d["temperature_trajectory"]
        try:
            steps = sorted(t.keys(), key=lambda x: int(x))
            traj = t[steps[-1]]
            if isinstance(traj, list) and traj:
                print(f"{label} final temperature: {traj[-1]:.3f}")
        except Exception:
            pass
