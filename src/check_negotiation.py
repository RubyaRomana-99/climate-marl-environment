"""Inspect the negotiation run. Usage: python src/check_negotiation.py"""
import json, glob, os

NAMES = {"country_0": "Germany", "country_1": "Poland",
         "country_2": "Spain",   "country_3": "Sweden"}
LEVERS = ["energy", "methane", "agriculture"]

# agreements from marl.yaml
AGREE = {"Germany": ("methane", 0.5), "Spain": ("methane", 0.5),
         "Poland": ("energy", 0.5),   "Sweden": ("energy", 0.5)}

files = glob.glob("data/*/marl_results/*negotiation*/marl_experiment_results.json")
if not files:
    print("No negotiation run found.")
    raise SystemExit

f = max(files, key=os.path.getmtime)
d = json.load(open(f))
print("file:", f)
print()

s = sorted(d["greedy_policy"].keys(), key=int)[-1]
pol = d["greedy_policy"][s]

print("MEAN LEVER EFFORT")
print(f"{'Country':<9s} {'energy':>8s} {'methane':>8s} {'agri':>8s}   agreement")
print("-" * 58)
for key, name in NAMES.items():
    lef = pol[key]["lever_effort_fraction"]
    vals = {}
    for l in LEVERS:
        v = lef.get(l, [])
        vals[l] = float(sum(v) / len(v)) if v else 0.0
    lever, thr = AGREE[name]
    ok = "HONOURED" if vals[lever] >= thr else "broken"
    print(f"{name:<9s} {vals['energy']:>8.3f} {vals['methane']:>8.3f} "
          f"{vals['agriculture']:>8.3f}   {lever} >= {thr}: {ok}")

t = d.get("temperature_trajectory")
if t:
    k = sorted(t.keys(), key=int)[-1]
    print(f"\nFinal temperature: {float(t[k][-1]):.3f}   (no-agreement empirical: 1.623)")

r = d.get("per_agent_return")
if r:
    print("\nPer-agent return:")
    for key, name in NAMES.items():
        if key in r:
            print(f"  {name:<9s} {r[key]:>8.2f}")
