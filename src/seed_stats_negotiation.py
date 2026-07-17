"""
Aggregate the 5-seed negotiation runs.
Usage:  python src/seed_stats_negotiation.py
"""
import json, glob, os
import numpy as np

NAMES  = ["Germany", "Poland", "Spain", "Sweden"]
KEYS   = ["country_0", "country_1", "country_2", "country_3"]
LEVERS = ["energy", "methane", "agriculture"]

# agreements from marl.yaml
AGREE = {"Germany": ("methane", 0.5), "Spain": ("methane", 0.5),
         "Poland": ("energy", 0.5),   "Sweden": ("energy", 0.5)}

# reference values from the empirical (no-agreement) condition
REF_TEMP   = 1.620
REF_EFFORT = {("Germany","methane"): 0.066, ("Spain","energy"): 0.011}

def load(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        try:
            out.append(json.load(open(f)))
        except Exception as e:
            print("  ! could not read", f, e)
    return out

runs = load("data/*/marl_results/*nego_ds*/marl_experiment_results.json")
print(f"Loaded {len(runs)} negotiation seeds.\n")
if not runs:
    raise SystemExit

def last_policy(d):
    s = sorted(d["greedy_policy"].keys(), key=lambda x: int(x))[-1]
    return d["greedy_policy"][s]

def effort(pol, key, lever):
    v = pol.get(key, {}).get("lever_effort_fraction", {}).get(lever, [])
    return float(np.mean(v)) if len(v) else 0.0

def final_temp(d):
    t = d.get("temperature_trajectory")
    if not t: return np.nan
    if isinstance(t, dict):
        t = t[sorted(t.keys(), key=lambda x: int(x))[-1]]
    a = np.asarray(t, float).ravel()
    return float(a[-1]) if a.size else np.nan

pols  = [last_policy(d) for d in runs]
temps = np.array([final_temp(d) for d in runs])

# ---------- 1. compliance ----------
print("=" * 74)
print("TABLE 1 -- AGREEMENT COMPLIANCE (mean +/- std over seeds)")
print("=" * 74)
print(f"{'Country':<9s} {'Lever':<9s} {'Threshold':>9s} {'Effort':>16s} {'Honoured':>10s}")
print("-" * 74)
for name, key in zip(NAMES, KEYS):
    lever, thr = AGREE[name]
    e = np.array([effort(p, key, lever) for p in pols])
    honoured = int((e >= thr).sum())
    print(f"{name:<9s} {lever:<9s} {thr:>9.2f} "
          f"{e.mean():>8.3f} +/- {e.std():<5.3f} {honoured:>6d}/{len(e)}")

# ---------- 2. effort per lever ----------
print("\n" + "=" * 74)
print("TABLE 2 -- MEAN EFFORT PER LEVER (mean +/- std over seeds)")
print("=" * 74)
for lever in LEVERS:
    print(f"\n{lever.upper()}:")
    print(f"{'Country':<9s} {'Effort':>16s}   note")
    for name, key in zip(NAMES, KEYS):
        e = np.array([effort(p, key, lever) for p in pols])
        agreed = "  <-- agreed lever" if AGREE[name][0] == lever else ""
        ref = REF_EFFORT.get((name, lever))
        refs = f"  (no-agreement: {ref})" if ref is not None else ""
        print(f"{name:<9s} {e.mean():>8.3f} +/- {e.std():<5.3f}{agreed}{refs}")

# ---------- 3. evenness ----------
print("\n" + "=" * 74)
print("TABLE 3 -- EVENNESS OF TOTAL MITIGATION EFFORT ACROSS COUNTRIES")
print("=" * 74)
per_seed_spread = []
for p in pols:
    totals = [sum(effort(p, k, l) for l in LEVERS) for k in KEYS]
    per_seed_spread.append(max(totals) - min(totals))
per_seed_spread = np.array(per_seed_spread)
print(f"Spread (max - min) of total effort: "
      f"{per_seed_spread.mean():.3f} +/- {per_seed_spread.std():.3f}")
print("(Lower = effort more evenly distributed across countries.)")

# ---------- 4. temperature ----------
print("\n" + "=" * 74)
print("TABLE 4 -- FINAL TEMPERATURE")
print("=" * 74)
print(f"Negotiation           : {temps.mean():.3f} +/- {temps.std():.3f} C")
print(f"Empirical (reference) : {REF_TEMP:.3f} C")
print(f"Change                : {temps.mean() - REF_TEMP:+.3f} C")

# ---------- 5. reward ----------
print("\n" + "=" * 74)
print("TABLE 5 -- FINAL REWARD PER COUNTRY (mean +/- std over seeds)")
print("=" * 74)
have = [d for d in runs if d.get("per_agent_return")]
if have:
    for name, key in zip(NAMES, KEYS):
        r = np.array([d["per_agent_return"][key] for d in have
                      if key in d["per_agent_return"]])
        if r.size:
            print(f"{name:<9s} {r.mean():>8.2f} +/- {r.std():<5.2f}")
else:
    print("per_agent_return not present in these files.")
