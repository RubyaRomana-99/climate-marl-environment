"""
Analyze the recalibrated (v2) runs: empirical + negotiation, 5 seeds each.
Usage:  python src/v2_analysis.py
"""
import json, glob, os, re
import numpy as np

NAMES  = ["Germany", "Poland", "Spain", "Sweden"]
KEYS   = ["country_0", "country_1", "country_2", "country_3"]
LEVERS = ["energy", "methane", "agriculture"]
AGREE  = {"Germany": ("methane", 0.5), "Spain": ("methane", 0.5),
          "Poland": ("energy", 0.5),   "Sweden": ("energy", 0.5)}

def load_runs(pattern):
    files = sorted(glob.glob(pattern))
    out = []
    for f in files:
        try: out.append(json.load(open(f)))
        except: print("  ! could not read", f)
    return out

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

def extract_returns(logfile, n_runs):
    pat = re.compile(r"\[greedy\] per-agent returns:\s*(\{.*?\})")
    import ast
    evals = []
    with open(logfile, errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                try: evals.append(ast.literal_eval(m.group(1)))
                except: pass
    if not evals: return []
    per = len(evals) // n_runs
    return [evals[(i+1)*per - 1] for i in range(n_runs)]

emp_runs  = load_runs("data/*/marl_results/*v2_empirical_ds*/marl_experiment_results.json")
nego_runs = load_runs("data/*/marl_results/*v2_nego_ds*/marl_experiment_results.json")
print(f"Loaded {len(emp_runs)} empirical seeds, {len(nego_runs)} negotiation seeds.\n")

emp_pols  = [last_policy(d) for d in emp_runs]
nego_pols = [last_policy(d) for d in nego_runs]
emp_temps  = np.array([final_temp(d) for d in emp_runs])
nego_temps = np.array([final_temp(d) for d in nego_runs])

# --- TABLE 1: REWARD (from log file) ---
logfiles = sorted(glob.glob("rerun_*.out"))
if logfiles:
    logfile = max(logfiles, key=os.path.getmtime)
    all_returns = extract_returns(logfile, 10)
    emp_returns  = all_returns[0::2]  # odd positions (emp runs first each seed)
    nego_returns = all_returns[1::2]  # even positions
    print("=" * 70)
    print("TABLE 1 -- FINAL REWARD PER COUNTRY (mean +/- std over seeds)")
    print("=" * 70)
    print(f"{'Country':<10s} {'Empirical':>20s} {'Negotiation':>20s}")
    print("-" * 55)
    for name, key in zip(NAMES, KEYS):
        e = np.array([r.get(key, np.nan) for r in emp_returns])
        n = np.array([r.get(key, np.nan) for r in nego_returns])
        print(f"{name:<10s} {e.mean():>8.2f} +/- {e.std():<5.2f}   "
              f"{n.mean():>8.2f} +/- {n.std():<5.2f}")
    emp_spreads  = np.array([max(r.values())-min(r.values()) for r in emp_returns])
    nego_spreads = np.array([max(r.values())-min(r.values()) for r in nego_returns])
    print(f"\nEmpirical spread: {emp_spreads.mean():.2f} +/- {emp_spreads.std():.2f}")
    print(f"Negotiation spread: {nego_spreads.mean():.2f} +/- {nego_spreads.std():.2f}")

# --- TABLE 2: TEMPERATURE ---
print("\n" + "=" * 70)
print("TABLE 2 -- FINAL TEMPERATURE (mean +/- std over seeds)")
print("=" * 70)
print(f"Empirical    : {emp_temps.mean():.3f} +/- {emp_temps.std():.3f} C")
print(f"Negotiation  : {nego_temps.mean():.3f} +/- {nego_temps.std():.3f} C")

# --- TABLE 3: MITIGATION EFFORT ---
print("\n" + "=" * 70)
print("TABLE 3 -- AVERAGE MITIGATION EFFORT (mean +/- std over seeds)")
print("=" * 70)
for lever in LEVERS:
    print(f"\n{lever.upper()}:")
    print(f"{'Country':<10s} {'Empirical':>16s} {'Negotiation':>16s}")
    for name, key in zip(NAMES, KEYS):
        e = np.array([effort(p, key, lever) for p in emp_pols])
        n = np.array([effort(p, key, lever) for p in nego_pols])
        agreed = " *" if AGREE[name][0] == lever else ""
        print(f"{name:<10s} {e.mean():>8.3f} +/- {e.std():<5.3f}  "
              f"{n.mean():>8.3f} +/- {n.std():<5.3f}{agreed}")

# --- TABLE 4: NEGOTIATION COMPLIANCE ---
print("\n" + "=" * 70)
print("TABLE 4 -- AGREEMENT COMPLIANCE")
print("=" * 70)
print(f"{'Country':<10s} {'Lever':<9s} {'Threshold':>9s} {'Effort':>16s} {'Honoured':>10s}")
print("-" * 60)
for name, key in zip(NAMES, KEYS):
    lever, thr = AGREE[name]
    e = np.array([effort(p, key, lever) for p in nego_pols])
    honoured = int((e >= thr).sum())
    print(f"{name:<10s} {lever:<9s} {thr:>9.2f} "
          f"{e.mean():>8.3f} +/- {e.std():<5.3f} {honoured:>6d}/{len(e)}")

# --- TABLE 5: EVENNESS ---
print("\n" + "=" * 70)
print("TABLE 5 -- EVENNESS OF EFFORT")
print("=" * 70)
for label, pols in [("Empirical", emp_pols), ("Negotiation", nego_pols)]:
    totals_per_seed = []
    for p in pols:
        t = [sum(effort(p, k, l) for l in LEVERS) for k in KEYS]
        totals_per_seed.append(t)
    arr = np.array(totals_per_seed)
    means = arr.mean(axis=0)
    cv = means.std() / means.mean() if means.mean() > 0.001 else float('inf')
    contributing = int((means > 0.01).sum())
    print(f"{label}:")
    print(f"  per-country totals: " + "  ".join(f"{m:.3f}" for m in means))
    print(f"  CV = {cv:.2f}   contributing = {contributing}/4")
