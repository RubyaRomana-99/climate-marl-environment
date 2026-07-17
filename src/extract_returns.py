"""
Pull the final per-agent returns for each seed out of a batch log.

Usage:
    python src/extract_returns.py negoseeds_28920205.out        # assumes 5 runs
    python src/extract_returns.py <logfile> <n_runs>
"""
import sys, re, ast
import numpy as np

path   = sys.argv[1] if len(sys.argv) > 1 else "negoseeds_28920205.out"
n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5

NAMES = {"country_0": "Germany", "country_1": "Poland",
         "country_2": "Spain",   "country_3": "Sweden"}

pat = re.compile(r"\[greedy\] per-agent returns:\s*(\{.*?\})")

evals = []
with open(path, errors="ignore") as f:
    for line in f:
        m = pat.search(line)
        if m:
            try:
                evals.append(ast.literal_eval(m.group(1)))
            except Exception:
                pass

print(f"{len(evals)} greedy evaluations found in {path}")
if len(evals) % n_runs:
    print(f"  ! {len(evals)} does not divide evenly by {n_runs} runs "
          f"-- check n_runs")
per = len(evals) // n_runs
print(f"  -> {per} evaluations per run, taking the last of each\n")

finals = [evals[(i + 1) * per - 1] for i in range(n_runs)]

for i, r in enumerate(finals, 1):
    vals = "  ".join(f"{NAMES.get(k,k)}={v:>7.2f}" for k, v in r.items())
    print(f"  seed {i}: {vals}")

print("\n" + "=" * 56)
print("FINAL REWARD PER COUNTRY (mean +/- std over seeds)")
print("=" * 56)
for key, name in NAMES.items():
    a = np.array([r[key] for r in finals if key in r], float)
    if a.size:
        print(f"{name:<9s} {a.mean():>8.2f} +/- {a.std():<5.2f}")

# inter-country spread, per seed
spreads = np.array([max(r.values()) - min(r.values()) for r in finals])
print(f"\nInter-country spread: {spreads.mean():.2f} +/- {spreads.std():.2f}")
