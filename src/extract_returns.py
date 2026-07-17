"""
Pull the final per-agent returns for each seed out of the batch log.
Usage:  python src/extract_returns.py negoseeds_28920205.out
"""
import sys, re, ast
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "negoseeds_28920205.out"
NAMES = {"country_0": "Germany", "country_1": "Poland",
         "country_2": "Spain",   "country_3": "Sweden"}

# every greedy-eval line, in order
pat = re.compile(r"\[greedy\] per-agent returns:\s*(\{.*?\})")
# a run ends when results are written
end = re.compile(r"marl_experiment_results\.json|Saved to .*marl_results")

runs, current = [], None
with open(path, errors="ignore") as f:
    for line in f:
        m = pat.search(line)
        if m:
            try:
                current = ast.literal_eval(m.group(1))
            except Exception:
                pass
        elif end.search(line) and current is not None:
            runs.append(current)
            current = None
if current is not None:
    runs.append(current)

# fall back: if boundary detection failed, split the greedy lines evenly
if len(runs) < 2:
    all_r = []
    with open(path, errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                try:
                    all_r.append(ast.literal_eval(m.group(1)))
                except Exception:
                    pass
    print(f"(boundary detection found {len(runs)} runs; "
          f"{len(all_r)} greedy evals total)")
    print("Showing the last evaluation only:")
    runs = all_r[-1:] if all_r else []

print(f"Found {len(runs)} completed runs.\n")
for i, r in enumerate(runs, 1):
    vals = ", ".join(f"{NAMES.get(k,k)}={v}" for k, v in r.items())
    print(f"  seed {i}: {vals}")

if len(runs) >= 2:
    print("\n" + "=" * 56)
    print("FINAL REWARD PER COUNTRY (mean +/- std over seeds)")
    print("=" * 56)
    for key, name in NAMES.items():
        a = np.array([r[key] for r in runs if key in r], float)
        if a.size:
            print(f"{name:<9s} {a.mean():>8.2f} +/- {a.std():<5.2f}   (n={a.size})")
