"""Check whether the negotiation penalty fired. Run: python src/check_penalty.py"""
import json, glob, os

files = glob.glob('data/*/marl_results/*/min_effort_results.json')
if not files:
    print("No min_effort_results.json found. Run: python src/marl_experiment.py --min-effort")
    raise SystemExit

f = max(files, key=os.path.getmtime)
d = json.load(open(f))

print("file:", f)
print("keys:", list(d.keys()))
print()
print("per_agent_return:", d.get("per_agent_return"))
print("total_return:", d.get("total_return"))
print()
print("Expected with negotiation ON and zero effort:")
print("  every agreement broken -> each country pays penalty 1.0/step")
print("  35 steps x 1.0 x 0.1 scaling = about -3.5 extra per country")
print("Compare against a run with use_negotiation: false to confirm.")
