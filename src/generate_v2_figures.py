"""
Generate all updated figures for the corrected (v2) results.
Run on the cluster:  python src/generate_v2_figures.py

Reads v2_empirical and v2_nego results, plus the old uniform results.
Outputs PDF+PNG for each figure, no titles inside (captions do that).
"""
import json, glob, os, re, ast
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
})

NAMES = ["Germany", "Poland", "Spain", "Sweden"]
KEYS  = ["country_0", "country_1", "country_2", "country_3"]
LEVERS = ["energy", "methane", "agriculture"]
COLORS = {"Germany":"#0072B2","Poland":"#D55E00","Spain":"#009E73","Sweden":"#CC79A7"}

def load_runs(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(pattern))]

def last_policy(d):
    s = sorted(d["greedy_policy"].keys(), key=lambda x: int(x))[-1]
    return d["greedy_policy"][s]

def effort(pol, key, lever):
    v = pol.get(key, {}).get("lever_effort_fraction", {}).get(lever, [])
    return float(np.mean(v)) if v else 0.0

def final_temp(d):
    t = d.get("temperature_trajectory")
    if not t: return np.nan
    if isinstance(t, dict):
        t = t[sorted(t.keys(), key=lambda x: int(x))[-1]]
    return float(np.asarray(t, float).ravel()[-1])

def temp_trajectory(d):
    t = d.get("temperature_trajectory")
    if not t: return []
    if isinstance(t, dict):
        t = t[sorted(t.keys(), key=lambda x: int(x))[-1]]
    return list(np.asarray(t, float).ravel())

def extract_returns(logfile, n_runs):
    pat = re.compile(r"\[greedy\] per-agent returns:\s*(\{.*?\})")
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

# load data
emp  = load_runs("data/*/marl_results/*v2_empirical_ds*/marl_experiment_results.json")
nego = load_runs("data/*/marl_results/*v2_nego_ds*/marl_experiment_results.json")
# uniform from old runs (unchanged by recalibration)
uni_files = sorted(glob.glob("data/*/marl_results/*uniform_ds*/marl_experiment_results.json"))
if not uni_files:
    uni_files = sorted(glob.glob("data/*/marl_results/*heterogenous*/marl_experiment_results.json"))
uni = [json.load(open(f)) for f in uni_files[:5]] if uni_files else []

print(f"Loaded: {len(emp)} empirical, {len(nego)} negotiation, {len(uni)} uniform")

emp_pols  = [last_policy(d) for d in emp]
nego_pols = [last_policy(d) for d in nego]

# get rewards from log
logfiles = sorted(glob.glob("rerun_*.out"))
emp_returns, nego_returns = [], []
if logfiles:
    all_ret = extract_returns(max(logfiles, key=os.path.getmtime), 10)
    emp_returns  = all_ret[0::2]
    nego_returns = all_ret[1::2]

# uniform rewards from old log
uni_returns = []
uni_logs = sorted(glob.glob("seeds_*.out"))
if uni_logs:
    # old seeds ran 10 runs (5 emp + 5 uni), uniform is the even-indexed
    all_old = extract_returns(max(uni_logs, key=os.path.getmtime), 10)
    # in old script: empirical first, then uniform per seed
    uni_returns = all_old[1::2] if len(all_old) >= 10 else []

outdir = "v2_figures"
os.makedirs(outdir, exist_ok=True)

# ---- FIGURE 1: BOXPLOT REWARD ----
if emp_returns:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(NAMES))
    w = 0.25
    for j, (label, rets, color) in enumerate([
        ("Uniform", uni_returns, "#B0B0B0"),
        ("Empirical", emp_returns, "#0072B2"),
        ("Negotiation", nego_returns, "#D55E00"),
    ]):
        if not rets: continue
        vals = [[r.get(k, np.nan) for r in rets] for k in KEYS]
        bp = ax.boxplot(vals, positions=x + (j-1)*w, widths=w*0.8,
                       patch_artist=True, showfliers=False)
        for patch in bp["boxes"]: patch.set_facecolor(color); patch.set_alpha(0.7)
        for median in bp["medians"]: median.set_color("black")
        ax.plot([], [], "s", color=color, label=label, ms=8)
    ax.set_xticks(x)
    ax.set_xticklabels(NAMES)
    ax.set_ylabel("Final reward")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/boxplot_reward.pdf", bbox_inches="tight")
    plt.savefig(f"{outdir}/boxplot_reward.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved boxplot_reward")

# ---- FIGURE 2: BOXPLOT MITIGATION ----
fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
for li, lever in enumerate(LEVERS):
    ax = axes[li]
    x = np.arange(len(NAMES))
    w = 0.3
    for j, (label, pols, color) in enumerate([
        ("Empirical", emp_pols, "#0072B2"),
        ("Negotiation", nego_pols, "#D55E00"),
    ]):
        vals = [[effort(p, k, lever) for p in pols] for k in KEYS]
        bp = ax.boxplot(vals, positions=x + (j-0.5)*w, widths=w*0.8,
                       patch_artist=True, showfliers=False)
        for patch in bp["boxes"]: patch.set_facecolor(color); patch.set_alpha(0.7)
        for median in bp["medians"]: median.set_color("black")
        if li == 0:
            ax.plot([], [], "s", color=color, label=label, ms=8)
    ax.set_xticks(x)
    ax.set_xticklabels(NAMES, rotation=30, ha="right")
    ax.set_title(lever.capitalize(), fontsize=9, fontweight="bold")
    if li == 0: ax.set_ylabel("Effort")
axes[0].legend(frameon=False, loc="upper right")
plt.tight_layout()
plt.savefig(f"{outdir}/boxplot_mitigation.pdf", bbox_inches="tight")
plt.savefig(f"{outdir}/boxplot_mitigation.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved boxplot_mitigation")

# ---- FIGURE 3: TEMPERATURE TRAJECTORIES ----
fig, ax = plt.subplots(figsize=(6, 4))
for label, runs, color, ls in [
    ("Uniform", uni, "#B0B0B0", "--"),
    ("Empirical", emp, "#0072B2", "-"),
    ("Negotiation", nego, "#D55E00", "-"),
]:
    trajs = [temp_trajectory(d) for d in runs]
    if not trajs or not trajs[0]: continue
    minlen = min(len(t) for t in trajs)
    arr = np.array([t[:minlen] for t in trajs])
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    years = np.arange(len(mean))
    ax.plot(years, mean, color=color, ls=ls, lw=1.6, label=label)
    ax.fill_between(years, mean-std, mean+std, color=color, alpha=0.15)
ax.set_xlabel("Simulation year")
ax.set_ylabel("Temperature (\u00b0C)")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"{outdir}/temperature_trajectories.pdf", bbox_inches="tight")
plt.savefig(f"{outdir}/temperature_trajectories.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved temperature_trajectories")

# ---- FIGURE 4: SLOPE VS MITIGATION ----
beta = 0.2499
norm = 182.47
params = json.load(open("src/damage_parameters.json"))
T0 = 1.5
fig, ax = plt.subplots(figsize=(5, 3.8))
for name, key in zip(NAMES, KEYS):
    c = params["countries"][name]
    rate = c["baseline_annual_rate"]
    mu, sig = c["severity"]["mu"], c["severity"]["sigma"]
    d = rate * np.exp(beta*T0) * np.exp(mu + 0.5*sig**2) / norm
    slope = beta * d
    mit = sum(effort(emp_pols[0], key, l) for l in LEVERS)
    mit_mean = np.mean([sum(effort(p, key, l) for l in LEVERS) for p in emp_pols])
    ax.scatter(slope, mit_mean, s=80, color=COLORS[name], zorder=5)
    ax.annotate(name, (slope, mit_mean), textcoords="offset points",
               xytext=(8, 4), fontsize=8.5)
ax.set_xlabel("Damage slope at 1.5\u00b0C  ($\\beta \\cdot d_i$)", fontsize=9)
ax.set_ylabel("Total mitigation effort (mean over seeds)", fontsize=9)
ax.set_xlim(left=-0.3)
ax.set_ylim(bottom=-0.15)
plt.tight_layout()
plt.savefig(f"{outdir}/slope_vs_mitigation.pdf", bbox_inches="tight")
plt.savefig(f"{outdir}/slope_vs_mitigation.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved slope_vs_mitigation")

# ---- FIGURE 5: DAMAGE CURVES ----
def emp_damage(country, T):
    c = params["countries"][country]
    rate = c["baseline_annual_rate"] * np.exp(beta * T)
    mu, sig = c["severity"]["mu"], c["severity"]["sigma"]
    return rate * np.exp(mu + 0.5*sig**2) / norm

T = np.linspace(0.0, 3.0, 400)
emp_mean = np.mean([emp_damage(c, T) for c in NAMES], axis=0)
uni_curve = 0.003 * T**4 * 1000.0

fig, ax = plt.subplots(figsize=(5, 3.4))
ax.plot(T, uni_curve, color="0.25", lw=1.6, ls="--", label="Assumed  $0.003\\,T^{4}$")
ax.plot(T, emp_mean, color="#1F6FB2", lw=1.8, label="Empirical (four-country mean)")
ax.axvline(1.5, color="0.7", lw=0.8, ls=":", zorder=0)
ax.annotate("calibration point", xy=(1.5, 1), fontsize=7.5, color="0.45",
           xytext=(1.58, 1), va="center")
ax.set_xlabel("Temperature anomaly \u0394T (\u00b0C)")
ax.set_ylabel("Expected annual damage (reward units)")
ax.set_xlim(0, 3.0)
ax.set_ylim(bottom=0)
ax.legend(frameon=False, loc="upper left")
plt.tight_layout()
plt.savefig(f"{outdir}/fig_damage_curves.pdf", bbox_inches="tight")
plt.savefig(f"{outdir}/fig_damage_curves.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved fig_damage_curves")

print(f"\nCalibration check at T=1.5:")
print(f"  uniform  = {0.003*1.5**4*1000:.2f}")
print(f"  empirical mean = {np.mean([emp_damage(c, 1.5) for c in NAMES]):.2f}")
print(f"  ratio = {0.003*1.5**4*1000 / np.mean([emp_damage(c, 1.5) for c in NAMES]):.2f}")

print(f"\nAll figures saved to {outdir}/")
