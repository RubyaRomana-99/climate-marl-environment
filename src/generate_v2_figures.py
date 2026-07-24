"""
Publication-quality figures for the corrected (v2) results.
Run on the cluster:  python src/generate_v2_figures.py
"""
import json, glob, os, re, ast
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "font.size": 10,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

C_UNI  = "#8C8C8C"
C_EMP  = "#2166AC"
C_NEGO = "#B2182B"
C_GER  = "#2166AC"
C_POL  = "#4DAC26"
C_SPA  = "#D6604D"
C_SWE  = "#9970AB"
COUNTRY_COLORS = {"Germany": C_GER, "Poland": C_POL, "Spain": C_SPA, "Sweden": C_SWE}

NAMES  = ["Germany", "Poland", "Spain", "Sweden"]
KEYS   = ["country_0", "country_1", "country_2", "country_3"]
LEVERS = ["energy", "methane", "agriculture"]

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

emp  = load_runs("data/*/marl_results/*v2_empirical_ds*/marl_experiment_results.json")
nego = load_runs("data/*/marl_results/*v2_nego_ds*/marl_experiment_results.json")
uni_files = sorted(glob.glob("data/*/marl_results/*uniform_ds*/marl_experiment_results.json"))
if not uni_files:
    uni_files = sorted(glob.glob("data/*/marl_results/*heterogenous*/marl_experiment_results.json"))
uni = [json.load(open(f)) for f in uni_files[:5]] if uni_files else []
print(f"Loaded: {len(emp)} empirical, {len(nego)} negotiation, {len(uni)} uniform")

emp_pols  = [last_policy(d) for d in emp]
nego_pols = [last_policy(d) for d in nego]

logfiles = sorted(glob.glob("rerun_*.out"))
emp_returns, nego_returns, uni_returns = [], [], []
if logfiles:
    all_ret = extract_returns(max(logfiles, key=os.path.getmtime), 10)
    emp_returns  = all_ret[0::2]
    nego_returns = all_ret[1::2]

outdir = "v2_figures"
os.makedirs(outdir, exist_ok=True)

# ======================================================================
# FIGURE 1: REWARD — grouped bar chart, not boxplot
# ======================================================================
if emp_returns:
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    x = np.arange(len(NAMES))
    w = 0.28

    emp_means  = [np.mean([r[k] for r in emp_returns]) for k in KEYS]
    emp_stds   = [np.std([r[k] for r in emp_returns]) for k in KEYS]
    nego_means = [np.mean([r[k] for r in nego_returns]) for k in KEYS]
    nego_stds  = [np.std([r[k] for r in nego_returns]) for k in KEYS]

    bars1 = ax.bar(x - w/2, emp_means, w, yerr=emp_stds, capsize=3,
                   color=C_EMP, alpha=0.8, edgecolor="white", linewidth=0.5,
                   label="Empirical", error_kw=dict(lw=0.8))
    bars2 = ax.bar(x + w/2, nego_means, w, yerr=nego_stds, capsize=3,
                   color=C_NEGO, alpha=0.8, edgecolor="white", linewidth=0.5,
                   label="Negotiation", error_kw=dict(lw=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(NAMES)
    ax.set_ylabel("Final reward")
    ax.axhline(0, color="0.8", lw=0.5, zorder=0)
    ax.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{outdir}/boxplot_reward.pdf")
    plt.savefig(f"{outdir}/boxplot_reward.png", dpi=300)
    plt.close()
    print("Saved boxplot_reward")

# ======================================================================
# FIGURE 2: MITIGATION — grouped bars per lever, side by side
# ======================================================================
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
for li, lever in enumerate(LEVERS):
    ax = axes[li]
    x = np.arange(len(NAMES))
    w = 0.35

    emp_vals  = [np.mean([effort(p, k, lever) for p in emp_pols]) for k in KEYS]
    emp_errs  = [np.std([effort(p, k, lever) for p in emp_pols]) for k in KEYS]
    nego_vals = [np.mean([effort(p, k, lever) for p in nego_pols]) for k in KEYS]
    nego_errs = [np.std([effort(p, k, lever) for p in nego_pols]) for k in KEYS]

    ax.bar(x - w/2, emp_vals, w, yerr=emp_errs, capsize=2.5,
           color=C_EMP, alpha=0.8, edgecolor="white", linewidth=0.5,
           error_kw=dict(lw=0.7))
    ax.bar(x + w/2, nego_vals, w, yerr=nego_errs, capsize=2.5,
           color=C_NEGO, alpha=0.8, edgecolor="white", linewidth=0.5,
           error_kw=dict(lw=0.7))

    ax.set_xticks(x)
    ax.set_xticklabels(NAMES, rotation=25, ha="right")
    ax.set_ylim(0, 1.15)
    ax.text(0.5, 0.95, lever.capitalize(), transform=ax.transAxes,
            ha="center", va="top", fontsize=10, fontweight="bold")
    if li == 0:
        ax.set_ylabel("Effort")

handles = [Patch(facecolor=C_EMP, alpha=0.8, label="Empirical"),
           Patch(facecolor=C_NEGO, alpha=0.8, label="Negotiation")]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, 1.02))
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"{outdir}/boxplot_mitigation.pdf")
plt.savefig(f"{outdir}/boxplot_mitigation.png", dpi=300)
plt.close()
print("Saved boxplot_mitigation")

# ======================================================================
# FIGURE 3: TEMPERATURE TRAJECTORIES — clean lines with shading
# ======================================================================
fig, ax = plt.subplots(figsize=(5.5, 3.8))
for label, runs, color, ls, lw in [
    ("Uniform $T^4$", uni, C_UNI, "--", 1.2),
    ("Empirical", emp, C_EMP, "-", 1.6),
    ("Empirical + agreements", nego, C_NEGO, "-", 1.6),
]:
    trajs = [temp_trajectory(d) for d in runs]
    if not trajs or not trajs[0]: continue
    minlen = min(len(t) for t in trajs)
    arr = np.array([t[:minlen] for t in trajs])
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    years = np.arange(len(mean))
    ax.plot(years, mean, color=color, ls=ls, lw=lw, label=label)
    ax.fill_between(years, mean-std, mean+std, color=color, alpha=0.1)
ax.set_xlabel("Simulation year")
ax.set_ylabel("Temperature (\u00b0C)")
ax.legend(frameon=False, fontsize=8.5)
plt.tight_layout()
plt.savefig(f"{outdir}/temperature_trajectories.pdf")
plt.savefig(f"{outdir}/temperature_trajectories.png", dpi=300)
plt.close()
print("Saved temperature_trajectories")

# ======================================================================
# FIGURE 4: SLOPE VS MITIGATION — labeled scatter with trend
# ======================================================================
beta = 0.2499
norm = 182.47
params = json.load(open("src/damage_parameters.json"))
T0 = 1.5

fig, ax = plt.subplots(figsize=(5, 3.8))
slopes, mits = [], []
for name, key in zip(NAMES, KEYS):
    c = params["countries"][name]
    rate = c["baseline_annual_rate"]
    mu, sig = c["severity"]["mu"], c["severity"]["sigma"]
    d = rate * np.exp(beta*T0) * np.exp(mu + 0.5*sig**2) / norm
    slope = beta * d
    mit_mean = np.mean([sum(effort(p, key, l) for l in LEVERS) for p in emp_pols])
    slopes.append(slope)
    mits.append(mit_mean)
    ax.scatter(slope, mit_mean, s=100, color=COUNTRY_COLORS[name],
               edgecolors="white", linewidth=0.8, zorder=5)
    offset = (10, 6) if name != "Sweden" else (10, -10)
    ax.annotate(name, (slope, mit_mean), textcoords="offset points",
               xytext=offset, fontsize=9, color=COUNTRY_COLORS[name],
               fontweight="bold")

xs = np.array(slopes)
ys = np.array(mits)
m, b = np.polyfit(xs, ys, 1)
xline = np.linspace(0, max(xs)*1.1, 100)
ax.plot(xline, m*xline + b, color="0.7", ls="--", lw=0.8, zorder=1)
r = np.corrcoef(xs, ys)[0,1]
ax.text(0.95, 0.05, f"$r = {r:.2f}$", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9, color="0.5")

ax.set_xlabel("Damage slope at 1.5\u00b0C  ($\\beta \\cdot d_i$)")
ax.set_ylabel("Total mitigation effort")
ax.set_xlim(left=-0.3)
ax.set_ylim(bottom=-0.15)
plt.tight_layout()
plt.savefig(f"{outdir}/slope_vs_mitigation.pdf")
plt.savefig(f"{outdir}/slope_vs_mitigation.png", dpi=300)
plt.close()
print("Saved slope_vs_mitigation")

# ======================================================================
# FIGURE 5: DAMAGE CURVES — assumed vs empirical
# ======================================================================
def emp_damage(country, T):
    c = params["countries"][country]
    rate = c["baseline_annual_rate"] * np.exp(beta * T)
    mu, sig = c["severity"]["mu"], c["severity"]["sigma"]
    return rate * np.exp(mu + 0.5*sig**2) / norm

T = np.linspace(0.0, 3.0, 400)
emp_mean = np.mean([emp_damage(c, T) for c in NAMES], axis=0)
uni_curve = 0.003 * T**4 * 1000.0

fig, ax = plt.subplots(figsize=(5.5, 3.8))
ax.plot(T, uni_curve, color=C_UNI, lw=1.8, ls="--",
        label="Assumed  $0.003\\,T^{4}$")
ax.plot(T, emp_mean, color=C_EMP, lw=2.0,
        label="Empirical (four-country mean)")
ax.axvline(1.5, color="0.82", lw=0.7, ls=":", zorder=0)
ax.annotate("calibration\npoint", xy=(1.53, 2), fontsize=7.5,
            color="0.5", va="center")
ax.set_xlabel("Temperature anomaly $\\Delta T$ (\u00b0C)")
ax.set_ylabel("Expected annual damage (reward units)")
ax.set_xlim(0, 3.0)
ax.set_ylim(bottom=0)
ax.legend(frameon=False, loc="upper left")
plt.tight_layout()
plt.savefig(f"{outdir}/fig_damage_curves.pdf")
plt.savefig(f"{outdir}/fig_damage_curves.png", dpi=300)
plt.close()
print("Saved fig_damage_curves")

# ======================================================================
# CALIBRATION CHECK
# ======================================================================
print(f"\nCalibration check at T=1.5:")
print(f"  uniform  = {0.003*1.5**4*1000:.2f}")
emp_avg = np.mean([emp_damage(c, 1.5) for c in NAMES])
print(f"  empirical mean = {emp_avg:.2f}")
print(f"  ratio = {0.003*1.5**4*1000 / emp_avg:.2f}")

print(f"\nDamage slopes at T=1.5:")
for name in NAMES:
    c = params["countries"][name]
    d = c["baseline_annual_rate"]*np.exp(beta*1.5)*np.exp(c["severity"]["mu"]+0.5*c["severity"]["sigma"]**2)/norm
    print(f"  {name:9s} d={d:>6.2f}  slope={beta*d:>5.2f}")

print(f"\nAll figures saved to {outdir}/")
