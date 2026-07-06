import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

COUNTRIES = ["Germany", "Poland", "Spain", "Sweden"]
COLORS = {"Germany": "#1f4e79", "Poland": "#c00000",
          "Spain": "#2e7d32", "Sweden": "#e69500"}
KEYS = ["country_0", "country_1", "country_2", "country_3"]
NAME = dict(zip(KEYS, COUNTRIES))

def load(pat):
    return json.load(open(glob.glob(pat)[0]))

emp = load("data/*/marl_results/*empirical_damage/marl_experiment_results.json")
uni = load("data/*/marl_results/*uniform_t4/marl_experiment_results.json")

def series(d):
    steps = sorted(d["train_reward"].keys(), key=lambda x: int(x))
    x = [int(s) for s in steps]
    ys = {k: [d["train_reward"][s][k] for s in steps] for k in KEYS}
    return x, ys

# ---------- PLOT 1: reward curves side by side ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, d, title in [(axes[0], emp, "Empirical damage (country-specific)"),
                     (axes[1], uni, "Uniform T\u2074 (Anne's baseline)")]:
    x, ys = series(d)
    for k in KEYS:
        ax.plot(x, ys[k], color=COLORS[NAME[k]], linewidth=2.2, label=NAME[k])
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Training steps")
axes[0].set_ylabel("Reward per country")
axes[0].legend(frameon=False, fontsize=10, loc="lower right")
fig.suptitle("Agent reward during training", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("plot1_reward_curves.png", bbox_inches="tight")
plt.close()

# ---------- PLOT 2: final reward bar comparison ----------
def final(d):
    s = sorted(d["train_reward"].keys(), key=lambda x: int(x))[-1]
    return d["train_reward"][s]
ef, uf = final(emp), final(uni)

fig, ax = plt.subplots(figsize=(9, 5.5))
xpos = np.arange(4)
w = 0.38
emp_v = [ef[k] for k in KEYS]
uni_v = [uf[k] for k in KEYS]
b1 = ax.bar(xpos - w/2, emp_v, w, label="Empirical damage",
            color="#1f4e79", edgecolor="white", linewidth=0.8)
b2 = ax.bar(xpos + w/2, uni_v, w, label="Uniform T\u2074",
            color="#b0b7c0", edgecolor="white", linewidth=0.8)
ax.set_xticks(xpos)
ax.set_xticklabels(COUNTRIES)
ax.set_ylabel("Final reward")
ax.set_title("Final reward by country: empirical vs uniform damage",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(frameon=False, fontsize=10)
ax.axhline(0, color="#333", linewidth=0.8)
for b in list(b1) + list(b2):
    h = b.get_height()
    ax.annotate(f"{h:.1f}", (b.get_x() + b.get_width()/2, h),
                ha="center", va="top" if h < 0 else "bottom",
                fontsize=9, xytext=(0, -3 if h < 0 else 3),
                textcoords="offset points", color="#333")
plt.tight_layout()
plt.savefig("plot2_final_rewards.png", bbox_inches="tight")
plt.close()

# ---------- PLOT 3: spread comparison ----------
fig, ax = plt.subplots(figsize=(6, 5.5))
emp_spread = max(emp_v) - min(emp_v)
uni_spread = max(uni_v) - min(uni_v)
bars = ax.bar(["Empirical", "Uniform T\u2074"], [emp_spread, uni_spread],
              color=["#1f4e79", "#b0b7c0"], edgecolor="white",
              linewidth=1, width=0.55)
ax.set_ylabel("Reward spread (max \u2212 min)")
ax.set_title("Country heterogeneity in outcomes",
             fontsize=13, fontweight="bold", pad=12)
for b in bars:
    h = b.get_height()
    ax.annotate(f"{h:.2f}", (b.get_x() + b.get_width()/2, h),
                ha="center", va="bottom", fontsize=12, fontweight="bold",
                xytext=(0, 3), textcoords="offset points")
ax.text(0.5, 0.92, f"{emp_spread/uni_spread:.0f}\u00d7 larger spread",
        transform=ax.transAxes, ha="center", fontsize=12,
        color="#1f4e79", fontweight="bold")
plt.tight_layout()
plt.savefig("plot3_spread.png", bbox_inches="tight")
plt.close()

# ---------- PLOT 4: temperature trajectory ----------
def temp_traj(d):
    if "temperature_trajectory" not in d:
        return None
    t = d["temperature_trajectory"]
    if isinstance(t, dict):
        s = sorted(t.keys(), key=lambda x: int(x))[-1]
        return t[s]
    return t

te, tu = temp_traj(emp), temp_traj(uni)
if te and tu:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(range(len(te)), te, color="#c00000", linewidth=2.4,
            label="Empirical damage", marker="o", markersize=3)
    ax.plot(range(len(tu)), tu, color="#1f4e79", linewidth=2.4,
            label="Uniform T\u2074", marker="s", markersize=3)
    ax.set_xlabel("Year (simulation horizon)")
    ax.set_ylabel("Global temperature anomaly (\u00b0C)")
    ax.set_title("Temperature outcome under each damage model",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(frameon=False, fontsize=10)
    plt.tight_layout()
    plt.savefig("plot4_temperature.png", bbox_inches="tight")
    plt.close()
    print("Saved plot4_temperature.png")

print("Saved: plot1_reward_curves.png, plot2_final_rewards.png, plot3_spread.png")
print("Done. Download these 4 PNG files for your slides.")
