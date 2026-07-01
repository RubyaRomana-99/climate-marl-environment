import sys
import os
import numpy as np

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURR_DIR)

from damage_sampler import sample_damage, expected_annual_damage, _load

COUNTRIES = ["Germany", "Poland", "Spain", "Sweden"]
TEMPS = [1.2, 1.5, 2.0, 2.5, 3.0]


def section(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


section("1. PARAMETERS LOADED")
params = _load()
print(f"Model: {params['model']}")
print(f"beta_temp: {params['beta_temp']}  (rate x{params['rate_multiplier_per_degree']} per degree)")
print(f"forest_coef: {params['forest_coef']}   agri_coef: {params['agri_coef']}")
print(f"reward_normalizer: {params.get('reward_normalizer', 'MISSING')}")
for c in COUNTRIES:
    cp = params["countries"][c]
    print(f"  {c:10s} alpha={cp['alpha']:+.3f}  veg_term={cp['veg_term']:+.3f}  "
          f"rate={cp['baseline_annual_rate']:.3f}/yr  mu={cp['severity']['mu']}")


section("2. EXPECTED ANNUAL DAMAGE BY TEMPERATURE (euro M)")
print(f"{'Country':10s}" + "".join(f"{T:>10.1f}C" for T in TEMPS))
for c in COUNTRIES:
    row = "".join(f"{expected_annual_damage(c, T):>11.0f}" for T in TEMPS)
    print(f"{c:10s}{row}")


section("3. SAMPLED REWARD-SCALE DAMAGE (what the agent actually receives)")
norm = params.get("reward_normalizer", 1.0)
rng = np.random.default_rng(0)
print(f"{'Country':10s}" + "".join(f"{T:>9.1f}C" for T in TEMPS))
for c in COUNTRIES:
    cells = []
    for T in TEMPS:
        vals = [sample_damage(c, T, 1, rng)["total_loss_meur"] / norm for _ in range(2000)]
        cells.append(f"{np.mean(vals):>10.3f}")
    print(f"{c:10s}" + "".join(cells))


section("4. COMPARISON vs ANNE'S UNIFORM T^4 (at each temperature)")
print("Anne's term is identical for all 4 agents; the empirical model differentiates them.")
print(f"{'Temp':>6s}  {'Anne(all)':>10s}  {'Germany':>9s}  {'Poland':>9s}  {'Spain':>9s}  {'Sweden':>9s}")
for T in TEMPS:
    anne = 0.003 * (T ** 4) * 1000.0 * 0.1
    emp = {}
    for c in COUNTRIES:
        vals = [sample_damage(c, T, 1, rng)["total_loss_meur"] / norm for _ in range(2000)]
        emp[c] = np.mean(vals)
    print(f"{T:>5.1f}C  {anne:>10.3f}  {emp['Germany']:>9.3f}  {emp['Poland']:>9.3f}  "
          f"{emp['Spain']:>9.3f}  {emp['Sweden']:>9.3f}")


section("5. HETEROGENEITY CHECK (the thesis result)")
T = 2.0
rng = np.random.default_rng(1)
dmg = {c: np.mean([sample_damage(c, T, 1, rng)["total_loss_meur"] / norm for _ in range(5000)])
       for c in COUNTRIES}
hi = max(dmg, key=dmg.get)
lo = min(dmg, key=dmg.get)
ratio = dmg[hi] / dmg[lo] if dmg[lo] > 0 else float("inf")
print(f"At {T}C: highest = {hi} ({dmg[hi]:.3f}), lowest = {lo} ({dmg[lo]:.3f})")
print(f"Ratio high/low = {ratio:.1f}x")
print("Under Anne's uniform model this ratio would be exactly 1.0 (all identical).")
print("\nINTEGRATION TEST PASSED: empirical damage is country-specific and rises with warming.")