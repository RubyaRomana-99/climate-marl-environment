import json
import numpy as np
import os

_PARAMS = None


def _load():
    global _PARAMS
    if _PARAMS is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "damage_parameters.json")
        if not os.path.exists(path):
            path = "damage_parameters.json"
        with open(path) as f:
            _PARAMS = json.load(f)
    return _PARAMS


def sample_damage(country, temp_anomaly, years=5, rng=None):
    """Stochastic per-country hazard damage.

    Two layers of randomness:
      1. Event occurrence (Poisson) and per-event loss (lognormal) - the underlying process.
      2. Francisco's request: the total damage is then treated as a draw from a
         Normal centred on the expected damage with a data-derived standard deviation,
         so the value the agent receives varies around the prediction.
    """
    p = _load()
    rng = rng or np.random.default_rng()
    c = p["countries"][country]
    veg = c.get("veg_term", 0.0)

    lam = np.exp(c["alpha"] + veg + p["beta_temp"] * temp_anomaly) * years
    n = rng.poisson(lam)
    if n == 0:
        process_loss = 0.0
        events = []
    else:
        mu, sigma = c["severity"]["mu"], c["severity"]["sigma"]
        losses = rng.lognormal(mu, sigma, n)
        hazards = list(c["hazard_shares"].keys())
        probs = np.array(list(c["hazard_shares"].values()))
        probs = probs / probs.sum()
        labels = rng.choice(hazards, size=n, p=probs)
        events = [{"hazard": h, "loss_meur": round(float(l), 1)} for h, l in zip(labels, losses)]
        process_loss = float(np.sum(losses))

    return {
        "country": country,
        "n_events": int(n),
        "total_loss_meur": round(process_loss, 1),
        "events": events,
    }


def sample_reward_damage(country, temp_anomaly, rng=None):
    """Return the reward-scale damage for ONE year, sampled stochastically.

    Implements Francisco's recipe directly:
        damage ~ Normal(mean = expected reward damage, std = data-derived error)
    Clipped at zero (damage cannot be negative).
    This is what the RL environment calls.
    """
    p = _load()
    rng = rng or np.random.default_rng()
    c = p["countries"][country]
    norm = float(p.get("reward_normalizer", 1.0))

    mean_reward = expected_annual_damage(country, temp_anomaly) / norm
    std_reward = float(c.get("annual_damage_reward_std", 0.0))

    if std_reward <= 0.0:
        return max(0.0, mean_reward)
    draw = rng.normal(mean_reward, std_reward)
    return float(max(0.0, draw))


def expected_annual_damage(country, temp_anomaly):
    p = _load()
    c = p["countries"][country]
    veg = c.get("veg_term", 0.0)
    lam = np.exp(c["alpha"] + veg + p["beta_temp"] * temp_anomaly)
    mean_loss = np.exp(c["severity"]["mu"] + 0.5 * c["severity"]["sigma"] ** 2)
    return float(lam * mean_loss)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    print("Stochastic reward damage (varies each draw at the SAME temperature):")
    for c in ["Germany", "Spain", "Poland", "Sweden"]:
        draws = [round(sample_reward_damage(c, 2.0, rng), 3) for _ in range(6)]
        print(f"  {c:10s} 2C draws: {draws}")