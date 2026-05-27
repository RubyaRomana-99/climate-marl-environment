import json
import numpy as np
from pathlib import Path

_PARAMS           = None
_LAYERED_PARAMS   = None
_PARAMS_PATH      = Path(__file__).parent / "hierarchical_parameters.json"
_RL_PATH          = Path(__file__).parent / "rl_parameters.json"
_LAYERED_PATH     = Path(__file__).parent / "layered_parameters.json"

MEAN_YEAR = 2005.5

_POPULATION_2021 = {
    "Austria":         8955797,  "Belgium":        11586195,
    "Bulgaria":        6507301,  "Croatia":         3878981,
    "Cyprus":          1317309,  "Czechia":        10505772,
    "Denmark":         5856733,  "Estonia":         1330932,
    "Finland":         5541017,  "France":         67842811,
    "Germany":        83196078,  "Greece":         10569207,
    "Hungary":         9630932,  "Ireland":         5110585,
    "Italy":          59133173,  "Latvia":          1884490,
    "Lithuania":       2808380,  "Luxembourg":       640064,
    "Malta":            518150,  "Netherlands":    17533044,
    "Norway":          5408320,  "Poland":         36981559,
    "Portugal":       10361831,  "Romania":        19122059,
    "Slovenia":        2108079,  "Spain":          47443821,
    "Sweden":         10415811,  "Switzerland":     8704546,
    "United Kingdom": 66984000,
}

_DEFAULT_POOLED_MAPPINGS = {
    "e": {"level_to_delta": {"0.0": 0.0,   "0.5": 0.35,   "1.0": 0.7}},
    "m": {"level_to_delta": {"0.0": 0.0,   "0.5": -0.161, "1.0": -0.3219}},
    "l": {"level_to_delta": {"0.0": 0.0,   "0.5": -0.099, "1.0": -0.1978}},
    "p": {"level_to_delta": {"0.0": 0.0,   "0.03": 0.285, "0.08": 0.7601}},
}

_DEFAULT_GERMANY_MAPPINGS = {
    "e": {"level_to_delta": {"0.0": 0.0,   "0.5": 0.4,    "1.0": 0.8}},
    "m": {"level_to_delta": {"0.0": 0.0,   "0.5": -1.242, "1.0": -2.484}},
    "l": {"level_to_delta": {"0.0": 0.0,   "0.5": -0.093, "1.0": -0.186}},
    "p": {"level_to_delta": {"0.0": 0.0,   "0.03": 0.290, "0.08": 0.773}},
}

STATE_BOUNDS = {
    "Renewables": (0.0, 100.0),
    "Methane":    (0.0, None),
    "AgriLand":   (0.0, 100.0),
    "Investment": (0.0, None),
    "EnergyPC":   (0.0, None),
    "GDP_PC":     (0.0, None),
}


def _load_params():
    global _PARAMS
    if _PARAMS is None:
        with open(_PARAMS_PATH) as f:
            _PARAMS = json.load(f)
        if _RL_PATH.exists():
            with open(_RL_PATH) as f:
                rl = json.load(f)
            if "action_mappings" in rl:
                _PARAMS["action_mappings"] = rl["action_mappings"]
    return _PARAMS


def _load_layered_params():
    global _LAYERED_PARAMS
    if _LAYERED_PARAMS is None and _LAYERED_PATH.exists():
        with open(_LAYERED_PATH) as f:
            _LAYERED_PARAMS = json.load(f)
    return _LAYERED_PARAMS


def _clip(value, var):
    lo, hi = STATE_BOUNDS.get(var, (None, None))
    if lo is not None:
        value = max(value, lo)
    if hi is not None:
        value = min(value, hi)
    return value


def _safe_log(value):
    if value is not None and float(value) > 0:
        return float(np.log(float(value)))
    return None


def _apply_layered_equation(gas, country, log_values, year):
    params = _load_layered_params()
    if params is None:
        return None
    country_params = (
        params.get("equations", {})
              .get(gas, {})
              .get("country_results", {})
              .get(country)
    )
    if country_params is None:
        return None
    intercept    = country_params.get("intercept")
    elasticities = country_params.get("elasticities", {})
    if intercept is None:
        return None

    ln_pred = float(intercept)
    for var, ln_val in log_values.items():
        coef = elasticities.get(var, {}).get("elasticity", 0.0)
        if ln_val is not None and np.isfinite(ln_val):
            ln_pred += coef * float(ln_val)
    ln_pred += elasticities.get("trend", {}).get("elasticity", 0.0) * (year - MEAN_YEAR)

    result = float(np.exp(ln_pred))
    return result if result > 0 else None


def predict_co2(country, state):
    params       = _load_params()
    elasticities = params["country_elasticities"][country]
    intercept    = params["country_intercepts"][country]
    all_coefs    = params.get("country_all_coefficients", {}).get(country, {})

    var_map = {
        "Renewables": ("e",  "ln_Renewables"),
        "Methane":    ("m",  "ln_Methane"),
        "AgriLand":   ("l",  "ln_AgriLand"),
        "Investment": ("p",  "ln_Investment"),
        "GDP_PC":     (None, "ln_GDP_PC"),
        "EnergyPC":   (None, "ln_EnergyPC"),
        "Industry":   (None, "ln_Industry"),
        "Urban":      (None, "ln_Urban"),
    }

    ln_co2 = intercept
    for var, (action, coef_key) in var_map.items():
        value = state.get(var)
        if value is None or value <= 0:
            raise ValueError(f"Invalid state value for '{var}' in '{country}': {value}.")
        coef    = (elasticities.get(action, {}).get("elasticity", 0.0) if action
                   else all_coefs.get(coef_key.replace("ln_", ""), {}).get("coefficient", 0.0))
        ln_co2 += coef * np.log(value)

    ln_co2 += all_coefs.get("trend", {}).get("coefficient", 0.0) * (min(state.get("year", 2021), 2021) - MEAN_YEAR)
    return round(float(np.exp(ln_co2)), 4)


def apply_action(country, state, actions, stochastic=True, seed=None):
    if seed is not None:
        np.random.seed(seed)

    params   = _load_params()
    am       = params.get("action_mappings", {})
    mappings = (
        am.get("country_specific", {}).get(country)
        or am.get("pooled")
        or (_DEFAULT_GERMANY_MAPPINGS if country == "Germany" else _DEFAULT_POOLED_MAPPINGS)
    )

    constraints   = params.get("constraint_relationships", {})
    ren_to_inv    = constraints.get("Renewables_to_Investment", {}).get("coefficient", 0.0082)
    inv_to_gdp    = constraints.get("Investment_to_GDP_PC",     {}).get("coefficient", -0.2498)
    gdp_to_eng    = constraints.get("GDP_PC_to_EnergyPC",       {}).get("coefficient", 0.3373)
    country_elast = params.get("country_elasticities", {}).get(country, {})
    EXPECTED_SIGN = {"e": -1, "m": 1, "l": 1, "p": -1}

    def effective_delta(action, delta):
        elast    = country_elast.get(action, {}).get("elasticity", 0.0)
        expected = EXPECTED_SIGN.get(action, 0)
        if elast == 0.0:
            return delta
        if expected != 0 and np.sign(elast) != np.sign(expected):
            return 0.0
        return delta

    new_state         = dict(state)
    new_state["year"] = state.get("year", 2021) + 1
    current_ren       = state.get("Renewables", 0.0)
    current_inv       = state.get("Investment", 0.0)
    current_gdp       = state.get("GDP_PC",     0.0)

    for action, var in {"e": "Renewables", "m": "Methane", "l": "AgriLand", "p": "Investment"}.items():
        level     = actions.get(action, 0.0)
        raw_delta = mappings.get(action, {}).get("level_to_delta", {}).get(str(float(level)), 0.0)
        delta     = effective_delta(action, raw_delta)
        current   = new_state.get(var, 0.0)

        if var == "Renewables":
            change = delta * max(0.1, 1.0 - current / 100.0) + ren_to_inv * np.log1p(current_inv) * level
        elif var == "Methane":
            change = delta * (current / (current + 10.0)) + (-gdp_to_eng * 0.001 * current_ren * level)
        elif var == "AgriLand":
            change = 0.5 * delta * (current / (current + 10.0))
        elif var == "Investment":
            change = delta * (1.0 + abs(inv_to_gdp) * np.log1p(current_gdp / 10000))
        else:
            change = delta

        if stochastic:
            change += np.random.normal(0, 0.005)
        new_state[var] = _clip(current + change, var)

    return new_state


def get_starting_state(country):
    params   = _load_params()
    sv       = params["starting_values_2021"].get(country)
    if sv is None:
        raise ValueError(f"No starting values found for {country}.")
    levers   = sv.get("policy_levers", {})
    controls = sv.get("controls", {})
    return {
        "Renewables": levers.get("e", {}).get("value"),
        "Methane":    levers.get("m", {}).get("value"),
        "AgriLand":   levers.get("l", {}).get("value"),
        "Investment": levers.get("p", {}).get("value"),
        "GDP_PC":     controls.get("GDP_PC"),
        "EnergyPC":   controls.get("EnergyPC"),
        "Industry":   controls.get("Industry"),
        "Urban":      controls.get("Urban"),
        "year":       2021,
    }


def get_emission_share(country, current_co2_by_country):
    total = sum(current_co2_by_country.values())
    if total == 0:
        return 0.0
    return round(current_co2_by_country[country] / total, 6)


def get_baseline_co2(country):
    return _load_params()["co2_baselines"][country]["value_2021_MtCO2eq"]


def predict_co2_total(country, state, year=2021):
    energy_pc  = state.get("EnergyPC")
    renewables = state.get("Renewables")
    population = state.get("Population") or _POPULATION_2021.get(country)

    fossil_energy_pc = (
        energy_pc * (1.0 - max(min(renewables, 100.0), 0.0) / 100.0)
        if energy_pc and renewables is not None and energy_pc > 0
        else energy_pc
    )

    return _apply_layered_equation("CO2_total", country, {
        "Population":     _safe_log(population),
        "GDP_PC":         _safe_log(state.get("GDP_PC")),
        "FossilEnergyPC": _safe_log(fossil_energy_pc),
        "Industry":       _safe_log(state.get("Industry")),
    }, year)


def predict_ch4(country, state, year=2021):
    return _apply_layered_equation("CH4", country, {
        "Population": _safe_log(state.get("Population") or _POPULATION_2021.get(country)),
        "AgriLand":   _safe_log(state.get("AgriLand")),
        "GDP_PC":     _safe_log(state.get("GDP_PC")),
    }, year)


def predict_n2o(country, state, year=2021):
    return _apply_layered_equation("N2O", country, {
        "AgriLand":   _safe_log(state.get("AgriLand")),
        "Population": _safe_log(state.get("Population") or _POPULATION_2021.get(country)),
    }, year)


if __name__ == "__main__":
    country = "Germany"
    state   = get_starting_state(country)
    co2     = predict_co2(country, state)
    actual  = get_baseline_co2(country)
    print(f"predict_co2: {co2} MtCO2eq  (actual {actual}, error {round(abs(co2-actual),2)})")
    print()
    for c in ["Germany", "France", "Poland", "Sweden"]:
        s   = get_starting_state(c)
        c_s = f"{predict_co2_total(c,s):.1f}" if predict_co2_total(c,s) else "N/A"
        m_s = f"{predict_ch4(c,s):.3f}"       if predict_ch4(c,s)       else "N/A"
        n_s = f"{predict_n2o(c,s):.3f}"       if predict_n2o(c,s)       else "N/A"
        print(f"  {c:<15} CO2={c_s:>8}  CH4={m_s:>8}  N2O={n_s:>8}")