import json
import numpy as np
from pathlib import Path

_GWP100_CH4           = 28.0
_GWP100_N2O           = 265.0
_N2O_N_RATIO          = 44.0 / 28.0
_MT_CO2_TO_PG_C       = 1.0 / 3664.0
_EU_BASELINE_AGRILAND = 42.705
_EU_BASELINE_FOREST   = 37.2
_MEAN_YEAR            = 2005.5

CONTROLLED_GAS_NAMES = ["CO2_FF", "CO2_AFOLU", "CH4", "N2O", "SO2"]

_layered_params = None


def _load_layered_params(params_path="layered_parameters.json"):
    global _layered_params
    if _layered_params is None:
        p = Path(params_path)
        if p.exists():
            with open(p) as f:
                _layered_params = json.load(f)
    return _layered_params


def _get_country_params(gas, country, params_path="layered_parameters.json"):
    params = _load_layered_params(params_path)
    if params is None:
        return None
    return (
        params
        .get("equations", {})
        .get(gas, {})
        .get("country_results", {})
        .get(country)
    )


def _get_afolu_forest_elasticity(params_path="layered_parameters.json"):
    params = _load_layered_params(params_path)
    if params is None:
        return -0.355
    return (
        params
        .get("afolu_structural", {})
        .get("country_fe_benchmark", {})
        .get("ForestArea", {})
        .get("elasticity", -0.355)
    )


def _apply_regression(country_params, state_log_values, year=2015):
    if country_params is None:
        return None
    intercept    = country_params.get("intercept")
    elasticities = country_params.get("elasticities", {})
    if intercept is None:
        return None

    ln_pred = float(intercept)
    for var, ln_val in state_log_values.items():
        coef = elasticities.get(var, {}).get("elasticity", 0.0)
        if ln_val is not None and np.isfinite(ln_val):
            ln_pred += coef * ln_val
    ln_pred += elasticities.get("trend", {}).get("elasticity", 0.0) * (year - _MEAN_YEAR)
    return float(np.exp(ln_pred))


def _safe_log(value):
    if value is not None and value > 0:
        return float(np.log(value))
    return None


def _predict_ch4(country, state, year=2015, params_path="layered_parameters.json"):
    log_vals = {
        "Population": _safe_log(state.get("Population")),
        "AgriLand":   _safe_log(state.get("AgriLand")),
        "GDP_PC":     _safe_log(state.get("GDP_PC")),
    }
    ch4_co2eq = _apply_regression(_get_country_params("CH4", country, params_path), log_vals, year)
    if ch4_co2eq is not None and ch4_co2eq > 0:
        return ch4_co2eq / _GWP100_CH4
    return max(float(state.get("Methane", 0.0)), 0.0) / _GWP100_CH4


def _predict_n2o(country, state, year=2015, params_path="layered_parameters.json"):
    log_vals = {
        "AgriLand":   _safe_log(state.get("AgriLand")),
        "Population": _safe_log(state.get("Population")),
    }
    n2o_co2eq = _apply_regression(_get_country_params("N2O", country, params_path), log_vals, year)
    if n2o_co2eq is not None and n2o_co2eq > 0:
        return (n2o_co2eq / _GWP100_N2O) / _N2O_N_RATIO
    return None


def _predict_co2_afolu(state, baseline_afolu_Pg_C, co2_total_Mt=None,
                        params_path="layered_parameters.json"):
    forest_elasticity = _get_afolu_forest_elasticity(params_path)
    agriland    = float(state.get("AgriLand",   _EU_BASELINE_AGRILAND))
    forest_area = float(state.get("ForestArea", _EU_BASELINE_FOREST))

    agri_scale   = agriland   / max(_EU_BASELINE_AGRILAND, 1.0)
    forest_scale = (forest_area / max(_EU_BASELINE_FOREST, 1.0)) ** forest_elasticity
    afolu_Pg_C   = baseline_afolu_Pg_C * agri_scale * forest_scale

    if co2_total_Mt is not None and co2_total_Mt > 0:
        afolu_Pg_C = min(float(afolu_Pg_C), co2_total_Mt * _MT_CO2_TO_PG_C * 0.15)
    return max(float(afolu_Pg_C), 0.0)


def disaggregate_to_gas_vector(
    country,
    state,
    co2_total_Mt,
    baseline_vector,
    gas_names=None,
    year=2015,
    params_path="layered_parameters.json",
):
    if gas_names is None:
        gas_names = CONTROLLED_GAS_NAMES

    idx    = {g: i for i, g in enumerate(gas_names)}
    result = baseline_vector.copy().astype(np.float64)

    ch4_tg     = _predict_ch4(country, state, year, params_path)
    afolu_Pg_C = _predict_co2_afolu(state, baseline_vector[idx["CO2_AFOLU"]], co2_total_Mt, params_path)
    n2o_tg_n   = _predict_n2o(country, state, year, params_path)

    result[idx["CH4"]]       = max(ch4_tg, 0.0)
    result[idx["CO2_AFOLU"]] = afolu_Pg_C

    if n2o_tg_n is not None:
        result[idx["N2O"]] = max(n2o_tg_n, 0.0)
    else:
        baseline_co2_Mt    = (baseline_vector[idx["CO2_FF"]] + baseline_vector[idx["CO2_AFOLU"]]) / _MT_CO2_TO_PG_C
        result[idx["N2O"]] = baseline_vector[idx["N2O"]] * (co2_total_Mt / max(baseline_co2_Mt, 1.0))

    baseline_co2_Mt    = (baseline_vector[idx["CO2_FF"]] + baseline_vector[idx["CO2_AFOLU"]]) / _MT_CO2_TO_PG_C
    result[idx["SO2"]] = baseline_vector[idx["SO2"]] * (co2_total_Mt / max(baseline_co2_Mt, 1.0))

    ch4_co2eq_Mt          = ch4_tg * _GWP100_CH4
    afolu_Mt              = afolu_Pg_C / _MT_CO2_TO_PG_C
    result[idx["CO2_FF"]] = max(co2_total_Mt - ch4_co2eq_Mt - afolu_Mt, 0.0) * _MT_CO2_TO_PG_C

    return result.astype(np.float32)


def get_baseline_gas_vector(emission_data, year=2015, gas_names=None):
    if gas_names is None:
        gas_names = CONTROLLED_GAS_NAMES
    if year not in emission_data.index:
        raise ValueError(
            f"Year {year} not found. "
            f"Available: {int(emission_data.index.min())}–{int(emission_data.index.max())}"
        )
    row = emission_data.loc[year]
    return np.array(
        [float(row[g]) if g in row.index else 0.0 for g in gas_names],
        dtype=np.float32,
    )