import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

warnings.filterwarnings("ignore")

COUNTRIES = [
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus",
    "Czechia", "Denmark", "Estonia", "Finland", "France",
    "Germany", "Greece", "Hungary", "Ireland", "Italy",
    "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands",
    "Norway", "Poland", "Portugal", "Romania", "Slovenia",
    "Spain", "Sweden", "Switzerland", "United Kingdom",
]

YEARS     = list(range(1990, 2022))
MEAN_YEAR = 2005.5
MIN_OBS   = 20

FILES = {
    "CO2":        "data/API_EN.GHG.CO2.MT.CE.AR5_DS2_en_csv_v2_7092.csv",
    "CH4":        "data/API_EN.GHG.CH4.MT.CE.AR5_DS2_en_csv_v2_12584.csv",
    "N2O":        "data/API_EN.GHG.N2O.MT.CE.AR5_DS2_en_csv_v2_22192.csv",
    "EnergyPC":   "data/API_EG.USE.PCAP.KG.OE_DS2_en_csv_v2_1837.csv",
    "Renewables": "data/API_EG.FEC.RNEW.ZS_DS2_en_csv_v2_4948.csv",
    "Industry":   "data/API_NV.IND.TOTL.ZS_DS2_en_csv_v2_36.csv",
    "GDP_PC":     "data/API_NY.GDP.PCAP.KD_DS2_en_csv_v2_82.csv",
    "Population": "data/API_SP.POP.TOTL_DS2_en_csv_v2_61.csv",
    "AgriLand":   "data/API_AG.LND.AGRI.ZS_DS2_en_csv_v2_696.csv",
    "ForestArea": "data/API_AG.LND.FRST.ZS_DS2_en_csv_v2_126986.csv",
}

EQUATION_SPECS = {
    "CO2_total": {
        "dependent":  "CO2",
        "regressors": ["Population", "GDP_PC", "FossilEnergyPC", "Industry"],
    },
    "CH4": {
        "dependent":  "CH4",
        "regressors": ["Population", "AgriLand", "GDP_PC"],
    },
    "N2O": {
        "dependent":  "N2O",
        "regressors": ["AgriLand", "Population"],
    },
}

AFOLU_STRUCTURAL_SPEC = {
    "dependent":  "CO2",
    "regressors": ["AgriLand", "ForestArea", "Population", "GDP_PC"],
}


def load_indicator(path, varname):
    df        = pd.read_csv(path, skiprows=4)
    year_cols = [str(y) for y in YEARS if str(y) in df.columns]
    df        = df[df["Country Name"].isin(COUNTRIES)][["Country Name"] + year_cols]
    df        = df.melt(id_vars="Country Name", var_name="Year", value_name=varname)
    df["Year"]  = df["Year"].astype(int)
    df[varname] = pd.to_numeric(df[varname], errors="coerce")
    return df.rename(columns={"Country Name": "Country"})


def build_panel():
    dfs = []
    for varname, path in FILES.items():
        if not Path(path).exists():
            print(f"  WARNING: {path} not found — {varname} skipped")
            continue
        dfs.append(load_indicator(path, varname))

    panel = dfs[0]
    for df in dfs[1:]:
        panel = panel.merge(df, on=["Country", "Year"], how="outer")

    panel = panel[panel["Country"].isin(COUNTRIES)]
    panel = panel.sort_values(["Country", "Year"]).reset_index(drop=True)
    panel["trend"] = panel["Year"] - MEAN_YEAR

    valid = (
        panel["EnergyPC"].notna()   &
        panel["Renewables"].notna() &
        (panel["EnergyPC"]   > 0)   &
        (panel["Renewables"] >= 0)  &
        (panel["Renewables"] <= 100)
    )
    panel.loc[valid, "FossilEnergyPC"] = (
        panel.loc[valid, "EnergyPC"] * (1.0 - panel.loc[valid, "Renewables"] / 100.0)
    )

    log_vars = list(set(
        v
        for spec in list(EQUATION_SPECS.values()) + [AFOLU_STRUCTURAL_SPEC]
        for v in [spec["dependent"]] + spec["regressors"]
    )) + ["FossilEnergyPC"]

    for var in log_vars:
        if var in panel.columns:
            mask = panel[var] > 0
            panel.loc[mask, f"ln_{var}"] = np.log(panel.loc[mask, var])

    return panel


def estimate_country_ols(country_data, dep_var, regressors):
    ln_dep  = f"ln_{dep_var}"
    ln_regs = [f"ln_{r}" for r in regressors] + ["trend"]
    clean   = country_data[[ln_dep] + ln_regs].dropna()

    if len(clean) < MIN_OBS:
        return None
    try:
        model = sm.OLS(clean[ln_dep], sm.add_constant(clean[ln_regs])).fit(cov_type="HC3")
    except Exception:
        return None

    result = {
        "n_obs":        int(len(clean)),
        "r2":           round(float(model.rsquared), 4),
        "r2_adj":       round(float(model.rsquared_adj), 4),
        "intercept":    round(float(model.params.get("const", np.nan)), 4),
        "elasticities": {},
    }
    for reg in ln_regs:
        coef = model.params.get(reg, np.nan)
        pval = model.pvalues.get(reg, np.nan)
        result["elasticities"][reg.replace("ln_", "")] = {
            "elasticity":  round(float(coef), 4),
            "pvalue":      round(float(pval), 4),
            "significant": float(pval) < 0.10,
        }
    return result


def estimate_country_fe(panel, dep_var, regressors):
    ln_dep  = f"ln_{dep_var}"
    ln_regs = [f"ln_{r}" for r in regressors] + ["trend"]
    clean   = panel[["Country", ln_dep] + ln_regs].dropna()

    dummies = pd.get_dummies(clean["Country"], drop_first=True)
    X       = sm.add_constant(pd.concat([clean[ln_regs], dummies], axis=1).astype(float))
    try:
        model = sm.OLS(clean[ln_dep], X).fit(cov_type="HC3")
    except Exception:
        return {}

    return {
        reg.replace("ln_", ""): {
            "elasticity":  round(float(model.params.get(reg, np.nan)), 4),
            "pvalue":      round(float(model.pvalues.get(reg, np.nan)), 4),
            "significant": float(model.pvalues.get(reg, 1.0)) < 0.10,
        }
        for reg in ln_regs
    }


def run_all_equations(panel):
    output = {
        "metadata": {
            "model":       "Layered Extended STIRPAT Framework",
            "countries":   COUNTRIES,
            "n_countries": len(COUNTRIES),
            "period":      "1990-2021",
        },
        "equations":        {},
        "afolu_structural": {},
    }

    for gas, spec in EQUATION_SPECS.items():
        dep  = spec["dependent"]
        regs = spec["regressors"]
        country_results, r2_list, skipped = {}, [], []

        for country in COUNTRIES:
            result = estimate_country_ols(panel[panel["Country"] == country], dep, regs)
            if result is None:
                skipped.append(country)
            else:
                country_results[country] = result
                r2_list.append(result["r2"])

        fe = estimate_country_fe(panel, dep, regs)

        print(f"\n{gas}")
        print(f"  Dependent  : ln({dep})")
        print(f"  Regressors : {regs}")
        print(f"  Estimated  : {len(country_results)} / {len(COUNTRIES)}")
        if skipped:
            print(f"  Skipped    : {skipped}")
        if r2_list:
            print(f"  Mean R²    : {np.mean(r2_list):.4f}")
            print(f"  R² range   : {np.min(r2_list):.4f} – {np.max(r2_list):.4f}")
        print("  Country FE benchmark:")
        for var, est in fe.items():
            sig = ("***" if est["pvalue"] < 0.01 else "**" if est["pvalue"] < 0.05
                   else "*" if est["pvalue"] < 0.10 else "ns")
            print(f"    {var:<22} {est['elasticity']:+.4f}  {sig}")

        output["equations"][gas] = {
            "dependent_variable":   dep,
            "regressors":           regs,
            "country_results":      country_results,
            "country_fe_benchmark": fe,
            "summary": {
                "n_countries_estimated": len(country_results),
                "n_skipped":             len(skipped),
                "skipped_countries":     skipped,
                "mean_r2": round(float(np.mean(r2_list)), 4) if r2_list else None,
                "min_r2":  round(float(np.min(r2_list)),  4) if r2_list else None,
                "max_r2":  round(float(np.max(r2_list)),  4) if r2_list else None,
            },
        }

    dep  = AFOLU_STRUCTURAL_SPEC["dependent"]
    regs = AFOLU_STRUCTURAL_SPEC["regressors"]
    fe   = estimate_country_fe(panel, dep, regs)

    print("\nStructural land use analysis (CO2_AFOLU disaggregation basis)")
    for var, est in fe.items():
        sig = ("***" if est["pvalue"] < 0.01 else "**" if est["pvalue"] < 0.05
               else "*" if est["pvalue"] < 0.10 else "ns")
        print(f"    {var:<22} {est['elasticity']:+.4f}  {sig}")

    output["afolu_structural"] = {
        "country_fe_benchmark": fe,
        "usage": "ForestArea and AgriLand elasticities inform CO2_AFOLU disaggregation.",
    }

    return output


def main():
    print("Layered Extended STIRPAT Framework")

    panel = build_panel()
    print(f"\nPanel: {panel['Country'].nunique()} countries, "
          f"{panel['Year'].min()}–{panel['Year'].max()}, {len(panel)} observations")

    results  = run_all_equations(panel)
    out_path = "layered_parameters.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSummary")
    for gas, eq in results["equations"].items():
        s = eq["summary"]
        print(f"  {gas:<15} n={s['n_countries_estimated']}  "
              f"mean_R²={s['mean_r2']}  range={s['min_r2']}–{s['max_r2']}")
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()