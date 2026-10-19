"""
Generate a realistic (but synthetic) portfolio of road construction projects.
Patterns roughly match common risk drivers in mountainous / variable-climate regions.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate(n: int = 820, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    regions = [
        "Absheron", "Ganja-Gazakh", "Shaki-Zagatala", "Lankaran",
        "Karabakh", "Nakhchivan", "Shirvan", "Quba-Khachmaz"
    ]
    terrains = ["flat", "hilly", "mountainous", "urban"]
    project_types = ["highway", "rural_road", "bridge", "urban_road", "tunnel_approach"]
    contractors = ["large", "medium", "small"]
    seasons = ["spring", "summer", "autumn", "winter"]

    data = {
        "project_id": [f"RD-{2020 + i // 110}-{i % 1100:04d}" for i in range(n)],
        "region": np.random.choice(regions, n, p=[0.18, 0.14, 0.12, 0.10, 0.15, 0.08, 0.13, 0.10]),
        "terrain": np.random.choice(terrains, n, p=[0.28, 0.30, 0.25, 0.17]),
        "project_type": np.random.choice(project_types, n, p=[0.30, 0.25, 0.15, 0.20, 0.10]),
        "contractor_size": np.random.choice(contractors, n, p=[0.28, 0.45, 0.27]),
        "planned_duration_days": np.random.randint(90, 700, n),
        "budget_mln_azn": np.round(np.random.lognormal(2.7, 0.65, n), 2),
        "start_season": np.random.choice(seasons, n),
        "num_utility_conflicts": np.random.poisson(1.8, n),
        "material_price_index": np.round(np.random.normal(100, 11, n), 1),
        "avg_monthly_rainfall_mm": np.round(np.random.gamma(2.0, 28, n), 1),
        "freeze_thaw_cycles": np.random.poisson(6, n),
        "previous_delay_rate": np.round(np.clip(np.random.beta(1.5, 7, n), 0, 0.5), 3),
        "design_complexity": np.random.choice(["low", "medium", "high"], n, p=[0.38, 0.45, 0.17]),
    }

    df = pd.DataFrame(data)

    # Latent risk score (logistic) — stronger but still noisy signal
    base = -2.15
    terrain_effect = df["terrain"].map({"flat": -0.7, "hilly": 0.0, "mountainous": 0.95, "urban": 0.45})
    type_effect = df["project_type"].map({
        "highway": 0.15, "rural_road": -0.35, "bridge": 0.55,
        "urban_road": 0.30, "tunnel_approach": 0.95
    })
    contractor_effect = df["contractor_size"].map({"large": -0.65, "medium": 0.0, "small": 0.70})
    season_effect = df["start_season"].map({"spring": 0.0, "summer": -0.40, "autumn": 0.15, "winter": 0.85})
    complexity_effect = df["design_complexity"].map({"low": -0.50, "medium": 0.0, "high": 0.70})

    score = (
        base
        + terrain_effect
        + type_effect
        + contractor_effect
        + season_effect
        + complexity_effect
        + 0.18 * df["num_utility_conflicts"]
        + 0.022 * (df["material_price_index"] - 100)
        + 0.005 * df["avg_monthly_rainfall_mm"]
        + 0.055 * df["freeze_thaw_cycles"]
        + 2.4 * df["previous_delay_rate"]
        - 0.0004 * df["planned_duration_days"]
        + np.random.normal(0, 0.45, n)
    )

    prob = 1 / (1 + np.exp(-score))
    df["delayed"] = (np.random.rand(n) < prob).astype(int)

    df["delay_days"] = 0
    mask = df["delayed"] == 1
    df.loc[mask, "delay_days"] = np.random.randint(10, 150, mask.sum())

    df["cost_overrun_pct"] = np.where(
        df["delayed"] == 1,
        np.round(
            np.clip(
                np.random.uniform(3, 35, n) + 0.11 * df["delay_days"] + np.random.normal(0, 3.5, n),
                0, 50
            ),
            1,
        ),
        np.round(np.random.uniform(-5, 5, n), 1),
    )

    # Intentional dirty records (cleaned in analysis)
    rng = np.random.default_rng(7)
    idx = rng.choice(df.index, 14, replace=False)
    df.loc[idx[:7], "avg_monthly_rainfall_mm"] = np.nan
    df.loc[idx[7:11], "region"] = "Absheron "
    df.loc[idx[11:], "region"] = "absheron"

    return df


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "road_projects.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows → {out}")
    print(f"Delay rate: {df['delayed'].mean()*100:.1f}%")
