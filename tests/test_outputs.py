"""Sanity checks on the processed outputs. Run: pytest -q"""
import json
from pathlib import Path

import pandas as pd

PROC = Path("data/processed")


def test_files_exist():
    for f in ["orders.parquet", "results.json", "dose_response.csv", "km_curves.csv",
              "cox_hazard_ratios.csv", "rank_category.csv", "rank_state.csv",
              "rank_route.csv", "rank_seller.csv", "text_aspects.csv"]:
        assert (PROC / f).exists(), f


def test_rates_in_range():
    o = pd.read_parquet(PROC / "orders.parquet")
    assert 0.03 < o["late"].mean() < 0.2
    assert 0.05 < o["low_score"].mean() < 0.3
    assert o["review_score"].between(1, 5).all()
    assert o["order_id"].is_unique


def test_effect_direction():
    r = json.loads((PROC / "results.json").read_text())
    a = r["stage_a_delay_to_low_score"]
    assert a["odds_ratio"] > 1 and a["or_ci"][0] > 1
    d = pd.read_csv(PROC / "dose_response.csv")
    assert d.loc[d.delay_bin == "late_7_14d", "adj_odds_ratio"].iloc[0] > \
           d.loc[d.delay_bin == "late_0_3d", "adj_odds_ratio"].iloc[0]


def test_ranking_sorted():
    r = pd.read_csv(PROC / "rank_category.csv")
    assert r["excess_low_reviews"].is_monotonic_decreasing
