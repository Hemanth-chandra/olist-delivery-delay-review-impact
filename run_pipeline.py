"""Run the full analysis and write everything the Streamlit app needs.

Usage:
    python run_pipeline.py                # expects the Olist CSVs in data/raw/
    python run_pipeline.py --raw path/to/csvs

Outputs go to data/processed/ (small files, safe to commit and deploy).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import analysis as A
from src.data_prep import build_order_table
from src.text_aspects import tag_aspects


def main(raw_dir: str, out_dir: str) -> None:
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("1/7 Building order table ...")
    df, cust = build_order_table(raw_dir)
    print(f"    {len(df):,} delivered+reviewed orders, {len(cust):,} first-order customers")

    print("2/7 Tagging review text ...")
    tags = tag_aspects(df["comment"])
    df = pd.concat([df.drop(columns=["comment"]), tags], axis=1)

    print("3/7 Stage A: effect of late delivery on low reviews ...")
    delay_res, effect_i, _ = A.fit_delay_effect(df)
    dose, effect_bin = A.dose_response(df)
    effect_i = effect_bin  # bin-specific effects drive rankings and totals
    fe = A.seller_fixed_effects_check(df)

    print("4/7 Stage B: effect of low review on repeat purchase ...")
    repeat_res, drop_draws = A.fit_repeat_effect(df)
    chain = A.revenue_chain(df, effect_i, repeat_res, drop_draws, delay_res)

    late_mask = df["late"] == 1
    severe = df["delay_days"] > 3
    delay_res["share_late_orders_over_3d"] = float((severe & late_mask).sum() / late_mask.sum())
    delay_res["excess_low_total_bin_model"] = float(effect_i.sum())
    delay_res["share_excess_low_from_over_3d"] = float(effect_i[severe.to_numpy()].sum() / effect_i.sum())

    print("5/7 Ranking fixes ...")
    per_low = chain["per_excess_low_score_brl"]
    by_cat = A.rank_segments(df, effect_i, per_low, "category_grp")
    by_state = A.rank_segments(df, effect_i, per_low, "customer_state")
    df["route"] = df["seller_state"].astype(str) + " -> " + df["customer_state"].astype(str)
    by_route = A.rank_segments(df, effect_i, per_low, "route", min_orders=200)
    by_seller = A.seller_ranking(df, effect_i, per_low)

    print("6/7 Survival layer ...")
    km, cox, surv_meta = A.survival_tables(cust)

    print("7/7 Text corroboration + experiment defaults ...")
    aspects = [c for c in tags.columns if c != "has_text"]
    txt = df[df["has_text"] == 1]
    rows = []
    for name, mask in [
        ("low score (1-2)", txt["low_score"] == 1),
        ("high score (4-5)", txt["review_score"] >= 4),
        ("late delivery", txt["late"] == 1),
        ("on-time delivery", txt["late"] == 0),
        ("late AND low score", (txt["late"] == 1) & (txt["low_score"] == 1)),
        ("on-time AND low score", (txt["late"] == 0) & (txt["low_score"] == 1)),
    ]:
        sub = txt[mask]
        r = {"group": name, "reviews_with_text": int(len(sub))}
        for a in aspects:
            r[a] = float(sub[a].mean()) if len(sub) else np.nan
        rows.append(r)
    aspect_df = pd.DataFrame(rows)

    exp_defaults = {
        "baseline_low_rate_late": delay_res["low_rate_late"],
        "baseline_late_rate": delay_res["late_rate"],
        "n_samples_example": A.sample_size_two_prop(delay_res["low_rate_late"], 0.15),
    }

    # ---- save ------------------------------------------------------------
    slim = df.drop(columns=["customer_unique_id", "seller_id"]).copy()
    slim["order_month"] = slim["order_purchase_timestamp"].dt.to_period("M").astype(str)
    slim = slim.drop(columns=["order_purchase_timestamp"])
    for c in ["delay_days", "promised_days", "actual_days", "freight_ratio", "log_price",
              "log_weight", "price_sum", "freight_sum", "payment_value", "next_order_value"]:
        slim[c] = slim[c].astype("float32")
    slim.to_parquet(out / "orders.parquet", index=False)

    by_cat.to_csv(out / "rank_category.csv", index=False)
    by_state.to_csv(out / "rank_state.csv", index=False)
    by_route.to_csv(out / "rank_route.csv", index=False)
    by_seller.to_csv(out / "rank_seller.csv", index=False)
    dose.to_csv(out / "dose_response.csv", index=False)
    km.to_csv(out / "km_curves.csv", index=False)
    cox.to_csv(out / "cox_hazard_ratios.csv", index=False)
    aspect_df.to_csv(out / "text_aspects.csv", index=False)

    results = {
        "dataset": {
            "orders_analyzed": int(len(df)),
            "first_order_customers": int(len(cust)),
            "date_min": slim["order_month"].min(),
            "date_max": slim["order_month"].max(),
            "share_with_text": float(df["has_text"].mean()),
        },
        "stage_a_delay_to_low_score": delay_res,
        "seller_fixed_effects_check": fe,
        "stage_b_low_score_to_repeat": repeat_res,
        "revenue_chain": chain,
        "survival_meta": surv_meta,
        "experiment_defaults": exp_defaults,
    }
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Done in {time.time() - t0:.0f}s. Files written to {out}/")
    print(json.dumps({k: results[k] for k in ["stage_a_delay_to_low_score", "revenue_chain"]}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()
    main(args.raw, args.out)
