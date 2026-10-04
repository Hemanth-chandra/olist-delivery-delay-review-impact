"""Build the order-level analysis table from the raw Olist CSV files.

One row = one delivered order that has a customer review.
All joins, filters and derived columns live here so the analysis is reproducible.
"""
from pathlib import Path

import numpy as np
import pandas as pd

MIN_CATEGORY_ORDERS = 500  # rarer categories are pooled into "other"
FOLLOW_UP_DAYS = 180       # window used to define "repeat purchase"


def _read(raw_dir: Path, name: str, **kw) -> pd.DataFrame:
    path = raw_dir / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the Olist CSVs and place them in {raw_dir}."
        )
    return pd.read_csv(path, **kw)


def build_order_table(raw_dir: str | Path = "data/raw"):
    """Return (orders_df, customers_df).

    orders_df    : delivered + reviewed orders with delay, controls and outcomes.
    customers_df : first order per customer with repeat-purchase follow-up (for survival).
    """
    raw = Path(raw_dir)

    ts_cols = [
        "order_purchase_timestamp",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    orders = _read(raw, "olist_orders_dataset.csv", parse_dates=ts_cols)
    customers = _read(raw, "olist_customers_dataset.csv")
    items = _read(raw, "olist_order_items_dataset.csv")
    products = _read(raw, "olist_products_dataset.csv")
    sellers = _read(raw, "olist_sellers_dataset.csv")
    pay = _read(raw, "olist_order_payments_dataset.csv")
    reviews = _read(
        raw, "olist_order_reviews_dataset.csv", parse_dates=["review_answer_timestamp"]
    )
    cat_tr = _read(raw, "product_category_name_translation.csv")

    data_end = orders["order_purchase_timestamp"].max()

    # ---- delivered orders only (delay is undefined otherwise) -------------
    d = orders[
        (orders["order_status"] == "delivered")
        & orders["order_delivered_customer_date"].notna()
    ].copy()
    d = d.merge(
        customers[["customer_id", "customer_unique_id", "customer_state"]],
        on="customer_id",
        how="left",
    )

    # ---- items: aggregate to order level, keep the main (priciest) item ----
    items = items.merge(
        products[["product_id", "product_category_name", "product_weight_g"]],
        on="product_id",
        how="left",
    ).merge(cat_tr, on="product_category_name", how="left")
    items["category"] = (
        items["product_category_name_english"].fillna("unknown").astype(str)
    )
    main_item = (
        items.sort_values("price", ascending=False)
        .drop_duplicates("order_id")[["order_id", "seller_id", "category"]]
    )
    agg = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        price_sum=("price", "sum"),
        freight_sum=("freight_value", "sum"),
        weight_g=("product_weight_g", "sum"),
    )
    items_o = main_item.merge(agg, on="order_id").merge(
        sellers[["seller_id", "seller_state"]], on="seller_id", how="left"
    )

    # ---- payments ---------------------------------------------------------
    pay_main = (
        pay.sort_values("payment_value", ascending=False)
        .drop_duplicates("order_id")[["order_id", "payment_type"]]
    )
    pay_agg = pay.groupby("order_id").agg(
        payment_value=("payment_value", "sum"),
        installments=("payment_installments", "max"),
    )
    pay_o = pay_main.merge(pay_agg, on="order_id")

    # ---- reviews: latest review per order ---------------------------------
    rev = (
        reviews.sort_values("review_answer_timestamp")
        .drop_duplicates("order_id", keep="last")
        .copy()
    )
    rev["comment"] = (
        rev["review_comment_title"].fillna("") + " " + rev["review_comment_message"].fillna("")
    ).str.strip()
    rev = rev[["order_id", "review_score", "comment"]]

    full = (
        d.merge(items_o, on="order_id", how="inner")
        .merge(pay_o, on="order_id", how="left")
        .merge(rev, on="order_id", how="inner")
    )

    # ---- derived columns --------------------------------------------------
    day = 86400.0
    full["delay_days"] = (
        full["order_delivered_customer_date"] - full["order_estimated_delivery_date"]
    ).dt.total_seconds() / day
    full["late"] = (full["delay_days"] > 0).astype(int)
    full["promised_days"] = (
        full["order_estimated_delivery_date"] - full["order_purchase_timestamp"]
    ).dt.total_seconds() / day
    full["actual_days"] = (
        full["order_delivered_customer_date"] - full["order_purchase_timestamp"]
    ).dt.total_seconds() / day
    full["same_state"] = (full["customer_state"] == full["seller_state"]).astype(int)
    full["freight_ratio"] = (full["freight_sum"] / full["price_sum"]).clip(0, 2)
    full["log_price"] = np.log1p(full["price_sum"])
    full["log_weight"] = np.log1p(full["weight_g"].fillna(full["weight_g"].median()))
    full["installments"] = full["installments"].fillna(1)
    full["payment_type"] = full["payment_type"].fillna("unknown")
    full["low_score"] = (full["review_score"] <= 2).astype(int)
    p = full["order_purchase_timestamp"]
    full["purchase_q"] = p.dt.year.astype(str) + "Q" + p.dt.quarter.astype(str)

    bins = [-np.inf, 0, 3, 7, 14, np.inf]
    labels = ["on_time_or_early", "late_0_3d", "late_3_7d", "late_7_14d", "late_14d_plus"]
    full["delay_bin"] = pd.cut(full["delay_days"], bins=bins, labels=labels).astype(str)

    counts = full["category"].value_counts()
    keep = counts[counts >= MIN_CATEGORY_ORDERS].index
    full["category_grp"] = np.where(full["category"].isin(keep), full["category"], "other")

    # ---- repeat purchase (customer level, first order only) ---------------
    cust = _build_customer_table(d, pay_o, full, data_end)
    full = full.merge(
        cust[["order_id", "is_first_order", "repeat_180d", "eligible_180d", "next_order_value"]],
        on="order_id",
        how="left",
    )
    for c in ["is_first_order", "repeat_180d", "eligible_180d"]:
        full[c] = full[c].fillna(0).astype(int)

    keep_cols = [
        "order_id", "customer_unique_id", "customer_state", "seller_id", "seller_state",
        "category", "category_grp", "n_items", "price_sum", "freight_sum", "payment_value",
        "installments", "payment_type", "delay_days", "late", "delay_bin", "promised_days",
        "actual_days", "same_state", "freight_ratio", "log_price", "log_weight",
        "review_score", "low_score", "comment", "purchase_q",
        "order_purchase_timestamp", "is_first_order", "repeat_180d", "eligible_180d",
        "next_order_value",
    ]
    return full[keep_cols].reset_index(drop=True), cust


def _build_customer_table(delivered, pay_o, full, data_end):
    """First delivered order per customer + time to next order (>= 1 day later)."""
    dd = delivered.merge(pay_o[["order_id", "payment_value"]], on="order_id", how="left")
    dd = dd.sort_values(["customer_unique_id", "order_purchase_timestamp"])
    first = dd.drop_duplicates("customer_unique_id", keep="first").copy()
    first = first.rename(
        columns={"order_purchase_timestamp": "first_ts", "payment_value": "first_value"}
    )

    later = dd[["customer_unique_id", "order_id", "order_purchase_timestamp", "payment_value"]].rename(
        columns={
            "order_id": "next_order_id",
            "order_purchase_timestamp": "next_ts",
            "payment_value": "next_order_value",
        }
    )
    m = first[["customer_unique_id", "order_id", "first_ts"]].merge(
        later, on="customer_unique_id"
    )
    # same-day orders are treated as split baskets, not a return visit
    m = m[m["next_ts"] >= m["first_ts"] + pd.Timedelta(days=1)]
    nxt = (
        m.sort_values("next_ts").drop_duplicates("customer_unique_id", keep="first")
        [["customer_unique_id", "next_ts", "next_order_value"]]
    )
    out = first[["customer_unique_id", "order_id", "first_ts"]].merge(
        nxt, on="customer_unique_id", how="left"
    )
    out["event"] = out["next_ts"].notna().astype(int)
    end = out["next_ts"].fillna(data_end)
    out["duration_days"] = (end - out["first_ts"]).dt.total_seconds() / 86400.0
    out["days_to_next"] = (out["next_ts"] - out["first_ts"]).dt.total_seconds() / 86400.0
    out["eligible_180d"] = (
        out["first_ts"] <= data_end - pd.Timedelta(days=FOLLOW_UP_DAYS)
    ).astype(int)
    out["repeat_180d"] = (out["days_to_next"] <= FOLLOW_UP_DAYS).astype(int)
    out["is_first_order"] = 1
    # attach review/delay info from the order table
    cols = ["order_id", "late", "low_score", "log_price", "freight_ratio", "promised_days",
            "category_grp", "customer_state", "payment_type", "purchase_q"]
    out = out.merge(full[cols], on="order_id", how="inner")
    return out
