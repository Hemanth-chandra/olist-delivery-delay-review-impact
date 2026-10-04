"""Delivery Delay and Customer Reviews: Streamlit app.

The app only reads files in data/processed/ (built by run_pipeline.py),
so it deploys without the raw Olist CSVs.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scipy.stats import norm


def sample_size_two_prop(p_control, relative_lift, alpha=0.05, power=0.8, cuped_rho=0.0):
    """Orders needed per arm to detect a relative drop in a rate (two-proportion test)."""
    p_treat = p_control * (1 - relative_lift)
    h = abs(2 * np.arcsin(np.sqrt(p_control)) - 2 * np.arcsin(np.sqrt(p_treat)))
    n = 2 * ((norm.ppf(1 - alpha / 2) + norm.ppf(power)) / h) ** 2
    return float(np.ceil(n * (1 - cuped_rho ** 2)))

PROC = Path(__file__).parent / "data" / "processed"

st.set_page_config(
    page_title="Delivery Delay and Customer Reviews",
    layout="wide",
)

BIN_LABELS = {
    "on_time_or_early": "On time / early",
    "late_0_3d": "Late 0-3 days",
    "late_3_7d": "Late 3-7 days",
    "late_7_14d": "Late 7-14 days",
    "late_14d_plus": "Late 14+ days",
}
ASPECT_LABELS = {
    "delivery_delay": "Delivery delay",
    "not_received": "Not received",
    "product_quality": "Product quality",
    "wrong_or_incomplete": "Wrong / incomplete item",
    "packaging": "Packaging",
    "seller_service": "Seller service / refunds",
    "praise": "Praise",
}


@st.cache_data
def load():
    res = json.loads((PROC / "results.json").read_text())
    out = {"res": res}
    for name in ["dose_response", "km_curves", "cox_hazard_ratios", "text_aspects",
                 "rank_category", "rank_state", "rank_route", "rank_seller"]:
        out[name] = pd.read_csv(PROC / f"{name}.csv")
    return out


@st.cache_data
def load_orders():
    return pd.read_parquet(PROC / "orders.parquet")


if not (PROC / "results.json").exists():
    st.error("Processed files not found. Run `python run_pipeline.py` first (see README).")
    st.stop()

D = load()
R = D["res"]
_ta = D["text_aspects"].set_index("group")
NR_LATE = float(_ta.loc["late delivery", "not_received"])
NR_ONTIME = float(_ta.loc["on-time delivery", "not_received"])
TEXT_SHARE = R["dataset"]["share_with_text"]
_cox = D["cox_hazard_ratios"].set_index("covariate")
COX_LOW_HR = float(_cox.loc["low_score", "hazard_ratio"])
COX_LOW_P = float(_cox.loc["low_score", "p_value"])
A = R["stage_a_delay_to_low_score"]
B = R["stage_b_low_score_to_repeat"]
C = R["revenue_chain"]
FE = R["seller_fixed_effects_check"]

page = st.sidebar.radio(
    "Section",
    ["1. The answer", "2. Delay → bad reviews", "3. Fix list (what to fix first)",
     "4. Repeat purchase (survival)", "5. Experiment planner", "6. Explore the data",
     "7. Method & limits"],
)
st.sidebar.caption(
    "Data: Olist Brazilian e-commerce (real, 2016-2018, CC BY-NC-SA 4.0). "
    "Observational analysis: effects are estimates under stated assumptions."
)

pct = lambda x, d=1: f"{x * 100:.{d}f}%"

# ======================================================================
if page.startswith("1"):
    st.title("Delivery Delay and Customer Reviews: Which Fix Comes First")
    st.markdown(
        "**Business question:** a marketplace gets thousands of bad reviews. "
        "*Which problem should it fix first, and how much is it worth?*"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Orders analysed", f"{A['n']:,}")
    c2.metric("Delivered late", pct(A["late_rate"]))
    c3.metric("Low reviews (1-2★): late vs on-time",
              f"{pct(A['low_rate_late'], 0)} vs {pct(A['low_rate_ontime'], 0)}")
    c4.metric("GMV on late orders", f"R$ {C['gmv_of_late_orders_brl'] / 1e6:.2f}M",
              f"{C['gmv_of_late_orders_brl'] / C['gmv_total_brl'] * 100:.1f}% of GMV", delta_color="off")

    st.subheader("What the data says")
    st.markdown(
        f"""
1. **Late delivery is the dominant driver of bad reviews.** After controlling for product
   category, price, freight, seller and customer state, payment type and time period, a late
   order has **{A['odds_ratio']:.1f}× the odds** of a 1-2★ review
   (95% CI {A['or_ci'][0]:.1f}-{A['or_ci'][1]:.1f}). Comparing orders *within the same seller*
   gives a similar effect (**+{FE['late_effect_pp']:.0f} percentage points**).
2. **There is a cliff, not a slope.** Orders up to 3 days late are only mildly penalised;
   beyond 3 days the low-review rate jumps to roughly 60-80% (section 2). **{pct(A['share_excess_low_from_over_3d'], 0)}
   of the reviews attributable to lateness come from orders more than 3 days late**, which are
   {pct(A['share_late_orders_over_3d'], 0)} of late orders. Fix the severe tail first.
3. **Review text corroborates the mechanism.** Among late orders with a comment, {pct(NR_LATE, 0)} mention
   non-receipt vs {pct(NR_ONTIME, 0)} for on-time orders (section 2).
4. **A fix list, not a report.** Section 3 ranks categories, states, routes and sellers by the
   number of low reviews that lateness explains.
5. **Honest limit on the money story:** repeat purchasing is tiny here (only {pct(B['repeat_rate_overall'])}
   of customers return within 180 days), and a low review is linked to only a small further drop
   (about {B['repeat_drop_pp']:.1f} pp; borderline significance, p={B['p_value']:.2f} in the 180-day model,
   Cox hazard ratio {COX_LOW_HR:.2f}, p={COX_LOW_P:.2f}). Chained through, that is only about
   R$ {C['revenue_at_risk_brl']:,.0f}, so the headline is **not** built on repeat revenue. The app reports GMV
   *exposed* to lateness instead (section 4 explains why).
"""
    )
    st.info(
        f"**Recommendation:** target the >3-day-late tail first. Roughly {A['excess_low_total_bin_model']:,.0f} "
        f"of the {A['n_late'] * A['low_rate_late']:,.0f} low reviews on late orders "
        "are attributable to lateness itself. Validate with the experiment design in section 5."
    )

# ======================================================================
elif page.startswith("2"):
    st.header("Delay → bad reviews")
    dr = D["dose_response"].copy()
    dr["label"] = dr["delay_bin"].map(BIN_LABELS)

    left, right = st.columns(2)
    with left:
        fig = px.bar(dr, x="label", y="raw_low_score_rate", text=dr["raw_low_score_rate"].map(lambda v: pct(v, 0)),
                     labels={"label": "", "raw_low_score_rate": "Share of 1-2★ reviews"})
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(title="Raw low-review rate by lateness", height=380)
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dr["label"], y=dr["adj_odds_ratio"], mode="markers+lines",
            error_y=dict(type="data", symmetric=False,
                         array=dr["ci_high"] - dr["adj_odds_ratio"],
                         arrayminus=dr["adj_odds_ratio"] - dr["ci_low"]),
            name="Adjusted odds ratio"))
        fig.add_hline(y=1, line_dash="dot")
        fig.update_layout(title="Adjusted odds ratio vs on-time (95% CI)", height=380,
                          yaxis_title="Odds ratio (log scale)", yaxis_type="log")
        st.plotly_chart(fig, width="stretch")
    st.caption("Adjusted for price, freight ratio, weight, items, promised days, same-state shipping, "
               "installments, product category, seller state, customer state, payment type, quarter. "
               "Standard errors clustered by seller.")

    st.subheader("Robustness: would a confounder explain this away?")
    r1, r2, r3 = st.columns(3)
    r1.metric("Logit odds ratio", f"{A['odds_ratio']:.1f}", f"CI {A['or_ci'][0]:.1f}-{A['or_ci'][1]:.1f}", delta_color="off")
    r2.metric("Seller fixed effects (LPM)", f"+{FE['late_effect_pp']:.1f} pp",
              f"CI {FE['ci_pp'][0]:.1f}-{FE['ci_pp'][1]:.1f}", delta_color="off")
    r3.metric("E-value (risk-ratio scale)", f"{A['evalue_point']:.1f}",
              f"CI bound {A['evalue_ci']:.1f}", delta_color="off")
    st.markdown(
        f"An unmeasured confounder would need to be associated with **both** lateness and bad reviews by a "
        f"risk ratio of about **{A['evalue_ci']:.0f}** (beyond all included controls) to fully explain the "
        "effect away. That is very strong; but it does not prove causation."
    )

    st.subheader("Does review text agree? (Portuguese keyword tagging)")
    ta = D["text_aspects"].copy()
    ta = ta[ta["group"].isin(["late delivery", "on-time delivery", "late AND low score", "on-time AND low score"])]
    long = ta.melt(id_vars=["group", "reviews_with_text"], var_name="aspect", value_name="share")
    long["aspect"] = long["aspect"].map(ASPECT_LABELS)
    fig = px.bar(long, x="aspect", y="share", color="group", barmode="group",
                 labels={"share": "Share of reviews with text", "aspect": ""})
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(height=400)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Only about {TEXT_SHARE:.0%} of reviews contain text. Keyword rules are transparent but miss sarcasm "
               "and misspellings (see section 7).")

# ======================================================================
elif page.startswith("3"):
    st.header("Fix list: what to fix first")
    st.markdown(
        "Ranked by **excess low reviews**: the number of 1-2★ reviews that lateness is estimated to add, "
        "using the lateness-bin effects from section 2 and each order's own risk profile. "
        "*GMV on late orders* is exposure, not lost revenue."
    )
    tab = st.tabs(["Product category", "Customer state", "Seller → customer route", "Individual sellers"])
    specs = [
        ("rank_category", "category_grp", "Category"),
        ("rank_state", "customer_state", "Customer state"),
        ("rank_route", "route", "Route"),
        ("rank_seller", "seller_id", "Seller (id prefix)"),
    ]
    for t, (file, key, label) in zip(tab, specs):
        with t:
            df = D[file].copy()
            top = df.head(12)
            fig = px.bar(top.iloc[::-1], x="excess_low_reviews", y=key, orientation="h",
                         labels={"excess_low_reviews": "Excess low reviews due to lateness", key: label},
                         hover_data=["orders", "late_rate", "avg_days_late"])
            fig.update_layout(height=420)
            st.plotly_chart(fig, width="stretch")
            show = df.rename(columns={key: label}).assign(
                late_rate=lambda x: x["late_rate"].map(lambda v: pct(v)),
                low_rate=lambda x: x["low_rate"].map(lambda v: pct(v)),
            )[[ "rank", label, "orders", "late_rate", "low_rate", "late_orders", "avg_days_late",
                "excess_low_reviews", "late_gmv_brl"]].round(1)
            st.dataframe(show, width="stretch", hide_index=True)
    st.caption("A segment ranks high either because it has many late orders (volume) or a high late rate. "
               "Check both columns before choosing the fix: volume problems and rate problems need different actions.")

# ======================================================================
elif page.startswith("4"):
    st.header("Repeat purchase (survival layer)")
    km = D["km_curves"]
    fig = go.Figure()
    for g, name in [("on_time", "First order on time"), ("late", "First order late")]:
        s = km[km["group"] == g]
        fig.add_trace(go.Scatter(x=s["day"], y=s["no_repeat_prob"], mode="lines", name=name))
        fig.add_trace(go.Scatter(x=list(s["day"]) + list(s["day"])[::-1],
                                 y=list(s["ci_high"]) + list(s["ci_low"])[::-1],
                                 fill="toself", opacity=0.15, line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"))
    fig.update_layout(height=420, xaxis_title="Days since first order",
                      yaxis_title="Share of customers not yet returned", yaxis_range=[0.95, 1.0])
    st.plotly_chart(fig, width="stretch")
    st.caption("Kaplan-Meier curves, 95% CI. The y-axis is zoomed: almost nobody returns.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Customers returning within 180 days", pct(B["repeat_rate_overall"], 2))
    c2.metric("Return rate after low review vs other", f"{pct(B['repeat_rate_low'], 2)} vs {pct(B['repeat_rate_other'], 2)}")
    c3.metric("Low review → repeat (odds ratio)", f"{B['odds_ratio']:.2f}",
              f"CI {B['or_ci'][0]:.2f}-{B['or_ci'][1]:.2f}, p={B['p_value']:.2f}", delta_color="off")
    st.markdown(
        f"""
**Reading this honestly.** The point estimate says a low review lowers the 180-day return chance by about
{B['repeat_drop_pp']:.2f} percentage points, but the confidence interval
({B['repeat_drop_ci_pp'][0]:.2f} to {B['repeat_drop_ci_pp'][1]:.2f} pp) includes zero. Chaining this with the
number of excess low reviews and the average next-order value (R$ {B['avg_next_order_value']:.0f}) gives an
estimated **R$ {C['revenue_at_risk_brl']:,.0f}** of repeat revenue (CI
R$ {C['revenue_at_risk_ci_brl'][0]:,.0f} to R$ {C['revenue_at_risk_ci_brl'][1]:,.0f}), which is negligible
against R$ {C['gmv_total_brl'] / 1e6:.1f}M GMV. Evidence is mixed: the 180-day model is borderline (p={B['p_value']:.2f}),
while the Cox model over the full follow-up finds a hazard ratio of {COX_LOW_HR:.2f} (p={COX_LOW_P:.2f}),
i.e. a modest slowdown in returning. Either way the money involved is small.

**Why report it anyway?** Because a small repeat-purchase effect is itself a finding: on this marketplace
the cost of a bad review is barely visible in repeat purchasing. The real cost probably sits elsewhere
(conversion, seller ranking, refunds, support contacts), and none of those are in this dataset.
"""
    )
    st.subheader("Cox proportional hazards (time to next order)")
    cox = D["cox_hazard_ratios"].round(3)
    st.dataframe(cox, width="stretch", hide_index=True)
    st.caption("Hazard ratio below 1 means slower return. With about 2,000 returning customers, estimates are imprecise. "
               "Proportional-hazards assumption not formally tested here.")

# ======================================================================
elif page.startswith("5"):
    st.header("Experiment planner")
    st.markdown(
        "How would we *prove* a fix works? Randomise at-risk orders to an intervention (for example, "
        "expedited shipping or proactive delay messaging) and compare low-review rates."
    )
    orders = load_orders()
    base_low = float(orders["low_score"].mean())
    attributable = A["excess_low_total_bin_model"] / A["n"]

    c1, c2 = st.columns(2)
    with c1:
        p0 = st.number_input("Baseline low-review rate in the test population", 0.01, 0.9,
                             round(base_low, 3), 0.005, format="%.3f")
        cut = st.slider("Expected relative reduction in late deliveries from the fix", 5, 80, 30, 5,
                        format="%d%%") / 100
        share = st.slider("Share of this population's low reviews that lateness explains", 1, 60,
                          int(round(attributable / base_low * 100)), 1, format="%d%%") / 100
    with c2:
        alpha = st.select_slider("Significance level (α)", [0.01, 0.05, 0.10], value=0.05)
        power = st.select_slider("Power", [0.7, 0.8, 0.9], value=0.8)
        rho = st.slider("CUPED correlation with a pre-period covariate (ρ)", 0.0, 0.7, 0.0, 0.05)

    rel_lift = share * cut
    n = sample_size_two_prop(p0, rel_lift, alpha=alpha, power=power, cuped_rho=rho)
    st.metric("Orders needed per arm", f"{n:,.0f}")
    st.markdown(
        f"- Fix removes **{cut:.0%}** of late deliveries, so the low-review rate falls by about "
        f"**{rel_lift:.1%} relative** ({p0:.1%} → {p0 * (1 - rel_lift):.1%}).\n"
        f"- Detecting that at α={alpha}, power={power} needs about **{n:,.0f} orders per arm**"
        f"{' (after CUPED variance reduction)' if rho > 0 else ''}.\n"
        "- **Decision rule to write down before the test:** ship if the low-review rate drops and the lower "
        "bound of the 95% CI is below zero; stop and rethink if the point estimate is zero or worse. "
        "Also track cost of the intervention, since a fix that costs more than it saves is not a win."
    )
    st.caption("Sample size uses a two-proportion z-test. CUPED reduces required n by (1 − ρ²). "
               "Run the experiment on at-risk orders only to concentrate the effect and cut sample size.")

# ======================================================================
elif page.startswith("6"):
    st.header("Explore the data")
    o = load_orders()
    f1, f2, f3 = st.columns(3)
    cats = f1.multiselect("Category", sorted(o["category_grp"].unique()))
    states = f2.multiselect("Customer state", sorted(o["customer_state"].unique()))
    months = f3.select_slider("Order month range", options=sorted(o["order_month"].unique()),
                              value=("2017-01", "2018-08"))
    m = o[(o["order_month"] >= months[0]) & (o["order_month"] <= months[1])]
    if cats:
        m = m[m["category_grp"].isin(cats)]
    if states:
        m = m[m["customer_state"].isin(states)]
    if len(m) < 100:
        st.warning("Too few orders for this filter.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Orders", f"{len(m):,}")
        k2.metric("Late rate", pct(m["late"].mean()))
        k3.metric("Low-review rate", pct(m["low_score"].mean()))
        k4.metric("Avg rating", f"{m['review_score'].mean():.2f}")
        t = m.groupby("order_month").agg(late_rate=("late", "mean"), low_rate=("low_score", "mean"),
                                         orders=("late", "size")).reset_index()
        fig = px.line(t, x="order_month", y=["late_rate", "low_rate"],
                      labels={"value": "Rate", "order_month": "Month", "variable": ""})
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")

# ======================================================================
else:
    st.header("Method & limits")
    st.markdown(
        f"""
### Pipeline
1. **Data build** (pandas; SQL version in `sql/`): join orders, items, products, sellers, payments and the
   latest review per order. Keep delivered orders with a delivery date and a review
   ({A['n']:,} orders).
2. **Delay** = actual delivery date minus the date promised at purchase. **Low review** = 1-2 stars.
3. **Stage A:** logistic regression of low review on lateness (binary and by bin) with controls; SE clustered by
   seller. Robustness: seller fixed-effects linear probability model; E-value for unmeasured confounding.
4. **Stage B:** among first orders with ≥180 days of follow-up, logistic regression of repeat purchase on low review,
   controlling for lateness and order characteristics.
5. **Ranking:** per-order effect (bin-specific) summed by segment.
6. **Survival:** Kaplan-Meier and Cox PH for time to next order.
7. **Text:** rule-based Portuguese keyword tagging used only to *corroborate* the mechanism.

### Assumptions the causal reading depends on
- No important unmeasured factor drives both lateness and bad reviews after controls (E-value quantifies this).
- Lateness is measured against the promise shown at purchase; customers with unrealistic expectations are not modelled.
- The effect of a low review on returning is similar for customers whose low review was caused by lateness.

### Known limitations
- **Observational data**, not randomised: estimates are not proof of causation.
- **Repeat purchases are rare** (about 3% of customers ever reorder in this dataset), so Stage B has little power.
- **Only about {TEXT_SHARE:.0%} of reviews have text**; keyword rules miss sarcasm, typos and mixed sentiment.
- **Sellers with few orders** are noisy; the seller table uses a 50-order minimum.
- **2016-2018 Brazilian marketplace**: do not assume the cliff location (about 3 days) transfers elsewhere.
- Rankings use each order's modelled effect, so they depend on the model specification.

### Reproduce
`python run_pipeline.py` rebuilds all files in `data/processed/` from the raw CSVs in `data/raw/`.
"""
    )
