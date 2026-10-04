"""Causal-style analysis chain:

  late delivery  ->  low review score (1-2 stars)  ->  lower repeat purchase  ->  revenue at risk

Everything is observational. Causal language in the app is always "estimated effect
under the stated assumptions", never "proof".
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import expit
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

RNG = np.random.default_rng(42)

CONTROLS = (
    "log_price + freight_ratio + log_weight + n_items + promised_days + same_state"
    " + installments + C(category_grp) + C(seller_state) + C(customer_state)"
    " + C(payment_type) + C(purchase_q)"
)


# ---------------------------------------------------------------- stage A
def fit_delay_effect(df: pd.DataFrame):
    """Logit: low_score ~ late + controls, SE clustered by seller."""
    groups = pd.factorize(df["seller_id"])[0]
    model = smf.logit(f"low_score ~ late + {CONTROLS}", data=df).fit(
        disp=0, method="bfgs", maxiter=2000, cov_type="cluster", cov_kwds={"groups": groups}
    )
    beta, se = model.params["late"], model.bse["late"]

    # ATT-style per-order effect on the probability scale for late orders
    eta1 = model.predict(df, which="linear")
    eta0 = eta1 - beta * df["late"].to_numpy()
    p1, p0 = expit(eta0 + beta), expit(eta0)
    effect_i = np.where(df["late"] == 1, p1 - p0, 0.0)

    late = df["late"] == 1
    att = float(effect_i[late].mean())
    rr = float(p1[late].mean() / p0[late].mean())
    draws = RNG.normal(beta, se, 4000)
    base = expit(eta0[late])
    att_draws = np.array([(expit(eta0[late] + b) - base).mean() for b in draws[:1000]])
    rr_draws = np.array([expit(eta0[late] + b).mean() / base.mean() for b in draws[:1000]])
    rr_lo = float(np.percentile(rr_draws, 2.5))
    res = {
        "beta": float(beta),
        "se": float(se),
        "odds_ratio": float(np.exp(beta)),
        "or_ci": [float(np.exp(beta - 1.96 * se)), float(np.exp(beta + 1.96 * se))],
        "att_pp": att * 100,
        "att_ci_pp": [float(np.percentile(att_draws, 2.5) * 100),
                      float(np.percentile(att_draws, 97.5) * 100)],
        "risk_ratio": rr,
        "evalue_point": evalue(rr),
        "risk_ratio_ci_low": rr_lo,
        "evalue_ci": evalue(rr_lo),
        "n": int(len(df)),
        "n_late": int(late.sum()),
        "late_rate": float(late.mean()),
        "low_rate_late": float(df.loc[late, "low_score"].mean()),
        "low_rate_ontime": float(df.loc[~late, "low_score"].mean()),
    }
    return res, effect_i, model


def evalue(rr: float) -> float:
    """VanderWeele-Ding E-value: how strong an unmeasured confounder must be
    (on the risk-ratio scale, with both exposure and outcome) to explain away rr."""
    if rr is None or rr <= 1:
        return 1.0
    return float(rr + np.sqrt(rr * (rr - 1)))


def dose_response(df: pd.DataFrame) -> pd.DataFrame:
    """Odds ratios by lateness bin vs on-time/early, same controls."""
    d = df.copy()
    d["delay_bin"] = pd.Categorical(
        d["delay_bin"],
        ["on_time_or_early", "late_0_3d", "late_3_7d", "late_7_14d", "late_14d_plus"],
    )
    groups = pd.factorize(d["seller_id"])[0]
    m = smf.logit(
        f"low_score ~ C(delay_bin, Treatment('on_time_or_early')) + {CONTROLS}", data=d
    ).fit(disp=0, method="bfgs", maxiter=2000, cov_type="cluster", cov_kwds={"groups": groups})
    rows = []
    raw = d.groupby("delay_bin", observed=True)["low_score"].agg(["mean", "size"])
    for b in ["on_time_or_early", "late_0_3d", "late_3_7d", "late_7_14d", "late_14d_plus"]:
        name = f"C(delay_bin, Treatment('on_time_or_early'))[T.{b}]"
        if b == "on_time_or_early":
            or_, lo, hi = 1.0, 1.0, 1.0
        else:
            b_, s_ = m.params[name], m.bse[name]
            or_, lo, hi = np.exp(b_), np.exp(b_ - 1.96 * s_), np.exp(b_ + 1.96 * s_)
        rows.append({
            "delay_bin": b,
            "orders": int(raw.loc[b, "size"]),
            "raw_low_score_rate": float(raw.loc[b, "mean"]),
            "adj_odds_ratio": float(or_),
            "ci_low": float(lo),
            "ci_high": float(hi),
        })
    # per-order effect on the probability scale, using each order's own lateness bin
    eta = m.predict(d, which="linear")
    eff = np.zeros(len(d))
    for b in ["late_0_3d", "late_3_7d", "late_7_14d", "late_14d_plus"]:
        name = f"C(delay_bin, Treatment('on_time_or_early'))[T.{b}]"
        mask = (d["delay_bin"] == b).to_numpy()
        e1 = eta.to_numpy()[mask]
        eff[mask] = expit(e1) - expit(e1 - m.params[name])
    return pd.DataFrame(rows), eff


def seller_fixed_effects_check(df: pd.DataFrame) -> dict:
    """Robustness: linear probability model with seller fixed effects (within-seller)."""
    d = df.copy()
    num = ["late", "log_price", "freight_ratio", "log_weight", "n_items",
           "promised_days", "same_state", "installments"]
    dummies = pd.get_dummies(d[["customer_state", "purchase_q"]], drop_first=True, dtype=float)
    X = pd.concat([d[num].astype(float), dummies], axis=1)
    y = d["low_score"].astype(float)
    g = d["seller_id"]
    Xw = X - X.groupby(g).transform("mean")
    yw = y - y.groupby(g).transform("mean")
    m = sm.OLS(yw, Xw).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g)[0]})
    b, s = m.params["late"], m.bse["late"]
    return {
        "late_effect_pp": float(b * 100),
        "ci_pp": [float((b - 1.96 * s) * 100), float((b + 1.96 * s) * 100)],
        "n_sellers": int(g.nunique()),
    }


# ---------------------------------------------------------------- stage B
def fit_repeat_effect(df: pd.DataFrame):
    """Among first orders with >=180d follow-up: repeat_180d ~ low_score + controls.

    We control for `late` so the coefficient isolates the review pathway
    (conservative: any direct late->repeat channel is not credited to reviews).
    """
    d = df[(df["is_first_order"] == 1) & (df["eligible_180d"] == 1)].copy()
    f = (
        "repeat_180d ~ low_score + late + log_price + freight_ratio + promised_days"
        " + C(category_grp) + C(customer_state) + C(payment_type) + C(purchase_q)"
    )
    mod = smf.logit(f, data=d)
    m0 = mod.fit(disp=0, method="bfgs", maxiter=2000)
    # one Newton step from the BFGS solution gives a finite, Hessian-based covariance
    m = mod.fit(start_params=m0.params, method="newton", maxiter=1, disp=0)
    g, se = m.params["low_score"], m.bse["low_score"]
    low = d["low_score"] == 1
    eta1 = m.predict(d, which="linear")
    eta0 = eta1 - g * d["low_score"].to_numpy()
    drop = float((expit(eta0[low]) - expit(eta1[low])).mean())  # repeat prob lost
    draws = RNG.normal(g, se, 1000)
    drop_draws = np.array([(expit(eta1[low] - g) - expit(eta1[low] - g + b)).mean() for b in draws])
    nxt = df.loc[df["next_order_value"].notna(), "next_order_value"]
    res = {
        "gamma": float(g),
        "se": float(se),
        "odds_ratio": float(np.exp(g)),
        "or_ci": [float(np.exp(g - 1.96 * se)), float(np.exp(g + 1.96 * se))],
        "p_value": float(m.pvalues["low_score"]),
        "repeat_rate_overall": float(d["repeat_180d"].mean()),
        "repeat_rate_low": float(d.loc[low, "repeat_180d"].mean()),
        "repeat_rate_other": float(d.loc[~low, "repeat_180d"].mean()),
        "repeat_drop_pp": drop * 100,
        "repeat_drop_ci_pp": [float(np.percentile(drop_draws, 2.5) * 100),
                              float(np.percentile(drop_draws, 97.5) * 100)],
        "avg_next_order_value": float(nxt.mean()),
        "n": int(len(d)),
        "n_repeaters": int(d["repeat_180d"].sum()),
    }
    return res, drop_draws


def revenue_chain(df, effect_i, repeat_res, drop_draws, att_res) -> dict:
    """Combine stages: excess low-score orders -> lost repeat prob -> BRL at risk."""
    late = df["late"] == 1
    excess_low = float(effect_i[late].sum())
    per_low = repeat_res["avg_next_order_value"]
    point = excess_low * (repeat_res["repeat_drop_pp"] / 100) * per_low
    lo = excess_low * (np.percentile(drop_draws, 2.5)) * per_low
    hi = excess_low * (np.percentile(drop_draws, 97.5)) * per_low
    return {
        "excess_low_score_orders": excess_low,
        "revenue_at_risk_brl": float(point),
        "revenue_at_risk_ci_brl": [float(lo), float(hi)],
        "gmv_of_late_orders_brl": float(df.loc[late, "payment_value"].sum()),
        "gmv_total_brl": float(df["payment_value"].sum()),
        "per_excess_low_score_brl": float(per_low * repeat_res["repeat_drop_pp"] / 100),
    }


# ---------------------------------------------------------------- ranking
def rank_segments(df, effect_i, per_low_brl, by, min_orders=100):
    d = df.assign(effect=effect_i)
    late = d[d["late"] == 1]
    g = d.groupby(by).agg(orders=("order_id", "size"), late_rate=("late", "mean"),
                          low_rate=("low_score", "mean"))
    gl = late.groupby(by).agg(
        late_orders=("order_id", "size"),
        avg_days_late=("delay_days", "mean"),
        excess_low_reviews=("effect", "sum"),
        late_gmv_brl=("payment_value", "sum"),
    )
    out = g.join(gl, how="left").fillna(0)
    out = out[out["orders"] >= min_orders].copy()
    out = out.sort_values("excess_low_reviews", ascending=False).reset_index()
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def seller_ranking(df, effect_i, per_low_brl, min_orders=50):
    r = rank_segments(df, effect_i, per_low_brl, "seller_id", min_orders=min_orders)
    st = df.groupby("seller_id")["seller_state"].first()
    r["seller_state"] = r["seller_id"].map(st)
    r["seller_id"] = r["seller_id"].str[:8]  # shortened id for display
    return r.head(200)


# ---------------------------------------------------------------- survival
def survival_tables(cust: pd.DataFrame):
    """Kaplan-Meier (late vs on-time first orders) + Cox PH hazard ratios."""
    from lifelines import CoxPHFitter, KaplanMeierFitter

    c = cust.copy()
    grid = np.arange(0, 541, 15)
    rows = []
    for label, sub in [("on_time", c[c["late"] == 0]), ("late", c[c["late"] == 1])]:
        km = KaplanMeierFitter().fit(sub["duration_days"], sub["event"], label=label)
        s = km.survival_function_at_times(grid).values
        ci = km.confidence_interval_
        lo = np.interp(grid, ci.index.values, ci.iloc[:, 0].values)
        hi = np.interp(grid, ci.index.values, ci.iloc[:, 1].values)
        for t, sv, l, h in zip(grid, s, lo, hi):
            rows.append({"group": label, "day": int(t), "no_repeat_prob": float(sv),
                         "ci_low": float(l), "ci_high": float(h), "n": int(len(sub))})
    km_df = pd.DataFrame(rows)

    cox_df = c[["duration_days", "event", "late", "low_score", "log_price",
                "freight_ratio", "promised_days"]].dropna()
    cox_df = cox_df[cox_df["duration_days"] > 0]
    cph = CoxPHFitter(penalizer=0.01).fit(cox_df, "duration_days", "event")
    s = cph.summary
    cox = pd.DataFrame({
        "covariate": s.index,
        "hazard_ratio": s["exp(coef)"].values,
        "ci_low": s["exp(coef) lower 95%"].values,
        "ci_high": s["exp(coef) upper 95%"].values,
        "p_value": s["p"].values,
    })
    return km_df, cox, {"n_customers": int(len(cox_df)), "n_repeaters": int(cox_df["event"].sum())}


# ---------------------------------------------------------------- experiment
def sample_size_two_prop(p_control, relative_lift, alpha=0.05, power=0.8, cuped_rho=0.0):
    """Orders needed per arm to detect a relative reduction in a binary outcome.

    cuped_rho: correlation between the outcome and a pre-experiment covariate.
    CUPED shrinks variance by (1 - rho^2), so n shrinks by the same factor.
    """
    p_treat = p_control * (1 - relative_lift)
    es = abs(proportion_effectsize(p_control, p_treat))
    n = NormalIndPower().solve_power(effect_size=es, alpha=alpha, power=power, ratio=1.0)
    return float(np.ceil(n * (1 - cuped_rho ** 2)))
