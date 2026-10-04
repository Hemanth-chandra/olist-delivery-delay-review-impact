# Interview prep: Delivery Delay and Customer Reviews

> You must be able to explain every number below from your own run. Re-run `python run_pipeline.py`,
> open `data/processed/results.json`, and confirm the figures before quoting them.

## 30-second pitch
"A marketplace gets thousands of bad reviews and doesn't know what to fix first. I used 96k real Olist orders to estimate how much late delivery drives 1-2★ reviews, controlling for category, price, freight, seller and region. Late orders have about 12× the odds of a bad review, and the effect is a cliff: orders more than 3 days late account for 92% of the damage. I turned that into a ranked fix list and designed the A/B test to validate it. I also found that repeat purchasing is too rare here to justify a revenue-at-risk claim, so I reported delivery exposure instead of inventing a number."

## Resume bullets (edit to your voice)
- Estimated the effect of late delivery on review ratings across 95.8k Olist orders (adjusted OR 12.1, 95% CI 11.2-13.1; seller fixed-effects check +43 pp; E-value ≈ 10), finding a >3-day "cliff" that accounts for 92% of lateness-driven low reviews.
- Built a segment-level fix list (category, state, route, seller) from per-order effect estimates, plus an experiment planner with power analysis and CUPED.
- Added survival analysis (Kaplan-Meier, Cox) and Portuguese review-text tagging to test the mechanism, and reported a null-ish repeat-purchase effect rather than overclaiming revenue impact.
- Deployed an interactive Streamlit app; SQL (DuckDB) and pandas pipelines verified equivalent.

## Questions to expect (and answers to prepare)
1. **Why logistic regression, not XGBoost?** The goal is an interpretable effect estimate with confidence intervals, not prediction accuracy.
2. **How do you know lateness *causes* bad reviews?** I don't have randomisation. I controlled for observed confounders, checked within-seller comparisons (seller fixed effects), computed an E-value (~10), and checked the text mechanism (non-receipt mentions). Residual risk: unmeasured factors (e.g. carrier quality for specific routes).
3. **Why cluster standard errors by seller?** Orders from the same seller are correlated; ignoring that understates uncertainty.
4. **Why a 1-2★ cut-off?** Scores are skewed (58% are 5★); 1-2★ is the clear dissatisfaction group. Try 1★ only and ≤3★ as sensitivity checks.
5. **What does the "cliff" mean operationally?** Prioritise orders heading for >3 days late (proactive messaging, expedite, reroute) over small slips.
6. **Why did you not claim revenue impact?** Only ~1.9% of customers reorder within 180 days and the review effect on that is small/borderline, so any BRL figure would be noise. The honest finding is that the review cost probably sits in unmeasured channels (conversion, ranking, support).
7. **What would you do with more data?** Conversion and seller-ranking data, refunds/support contacts, carrier IDs, and a randomised pilot.
8. **Failure modes?** Keyword text tagging, sparse repeat purchases, model-dependent rankings, time period (2016-2018), Brazil-specific logistics.
9. **How would you validate in production?** Randomise at-risk orders to an intervention, pre-register the decision rule, track cost per order, use CUPED to cut sample size (Experiment planner page).
10. **Why SQL *and* pandas?** SQL is how analysts build tables in practice; the check script proves the two agree on row counts and key rates.

## Things to try before the interview (makes the project more yours)
- Re-run with low review defined as ≤3★ and as 1★ only; note what changes.
- Add one more control you think matters (e.g. freight value bins) and see if the OR moves.
- Replace the keyword tagger with a multilingual zero-shot model on 1,000 reviews and compare agreement.
- Look at the top 5 sellers in `rank_seller.csv` and write a one-paragraph recommendation.
