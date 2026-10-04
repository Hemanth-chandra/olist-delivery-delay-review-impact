# Delivery Delay and Customer Reviews: Which Fix Comes First

I built this project to answer one business question: a marketplace gets a lot of bad reviews, so what should it fix first?

I used the Olist Brazilian e-commerce dataset (real orders from 2016 to 2018). After joining the tables and keeping delivered orders with a review, I analysed 95,824 orders.

Live app: <add your Streamlit link here>

## What I found

- 8.0% of orders were delivered late. Late orders got a 1 or 2 star review 54% of the time, compared to 9% for on-time orders.
- After controlling for product category, price, freight, seller, customer state, payment type and time period, a late order has about 12 times the odds of a bad review (95% CI 11.2 to 13.1).
- I checked this two more ways: comparing orders within the same seller (about 43 percentage points higher), and an E-value of about 10, which means an unmeasured factor would have to be very strong to explain the result away.
- The effect is a cliff, not a slope. Orders up to 3 days late are only mildly affected. After 3 days, bad reviews jump to 60 to 80%. About 92% of the bad reviews linked to lateness come from orders more than 3 days late.
- Review text agrees with this. Among late orders with a comment, 37% mention not receiving the product, against 3% for on-time orders.

One thing I want to be clear about: I tried to connect bad reviews to lost repeat purchases, but only 1.9% of customers order again within 180 days, and the effect was small and borderline. So I did not claim a revenue loss. I report the value of late orders (R$ 1.31M, 8.6% of total sales) only as exposure.

## What the app has

1. Summary of the answer
2. Delay vs bad reviews, with the cost of each delay level
3. Ranked fix list by category, state, route and seller
4. Repeat purchase analysis (Kaplan-Meier and Cox model)
5. Experiment planner (sample size for an A/B test, with CUPED)
6. Data explorer
7. Method and limits

## How I did it

- Data build in pandas, plus the same table in SQL (DuckDB). A script checks that both give the same result.
- Logistic regression with controls, standard errors clustered by seller.
- Seller fixed-effects check and E-value for robustness.
- Per-order effects added up by segment to rank what to fix first.
- Simple keyword tagging on Portuguese review text, only to check the mechanism.

## Folder structure

```
app.py                  Streamlit app
run_pipeline.py         rebuilds all processed files from raw data
src/                    data prep, analysis, text tagging
sql/                    SQL version of the data build
tests/                  basic checks on the outputs
data/processed/         output files the app reads
data/raw/               put the Olist CSV files here (not uploaded)
docs/                   notes for interview preparation
```

## Run it on your computer

```bash
pip install -r requirements.txt
streamlit run app.py
```

To rebuild everything from the raw data, download the 9 Olist CSV files from Kaggle into `data/raw/`, then:

```bash
pip install -r requirements-pipeline.txt
python run_pipeline.py
```

## Limitations

- This is observational data, so the results are estimates and not proof that lateness causes bad reviews.
- Repeat purchases are very rare in this data, so I could not size the revenue effect.
- Only about 42% of reviews have text, and keyword rules miss sarcasm and spelling mistakes.
- The data is from a Brazilian marketplace in 2016 to 2018, so the 3-day cliff may not hold elsewhere.
- Small sellers give noisy results, so the seller table only includes sellers with at least 50 orders.

## Data

Olist Brazilian E-Commerce Public Dataset on Kaggle, licensed CC BY-NC-SA 4.0 (non-commercial use with credit). This repo only has derived files, not the raw data.

## Next steps

- Replace the keyword tagging with a multilingual language model and compare.
- Try other definitions of a bad review (1 star only, or 3 stars and below).
- Add data on conversion and refunds to measure the real cost of bad reviews.
