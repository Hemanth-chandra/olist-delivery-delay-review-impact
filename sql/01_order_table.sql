-- Order-level analysis table (DuckDB SQL), equivalent to src/data_prep.py.
-- One row = one delivered order that has a delivery date and a review.
-- Run:  duckdb < sql/01_order_table.sql     (from the project root)
-- or:   python sql/run_sql_check.py          (builds it and compares with the pandas table)

CREATE OR REPLACE TABLE orders AS
SELECT * FROM read_csv_auto('data/raw/olist_orders_dataset.csv');

CREATE OR REPLACE TABLE customers AS
SELECT * FROM read_csv_auto('data/raw/olist_customers_dataset.csv');

CREATE OR REPLACE TABLE items AS
SELECT * FROM read_csv_auto('data/raw/olist_order_items_dataset.csv');

CREATE OR REPLACE TABLE products AS
SELECT * FROM read_csv_auto('data/raw/olist_products_dataset.csv');

CREATE OR REPLACE TABLE sellers AS
SELECT * FROM read_csv_auto('data/raw/olist_sellers_dataset.csv');

CREATE OR REPLACE TABLE payments AS
SELECT * FROM read_csv_auto('data/raw/olist_order_payments_dataset.csv');

CREATE OR REPLACE TABLE reviews AS
SELECT * FROM read_csv_auto('data/raw/olist_order_reviews_dataset.csv');

-- main (priciest) item per order + order totals
CREATE OR REPLACE TABLE order_items AS
WITH ranked AS (
    SELECT i.order_id, i.seller_id,
           ROW_NUMBER() OVER (PARTITION BY i.order_id ORDER BY i.price DESC) AS rn
    FROM items i
),
totals AS (
    SELECT order_id,
           COUNT(*)           AS n_items,
           SUM(price)         AS price_sum,
           SUM(freight_value) AS freight_sum
    FROM items
    GROUP BY order_id
)
SELECT r.order_id, r.seller_id, s.seller_state, t.n_items, t.price_sum, t.freight_sum
FROM ranked r
JOIN totals t USING (order_id)
LEFT JOIN sellers s USING (seller_id)
WHERE r.rn = 1;

-- latest review per order
CREATE OR REPLACE TABLE latest_review AS
SELECT order_id, review_score
FROM (
    SELECT order_id, review_score,
           ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY review_answer_timestamp DESC) AS rn
    FROM reviews
)
WHERE rn = 1;

CREATE OR REPLACE TABLE analysis_orders AS
SELECT
    o.order_id,
    c.customer_unique_id,
    c.customer_state,
    oi.seller_id,
    oi.seller_state,
    oi.n_items,
    oi.price_sum,
    oi.freight_sum,
    lr.review_score,
    CASE WHEN lr.review_score <= 2 THEN 1 ELSE 0 END                          AS low_score,
    DATE_DIFF('second', o.order_estimated_delivery_date,
                        o.order_delivered_customer_date) / 86400.0           AS delay_days,
    CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date
         THEN 1 ELSE 0 END                                                    AS late,
    DATE_DIFF('second', o.order_purchase_timestamp,
                        o.order_estimated_delivery_date) / 86400.0           AS promised_days
FROM orders o
JOIN customers   c  USING (customer_id)
JOIN order_items oi USING (order_id)
JOIN latest_review lr USING (order_id)
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL;

-- headline check: late vs on-time low-review rate
SELECT late,
       COUNT(*)                        AS orders,
       ROUND(AVG(low_score), 4)        AS low_review_rate
FROM analysis_orders
GROUP BY late
ORDER BY late;
