"""Build the SQL table with DuckDB and confirm it matches the pandas pipeline."""
import duckdb
import pandas as pd

con = duckdb.connect()
con.execute(open("sql/01_order_table.sql").read())
sql = con.execute("SELECT * FROM analysis_orders").df()
pq = pd.read_parquet("data/processed/orders.parquet")

print(f"SQL rows:    {len(sql):,}")
print(f"pandas rows: {len(pq):,}")
print(f"late rate    SQL {sql['late'].mean():.4f} | pandas {pq['late'].mean():.4f}")
print(f"low rate     SQL {sql['low_score'].mean():.4f} | pandas {pq['low_score'].mean():.4f}")
assert len(sql) == len(pq), "row counts differ"
assert abs(sql["late"].mean() - pq["late"].mean()) < 1e-9
assert abs(sql["low_score"].mean() - pq["low_score"].mean()) < 1e-9
print("OK: SQL and pandas pipelines agree.")
