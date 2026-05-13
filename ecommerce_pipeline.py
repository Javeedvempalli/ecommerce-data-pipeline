"""
Ecommerce Order Dataset - Full Pipeline
========================================
Steps:
  1. Data Cleaning   - remove duplicates, standardize text, handle nulls
  2. Schema Validation - validate IDs, prices, dates, duplicate orders
  3. KPI Generator   - revenue, AOV, top categories, retention, monthly actives

Dataset tables (train/):
  df_Customers.csv  | df_Orders.csv | df_OrderItems.csv
  df_Payments.csv   | df_Products.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  –  update DATA_DIR to point at your local train/ folder
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = "train/"   # <-- change if needed
TODAY     = datetime.today()


# =============================================================================
# STEP 1 – LOAD RAW DATA
# =============================================================================
def load_data(data_dir: str) -> dict[str, pd.DataFrame]:
    """Load all five CSV files and return a dict of DataFrames."""
    files = {
        "customers":   "df_Customers.csv",
        "orders":      "df_Orders.csv",
        "order_items": "df_OrderItems.csv",
        "payments":    "df_Payments.csv",
        "products":    "df_Products.csv",
    }
    raw: dict[str, pd.DataFrame] = {}
    for key, filename in files.items():
        path = data_dir + filename
        raw[key] = pd.read_csv(path)
        print(f"  Loaded {filename:30s}  shape={raw[key].shape}")
    return raw


# =============================================================================
# STEP 2 – DATA CLEANING
# =============================================================================

# ---------- 2a. Remove Duplicates --------------------------------------------
def remove_duplicates(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Drop exact-row duplicates across all tables.
    Products has many duplicate rows (same product_id repeated for each order
    it appeared in); we keep only the first occurrence per product_id.
    """
    cleaned: dict[str, pd.DataFrame] = {}
    for name, df in dfs.items():
        before = len(df)
        if name == "products":
            df = df.drop_duplicates(subset=["product_id"], keep="first")
        else:
            df = df.drop_duplicates()
        after = len(df)
        if before != after:
            print(f"  [{name}] removed {before - after:,} duplicate rows")
        cleaned[name] = df.reset_index(drop=True)
    return cleaned


# ---------- 2b. Standardise Text ---------------------------------------------
def standardize_text(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    For every string column: strip leading/trailing whitespace and convert
    to lowercase.  City names and category names are the main targets.
    """
    for name, df in dfs.items():
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            df[col] = df[col].str.strip().str.lower()
        dfs[name] = df
    print("  Text standardization complete (strip + lowercase)")
    return dfs


# ---------- 2c. Handle Nulls -------------------------------------------------
def handle_nulls(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Null-handling strategy per table / column:

    orders:
      order_approved_at          → fill with order_purchase_timestamp (assume
                                   same-day approval for the 9 nulls)
      order_delivered_timestamp  → leave as NaT (not yet delivered); a new
                                   boolean flag 'is_delivered' captures this.

    products:
      product_category_name      → fill with 'unknown_category'
      product_weight_g / dims    → fill with column median (15 rows)
    """
    # ── orders ──
    o = dfs["orders"].copy()
    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_timestamp",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        o[col] = pd.to_datetime(o[col], errors="coerce")

    null_approved = o["order_approved_at"].isna().sum()
    o["order_approved_at"] = o["order_approved_at"].fillna(
        o["order_purchase_timestamp"]
    )
    o["is_delivered"] = o["order_delivered_timestamp"].notna()
    print(f"  [orders] filled {null_approved} null order_approved_at values")
    print(f"  [orders] added 'is_delivered' flag "
          f"({o['is_delivered'].sum():,} delivered)")
    dfs["orders"] = o

    # ── products ──
    p = dfs["products"].copy()
    null_cat = p["product_category_name"].isna().sum()
    p["product_category_name"] = p["product_category_name"].fillna(
        "unknown_category"
    )
    numeric_cols = [
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    for col in numeric_cols:
        median_val = p[col].median()
        null_count = p[col].isna().sum()
        p[col] = p[col].fillna(median_val)
        if null_count:
            print(f"  [products] filled {null_count} nulls in {col} "
                  f"with median={median_val:.1f}")
    print(f"  [products] filled {null_cat} null category names → "
          f"'unknown_category'")
    dfs["products"] = p

    return dfs


# =============================================================================
# STEP 3 – SCHEMA VALIDATION FRAMEWORK
# =============================================================================

class ValidationReport:
    """Accumulates validation findings and prints a summary."""

    def __init__(self):
        self.issues: list[dict] = []

    def log(self, rule: str, count: int, severity: str = "WARNING",
            sample=None):
        self.issues.append(
            {"rule": rule, "count": count, "severity": severity,
             "sample": sample}
        )
        flag = "🔴" if severity == "ERROR" else "🟡"
        print(f"  {flag} [{severity}] {rule}: {count:,} issue(s)")

    def summary(self):
        errors   = sum(1 for i in self.issues if i["severity"] == "ERROR")
        warnings = sum(1 for i in self.issues if i["severity"] == "WARNING")
        print(f"\n  Validation complete – {errors} error(s), {warnings} warning(s)")
        return self.issues


def validate_schema(dfs: dict[str, pd.DataFrame]) -> list[dict]:
    """Run all validation rules and return a list of issue dicts."""
    report = ValidationReport()
    customers   = dfs["customers"]
    orders      = dfs["orders"]
    order_items = dfs["order_items"]
    payments    = dfs["payments"]
    products    = dfs["products"]

    # ── Rule 1: Invalid customer IDs ─────────────────────────────────────────
    # Every customer_id in orders must exist in the customers table.
    valid_cust_ids = set(customers["customer_id"])
    bad_cust = orders[~orders["customer_id"].isin(valid_cust_ids)]
    report.log(
        "Invalid customer_id in orders",
        len(bad_cust),
        severity="ERROR" if len(bad_cust) else "WARNING",
        sample=bad_cust["customer_id"].head(3).tolist() if len(bad_cust) else None,
    )

    # ── Rule 2: Invalid product IDs ──────────────────────────────────────────
    valid_prod_ids = set(products["product_id"])
    bad_prod = order_items[~order_items["product_id"].isin(valid_prod_ids)]
    report.log(
        "Invalid product_id in order_items",
        len(bad_prod),
        severity="ERROR" if len(bad_prod) else "WARNING",
    )

    # ── Rule 3: Negative prices ───────────────────────────────────────────────
    neg_price    = (order_items["price"] < 0).sum()
    neg_shipping = (order_items["shipping_charges"] < 0).sum()
    neg_payment  = (payments["payment_value"] < 0).sum()
    report.log("Negative price in order_items",      neg_price,    "ERROR")
    report.log("Negative shipping_charges",          neg_shipping, "ERROR")
    report.log("Negative payment_value in payments", neg_payment,  "ERROR")

    # ── Rule 4: Future purchase dates ────────────────────────────────────────
    future_orders = orders[orders["order_purchase_timestamp"] > TODAY]
    report.log(
        "Future order_purchase_timestamp",
        len(future_orders),
        severity="ERROR" if len(future_orders) else "WARNING",
    )

    # Estimated delivery before purchase (logical impossibility)
    est_col = pd.to_datetime(orders["order_estimated_delivery_date"],
                             errors="coerce")
    bad_est = orders[est_col < orders["order_purchase_timestamp"]]
    report.log(
        "Estimated delivery before purchase date",
        len(bad_est),
        "ERROR",
    )

    # ── Rule 5: Duplicate orders ──────────────────────────────────────────────
    # An order_id should not appear more than once in the orders table.
    dup_orders = orders.duplicated(subset=["order_id"]).sum()
    report.log("Duplicate order_id in orders table", dup_orders, "ERROR")

    # Duplicate (order_id, product_id) in order_items
    dup_items = order_items.duplicated(subset=["order_id", "product_id"]).sum()
    report.log(
        "Duplicate (order_id, product_id) in order_items",
        dup_items,
        "WARNING",
    )

    # ── Rule 6: Zero-value payments ──────────────────────────────────────────
    zero_payments = (payments["payment_value"] == 0).sum()
    report.log("Zero-value payments", zero_payments, "WARNING")

    # ── Rule 7: Orders with no corresponding payment ─────────────────────────
    paid_orders    = set(payments["order_id"])
    orders_no_pay  = orders[~orders["order_id"].isin(paid_orders)]
    report.log("Orders with no payment record", len(orders_no_pay), "ERROR")

    return report.summary()


# =============================================================================
# STEP 4 – KPI GENERATOR
# =============================================================================

def generate_kpis(dfs: dict[str, pd.DataFrame]) -> dict:
    """Compute and print all requested KPIs. Returns a dict of results."""

    orders      = dfs["orders"]
    order_items = dfs["order_items"]
    payments    = dfs["payments"]
    products    = dfs["products"]
    customers   = dfs["customers"]

    # ── Merge base tables ────────────────────────────────────────────────────
    # Items enriched with product category
    items_prod = order_items.merge(
        products[["product_id", "product_category_name"]],
        on="product_id", how="left"
    )

    # Payments aggregated to one row per order (handles split payments)
    pay_agg = payments.groupby("order_id", as_index=False)["payment_value"].sum()

    # Full order view: orders + payments
    orders_pay = orders.merge(pay_agg, on="order_id", how="left")

    # ── KPI 1: Total Revenue ─────────────────────────────────────────────────
    total_revenue = pay_agg["payment_value"].sum()

    # ── KPI 2: Average Order Value (AOV) ─────────────────────────────────────
    aov = pay_agg["payment_value"].mean()

    # ── KPI 3: Top 10 Categories by Revenue ──────────────────────────────────
    cat_revenue = (
        items_prod
        .merge(pay_agg, on="order_id", how="left")
        .groupby("product_category_name")["payment_value"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={"product_category_name": "category",
                          "payment_value": "revenue"})
    )

    # ── KPI 4: Customer Retention ─────────────────────────────────────────────
    # A customer is "retained" if they placed more than one order.
    order_count_per_cust = (
        orders_pay
        .groupby("customer_id")["order_id"]
        .count()
        .reset_index(name="order_count")
    )
    retained   = (order_count_per_cust["order_count"] > 1).sum()
    total_cust = len(order_count_per_cust)
    retention_rate = retained / total_cust * 100

    # ── KPI 5: Monthly Active Customers ──────────────────────────────────────
    orders_pay["year_month"] = (
        orders_pay["order_purchase_timestamp"].dt.to_period("M")
    )
    monthly_active = (
        orders_pay
        .groupby("year_month")["customer_id"]
        .nunique()
        .reset_index(name="active_customers")
        .sort_values("year_month")
    )

    # ── Print results ─────────────────────────────────────────────────────────
    sep = "─" * 60

    print(f"\n{sep}")
    print("  KPI 1 │ Total Revenue")
    print(f"         ${total_revenue:,.2f}")

    print(f"\n{sep}")
    print("  KPI 2 │ Average Order Value (AOV)")
    print(f"         ${aov:,.2f} per order")

    print(f"\n{sep}")
    print("  KPI 3 │ Top 10 Categories by Revenue")
    for _, row in cat_revenue.iterrows():
        bar = "█" * int(row["revenue"] / cat_revenue["revenue"].max() * 30)
        print(f"         {row['category']:<35s} ${row['revenue']:>12,.0f}  {bar}")

    print(f"\n{sep}")
    print("  KPI 4 │ Customer Retention")
    print(f"         Total customers : {total_cust:,}")
    print(f"         Returning        : {retained:,}")
    print(f"         Retention rate   : {retention_rate:.1f}%")

    print(f"\n{sep}")
    print("  KPI 5 │ Monthly Active Customers")
    for _, row in monthly_active.iterrows():
        bar = "█" * int(row["active_customers"] / monthly_active["active_customers"].max() * 30)
        print(f"         {str(row['year_month']):<10s}  {row['active_customers']:>6,}  {bar}")

    print(f"\n{sep}\n")

    return {
        "total_revenue":     total_revenue,
        "aov":               aov,
        "top_categories":    cat_revenue,
        "retention_rate":    retention_rate,
        "monthly_active":    monthly_active,
    }


# =============================================================================
# MAIN  – wire all steps together
# =============================================================================
def main():
    print("\n" + "=" * 60)
    print("  ECOMMERCE DATA PIPELINE")
    print("=" * 60)

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    print("\n[STEP 1] Loading raw data …")
    dfs = load_data(DATA_DIR)

    # ── Step 2a: Remove duplicates ────────────────────────────────────────────
    print("\n[STEP 2a] Removing duplicates …")
    dfs = remove_duplicates(dfs)

    # ── Step 2b: Standardize text ─────────────────────────────────────────────
    print("\n[STEP 2b] Standardizing text …")
    dfs = standardize_text(dfs)

    # ── Step 2c: Handle nulls ─────────────────────────────────────────────────
    print("\n[STEP 2c] Handling nulls …")
    dfs = handle_nulls(dfs)

    # ── Step 3: Validate schema ───────────────────────────────────────────────
    print("\n[STEP 3] Running schema validation …")
    issues = validate_schema(dfs)

    # ── Step 4: Generate KPIs ─────────────────────────────────────────────────
    print("\n[STEP 4] Generating KPIs …")
    kpis = generate_kpis(dfs)

    print("Pipeline complete.")
    return dfs, issues, kpis


if __name__ == "__main__":
    dfs, issues, kpis = main()
