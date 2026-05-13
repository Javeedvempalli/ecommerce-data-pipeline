"""
Ecommerce ETL Pipeline
======================
Flow:
  Raw CSV → Validation → Cleaning → Transformation → Analytics → PostgreSQL

Output:
  - logs/etl_YYYYMMDD_HHMMSS.log   (full run log)
  - logs/errors_YYYYMMDD_HHMMSS.csv (skipped bad rows)
  - PostgreSQL tables:
      dim_customers, dim_products, fact_orders,
      analytics_revenue, analytics_monthly_active, analytics_top_categories

Setup (run once in CMD):
  pip install pandas numpy psycopg2-binary sqlalchemy
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — update these before running
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR   = "train/"                  # folder containing the 5 CSVs
OUTPUT_DIR = "etl_output"             # logs + error files saved here

# PostgreSQL connection — update password to yours
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "ecommerce_db",       # will be created if it doesn't exist
    "user":     "postgres",
    "password": "javeedsql", # <-- CHANGE THIS
}

TODAY = datetime.today()


# =============================================================================
# LOGGER SETUP
# =============================================================================

def setup_logger(output_dir: str) -> tuple[logging.Logger, str, str]:
    """Create log file and error CSV path. Returns logger + both file paths."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path    = os.path.join(output_dir, f"etl_{timestamp}.log")
    error_path  = os.path.join(output_dir, f"errors_{timestamp}.csv")

    logger = logging.getLogger("etl_pipeline")
    logger.setLevel(logging.DEBUG)

    # File handler — full detail
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # Console handler — info and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)-8s | %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger, log_path, error_path


# =============================================================================
# STEP 1 — EXTRACT (Raw CSV Load)
# =============================================================================

def extract(data_dir: str, logger: logging.Logger) -> dict[str, pd.DataFrame]:
    """Load all 5 raw CSV files."""
    logger.info("=" * 60)
    logger.info("STEP 1 — EXTRACT")
    logger.info("=" * 60)

    files = {
        "customers":   "df_Customers.csv",
        "orders":      "df_Orders.csv",
        "order_items": "df_OrderItems.csv",
        "payments":    "df_Payments.csv",
        "products":    "df_Products.csv",
    }

    raw = {}
    for key, filename in files.items():
        path = os.path.join(data_dir, filename)
        try:
            raw[key] = pd.read_csv(path)
            logger.info(f"  Loaded {filename:30s}  rows={len(raw[key]):,}  cols={raw[key].shape[1]}")
        except FileNotFoundError:
            logger.error(f"  MISSING FILE: {path}")
            raise

    return raw


# =============================================================================
# STEP 2 — VALIDATE
# =============================================================================

def validate(
    dfs: dict[str, pd.DataFrame],
    logger: logging.Logger,
    error_path: str
) -> dict[str, pd.DataFrame]:
    """
    Run validation rules. Bad rows are logged to the error CSV and
    dropped — pipeline continues with clean rows only.
    """
    logger.info("=" * 60)
    logger.info("STEP 2 — VALIDATE")
    logger.info("=" * 60)

    error_records = []

    def flag_bad(table: str, bad_df: pd.DataFrame, rule: str):
        """Log and collect bad rows."""
        if len(bad_df) == 0:
            logger.info(f"  [PASS] {rule}")
            return
        logger.warning(f"  [SKIP] {rule} — {len(bad_df):,} bad row(s) dropped")
        for idx, row in bad_df.iterrows():
            error_records.append({
                "table":   table,
                "rule":    rule,
                "row_idx": idx,
                "data":    str(row.to_dict()),
            })

    customers   = dfs["customers"].copy()
    orders      = dfs["orders"].copy()
    order_items = dfs["order_items"].copy()
    payments    = dfs["payments"].copy()
    products    = dfs["products"].copy()

    # Parse dates early so comparisons work
    for col in ["order_purchase_timestamp", "order_approved_at",
                "order_delivered_timestamp", "order_estimated_delivery_date"]:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    # ── Rule 1: Invalid customer IDs ─────────────────────────────────────────
    valid_cust = set(customers["customer_id"])
    bad = orders[~orders["customer_id"].isin(valid_cust)]
    flag_bad("orders", bad, "Invalid customer_id in orders")
    orders = orders[orders["customer_id"].isin(valid_cust)]

    # ── Rule 2: Invalid product IDs ──────────────────────────────────────────
    valid_prod = set(products["product_id"])
    bad = order_items[~order_items["product_id"].isin(valid_prod)]
    flag_bad("order_items", bad, "Invalid product_id in order_items")
    order_items = order_items[order_items["product_id"].isin(valid_prod)]

    # ── Rule 3: Negative prices ───────────────────────────────────────────────
    bad = order_items[order_items["price"] < 0]
    flag_bad("order_items", bad, "Negative price")
    order_items = order_items[order_items["price"] >= 0]

    bad = order_items[order_items["shipping_charges"] < 0]
    flag_bad("order_items", bad, "Negative shipping_charges")
    order_items = order_items[order_items["shipping_charges"] >= 0]

    bad = payments[payments["payment_value"] < 0]
    flag_bad("payments", bad, "Negative payment_value")
    payments = payments[payments["payment_value"] >= 0]

    # ── Rule 4: Future purchase dates ─────────────────────────────────────────
    bad = orders[orders["order_purchase_timestamp"] > TODAY]
    flag_bad("orders", bad, "Future order_purchase_timestamp")
    orders = orders[orders["order_purchase_timestamp"] <= TODAY]

    # ── Rule 5: Estimated delivery before purchase ────────────────────────────
    bad = orders[orders["order_estimated_delivery_date"] < orders["order_purchase_timestamp"]]
    flag_bad("orders", bad, "Estimated delivery before purchase date")
    orders = orders[
        orders["order_estimated_delivery_date"] >= orders["order_purchase_timestamp"]
    ]

    # ── Rule 6: Duplicate order_ids ───────────────────────────────────────────
    bad = orders[orders.duplicated(subset=["order_id"], keep="first")]
    flag_bad("orders", bad, "Duplicate order_id")
    orders = orders.drop_duplicates(subset=["order_id"], keep="first")

    # ── Rule 7: Zero-value payments ───────────────────────────────────────────
    bad = payments[payments["payment_value"] == 0]
    flag_bad("payments", bad, "Zero-value payment")
    payments = payments[payments["payment_value"] > 0]

    # ── Rule 8: Null primary keys ─────────────────────────────────────────────
    for name, df, pk in [
        ("customers",   customers,   "customer_id"),
        ("orders",      orders,      "order_id"),
        ("order_items", order_items, "order_id"),
        ("payments",    payments,    "order_id"),
        ("products",    products,    "product_id"),
    ]:
        bad = df[df[pk].isna()]
        flag_bad(name, bad, f"Null primary key ({pk})")

    # Save error report
    if error_records:
        err_df = pd.DataFrame(error_records)
        err_df.to_csv(error_path, index=False)
        logger.warning(f"  Error report saved → {error_path}")
    else:
        logger.info("  No errors found — error report not created")

    total_errors = len(error_records)
    logger.info(f"  Validation complete — {total_errors} row(s) skipped total")

    return {
        "customers":   customers,
        "orders":      orders,
        "order_items": order_items,
        "payments":    payments,
        "products":    products,
    }


# =============================================================================
# STEP 3 — CLEAN
# =============================================================================

def clean(
    dfs: dict[str, pd.DataFrame],
    logger: logging.Logger
) -> dict[str, pd.DataFrame]:
    """Remove duplicates, standardize text, fill nulls."""
    logger.info("=" * 60)
    logger.info("STEP 3 — CLEAN")
    logger.info("=" * 60)

    # ── Remove duplicates ─────────────────────────────────────────────────────
    before = len(dfs["products"])
    dfs["products"] = dfs["products"].drop_duplicates(
        subset=["product_id"], keep="first"
    ).reset_index(drop=True)
    after = len(dfs["products"])
    logger.info(f"  [products] removed {before - after:,} duplicate rows")

    # ── Standardize text ──────────────────────────────────────────────────────
    for name, df in dfs.items():
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            df[col] = df[col].str.strip().str.lower()
        dfs[name] = df
    logger.info("  Text standardized (strip + lowercase) across all tables")

    # ── Handle nulls — orders ─────────────────────────────────────────────────
    o = dfs["orders"]
    n = o["order_approved_at"].isna().sum()
    o["order_approved_at"] = o["order_approved_at"].fillna(
        o["order_purchase_timestamp"]
    )
    o["is_delivered"] = o["order_delivered_timestamp"].notna()
    logger.info(f"  [orders] filled {n} null order_approved_at values")
    logger.info(f"  [orders] added is_delivered flag")
    dfs["orders"] = o

    # ── Handle nulls — products ───────────────────────────────────────────────
    p = dfs["products"]
    n_cat = p["product_category_name"].isna().sum()
    p["product_category_name"] = p["product_category_name"].fillna("unknown_category")
    for col in ["product_weight_g", "product_length_cm",
                "product_height_cm", "product_width_cm"]:
        n = p[col].isna().sum()
        p[col] = p[col].fillna(p[col].median())
        if n:
            logger.info(f"  [products] filled {n} nulls in {col} with median")
    logger.info(f"  [products] filled {n_cat} null category names")
    dfs["products"] = p

    logger.info("  Cleaning complete")
    return dfs


# =============================================================================
# STEP 4 — TRANSFORM
# =============================================================================

def transform(
    dfs: dict[str, pd.DataFrame],
    logger: logging.Logger
) -> dict[str, pd.DataFrame]:
    """
    Build final analytical tables:
      dim_customers        — customer dimension
      dim_products         — product dimension
      fact_orders          — central fact table
      analytics_revenue    — revenue KPIs
      analytics_monthly    — monthly active customers
      analytics_categories — top categories by revenue
    """
    logger.info("=" * 60)
    logger.info("STEP 4 — TRANSFORM")
    logger.info("=" * 60)

    customers   = dfs["customers"]
    orders      = dfs["orders"]
    order_items = dfs["order_items"]
    payments    = dfs["payments"]
    products    = dfs["products"]

    # ── dim_customers ─────────────────────────────────────────────────────────
    dim_customers = customers.rename(columns={
        "customer_id":               "customer_id",
        "customer_zip_code_prefix":  "zip_code",
        "customer_city":             "city",
        "customer_state":            "state",
    })
    logger.info(f"  dim_customers        rows={len(dim_customers):,}")

    # ── dim_products ──────────────────────────────────────────────────────────
    dim_products = products.rename(columns={
        "product_id":             "product_id",
        "product_category_name":  "category",
        "product_weight_g":       "weight_g",
        "product_length_cm":      "length_cm",
        "product_height_cm":      "height_cm",
        "product_width_cm":       "width_cm",
    })
    logger.info(f"  dim_products         rows={len(dim_products):,}")

    # ── Aggregate payments (one row per order) ────────────────────────────────
    pay_agg = payments.groupby("order_id", as_index=False).agg(
        total_payment=("payment_value", "sum"),
        payment_installments=("payment_installments", "max"),
        payment_type=("payment_type", "first"),
    )

    # ── fact_orders ───────────────────────────────────────────────────────────
    fact_orders = (
        orders
        .merge(pay_agg, on="order_id", how="left")
        .merge(
            order_items.groupby("order_id", as_index=False).agg(
                item_count=("product_id", "count"),
                items_revenue=("price", "sum"),
                shipping_total=("shipping_charges", "sum"),
            ),
            on="order_id", how="left"
        )
    )
    # Derived columns
    fact_orders["order_purchase_date"] = pd.to_datetime(
        fact_orders["order_purchase_timestamp"]
    ).dt.date
    fact_orders["delivery_days"] = (
        fact_orders["order_delivered_timestamp"] -
        fact_orders["order_purchase_timestamp"]
    ).dt.days
    logger.info(f"  fact_orders          rows={len(fact_orders):,}")

    # ── analytics_revenue ─────────────────────────────────────────────────────
    total_revenue  = pay_agg["total_payment"].sum()
    total_orders   = len(pay_agg)
    aov            = pay_agg["total_payment"].mean()
    avg_items      = fact_orders["item_count"].mean()
    avg_delivery   = fact_orders["delivery_days"].mean()

    analytics_revenue = pd.DataFrame([{
        "metric":                 "summary",
        "total_revenue":          round(total_revenue, 2),
        "total_orders":           total_orders,
        "avg_order_value":        round(aov, 2),
        "avg_items_per_order":    round(avg_items, 2),
        "avg_delivery_days":      round(avg_delivery, 1),
        "generated_at":           datetime.now(),
    }])
    logger.info(f"  analytics_revenue    rows={len(analytics_revenue):,}")

    # ── analytics_monthly ─────────────────────────────────────────────────────
    fact_orders["year_month"] = pd.to_datetime(
        fact_orders["order_purchase_timestamp"]
    ).dt.to_period("M").astype(str)

    analytics_monthly = (
        fact_orders
        .groupby("year_month")
        .agg(
            active_customers=("customer_id", "nunique"),
            total_orders=("order_id", "count"),
            monthly_revenue=("total_payment", "sum"),
        )
        .reset_index()
        .sort_values("year_month")
    )
    analytics_monthly["monthly_revenue"] = analytics_monthly["monthly_revenue"].round(2)
    logger.info(f"  analytics_monthly    rows={len(analytics_monthly):,}")

    # ── analytics_categories ──────────────────────────────────────────────────
    items_cat = order_items.merge(
        products[["product_id", "product_category_name"]],
        on="product_id", how="left"
    ).merge(pay_agg[["order_id", "total_payment"]], on="order_id", how="left")

    analytics_categories = (
        items_cat
        .groupby("product_category_name")
        .agg(
            total_revenue=("total_payment", "sum"),
            total_orders=("order_id", "nunique"),
            avg_price=("price", "mean"),
        )
        .reset_index()
        .rename(columns={"product_category_name": "category"})
        .sort_values("total_revenue", ascending=False)
        .reset_index(drop=True)
    )
    analytics_categories["total_revenue"] = analytics_categories["total_revenue"].round(2)
    analytics_categories["avg_price"]     = analytics_categories["avg_price"].round(2)
    logger.info(f"  analytics_categories rows={len(analytics_categories):,}")

    logger.info("  Transformation complete")

    return {
        "dim_customers":        dim_customers,
        "dim_products":         dim_products,
        "fact_orders":          fact_orders,
        "analytics_revenue":    analytics_revenue,
        "analytics_monthly":    analytics_monthly,
        "analytics_categories": analytics_categories,
    }


# =============================================================================
# STEP 5 — LOAD TO POSTGRESQL
# =============================================================================

def create_database_if_not_exists(config: dict, logger: logging.Logger):
    """Connect to the default postgres DB and create ecommerce_db if missing."""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=config["host"],
            port=config["port"],
            dbname="postgres",
            user=config["user"],
            password=config["password"],
        )
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (config["database"],)
        )
        if not cur.fetchone():
            cur.execute(f'CREATE DATABASE {config["database"]}')
            logger.info(f"  Created database: {config['database']}")
        else:
            logger.info(f"  Database already exists: {config['database']}")
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"  Could not create database: {e}")
        raise


def load(
    tables: dict[str, pd.DataFrame],
    config: dict,
    logger: logging.Logger
):
    """Load all transformed tables into PostgreSQL."""
    logger.info("=" * 60)
    logger.info("STEP 5 — LOAD TO POSTGRESQL")
    logger.info("=" * 60)

    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        logger.error("sqlalchemy not installed. Run: pip install sqlalchemy")
        raise

    # Create DB if needed
    create_database_if_not_exists(config, logger)

    # Build connection string
    conn_str = (
        f"postgresql+psycopg2://{config['user']}:{config['password']}"
        f"@{config['host']}:{config['port']}/{config['database']}"
    )

    try:
        engine = create_engine(conn_str)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info(f"  Connected to PostgreSQL → {config['database']}")
    except Exception as e:
        logger.error(f"  Connection failed: {e}")
        raise

    # Load each table
    load_order = [
        ("dim_customers",        "replace"),
        ("dim_products",         "replace"),
        ("fact_orders",          "replace"),
        ("analytics_revenue",    "replace"),
        ("analytics_monthly",    "replace"),
        ("analytics_categories", "replace"),
    ]

    for table_name, if_exists in load_order:
        df = tables[table_name]
        try:
            # Convert Period/date columns to string for PostgreSQL compatibility
            for col in df.select_dtypes(include=["period", "object"]).columns:
                df[col] = df[col].astype(str)
            for col in df.select_dtypes(include=["datetimetz"]).columns:
                df[col] = df[col].dt.tz_localize(None)

            df.to_sql(
                table_name,
                engine,
                if_exists=if_exists,
                index=False,
                method="multi",
                chunksize=1000,
            )
            logger.info(f"  Loaded {table_name:30s} → {len(df):,} rows")
        except Exception as e:
            logger.error(f"  Failed to load {table_name}: {e}")
            raise

    logger.info("  All tables loaded successfully")
    engine.dispose()


# =============================================================================
# MAIN — Wire all steps together
# =============================================================================

def main():
    # Setup
    logger, log_path, error_path = setup_logger(OUTPUT_DIR)

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║           ECOMMERCE ETL PIPELINE STARTING               ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info(f"  Log file    → {log_path}")
    logger.info(f"  Error file  → {error_path}")
    logger.info(f"  Data source → {DATA_DIR}")
    logger.info(f"  Database    → {DB_CONFIG['database']} on {DB_CONFIG['host']}")

    start_time = datetime.now()

    try:
        # Step 1 — Extract
        raw = extract(DATA_DIR, logger)

        # Step 2 — Validate (bad rows skipped, logged to error CSV)
        validated = validate(raw, logger, error_path)

        # Step 3 — Clean
        cleaned = clean(validated, logger)

        # Step 4 — Transform
        transformed = transform(cleaned, logger)

        # Step 5 — Load to PostgreSQL
        load(transformed, DB_CONFIG, logger)

        elapsed = (datetime.now() - start_time).seconds
        logger.info("=" * 60)
        logger.info(f"  PIPELINE COMPLETE in {elapsed}s")
        logger.info(f"  Tables created in PostgreSQL database '{DB_CONFIG['database']}':")
        for t in transformed:
            logger.info(f"    • {t}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"  PIPELINE FAILED: {e}")
        logger.error("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
