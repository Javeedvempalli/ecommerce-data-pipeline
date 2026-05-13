# 🛒 Ecommerce Data Pipeline

An automated end-to-end data pipeline that processes raw e-commerce CSV data through validation, cleaning, transformation, and loads analysis-ready tables into PostgreSQL — with a separate analytics script for KPI reporting.

---

## 📁 Repository Structure

```
ecommerce-data-pipeline/
│
├── etl_pipeline.py          # Full ETL pipeline → loads data into PostgreSQL
├── ecommerce_pipeline.py    # Analysis script → prints KPIs to console
└── README.md
```

---

## 🔁 Pipeline Flow

```
Raw CSV Files → Validate → Clean → Transform → Load to PostgreSQL
```

| Step | What Happens |
|------|-------------|
| **Extract** | Loads 5 raw CSV files from the `train/` folder |
| **Validate** | Checks for bad data — invalid IDs, negative prices, duplicate orders, future dates |
| **Clean** | Removes duplicates, standardizes text, fills missing values |
| **Transform** | Builds star schema tables (dim, fact, analytics) |
| **Load** | Writes all 6 tables into PostgreSQL |

---

## 📊 Output Tables (PostgreSQL)

| Table | Description |
|-------|-------------|
| `dim_customers` | Customer master data — city, state, zip code |
| `dim_products` | Product master data — category, dimensions, weight |
| `fact_orders` | Central fact table — orders with payment, items, delivery days |
| `analytics_revenue` | Summary KPIs — total revenue, AOV, avg delivery days |
| `analytics_monthly` | Month-by-month active customers and revenue |
| `analytics_categories` | Revenue breakdown by product category |

---

## 📈 KPIs Generated (`ecommerce_pipeline.py`)

- **Total Revenue**
- **Average Order Value (AOV)**
- **Top 10 Categories by Revenue**
- **Customer Retention Rate**
- **Monthly Active Customers**

---

## 🗂️ Input Data

Place the following CSV files inside a `train/` folder:

```
train/
├── df_Customers.csv
├── df_Orders.csv
├── df_OrderItems.csv
├── df_Payments.csv
└── df_Products.csv
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| PostgreSQL | Target database |
| SQLAlchemy | Database ORM / connection |
| psycopg2 | PostgreSQL driver |

---

## ⚙️ Setup & Installation

**1. Install dependencies**
```bash
pip install pandas numpy psycopg2-binary sqlalchemy
```

**2. Configure your database**

Open `etl_pipeline.py` and update the `DB_CONFIG` section:
```python
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "ecommerce_db",
    "user":     "postgres",
    "password": "your_password_here",   # <-- update this
}
```

**3. Run the ETL pipeline**
```bash
python etl_pipeline.py
```

**4. Run the KPI analysis script**
```bash
python ecommerce_pipeline.py
```

---

## 📋 Validation Rules

The pipeline automatically checks and removes bad data:

| Rule | Action |
|------|--------|
| Orders with invalid customer IDs | Dropped |
| Order items with invalid product IDs | Dropped |
| Negative prices or payment values | Dropped |
| Orders with future purchase dates | Dropped |
| Estimated delivery before purchase date | Dropped |
| Duplicate order IDs | Deduplicated |
| Zero-value payments | Dropped |
| Null primary keys | Dropped |

All rejected rows are saved to `etl_output/errors_YYYYMMDD_HHMMSS.csv` for auditing.

---

## 📝 Logs & Error Reports

Every pipeline run generates:
- `etl_output/etl_YYYYMMDD_HHMMSS.log` — full run log with timestamps
- `etl_output/errors_YYYYMMDD_HHMMSS.csv` — all dropped rows with the reason

---

## 🆚 Script Comparison

| Feature | `etl_pipeline.py` | `ecommerce_pipeline.py` |
|---------|:-----------------:|:-----------------------:|
| Loads to PostgreSQL | ✅ | ❌ |
| Prints KPIs to console | ❌ | ✅ |
| Customer retention KPI | ❌ | ✅ |
| Saves log file | ✅ | ❌ |
| Saves error report CSV | ✅ | ❌ |
| Error handling (try/except) | ✅ | ❌ |
| Best for | Production / Scheduled runs | Quick analysis |

---

## 👤 Author

**Javeed Vempalli**  
[GitHub](https://github.com/Javeedvempalli)
