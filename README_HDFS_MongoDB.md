# HDFS Upload & MongoDB Schema/Data Load

**Responsible:** HDFS upload, MongoDB schema & data load
**Project:** Walmart Sales Forecast Big Data Pipeline (COMP/IERG 7095)

---

## Overview

This module implements two storage components of the big data pipeline:

1. **HDFS Upload** -- Upload raw CSVs and cleaned Parquet files to HDFS with a structured directory hierarchy, ready for downstream Spark processing.
2. **MongoDB Schema & Data Load** -- Create MongoDB collections with validation rules and analytics-optimized indexes, load data, and build a merged dataset.

### How It Fits in the Pipeline

```
 [Kaggle Dataset]
       |
       v
 +---------------------+
 | HDFS Upload (this)   |  raw CSV -> HDFS /raw/
 | + Parquet Cleaning   |  cleaned  -> HDFS /cleaned/ (Parquet)
 +---------------------+
       |
       v
 +---------------------+
 | Spark Processing     |  Reads from HDFS /cleaned/
 | (team member)        |  Joins, Spark SQL, MLlib
 |                      |  Writes to HDFS /enriched/
 +---------------------+
       |
       v
 +---------------------+
 | MongoDB (this)       |  Stores data with indexes for
 |                      |  fast analytical queries
 +---------------------+
       |
       v
 +---------------------+
 | Visualization        |  Reads from MongoDB / Spark results
 | (team member)        |  Matplotlib, Seaborn
 +---------------------+
```

---

## Dataset

| File | Rows | Description |
|------|------|-------------|
| `train.csv` | ~421K | Historical weekly sales (Store, Dept, Date, Weekly_Sales, IsHoliday) |
| `test.csv` | ~115K | Prediction targets (no Weekly_Sales column) |
| `features.csv` | ~8K | Weekly store-level features (Temperature, Fuel_Price, MarkDown1-5, CPI, Unemployment) |
| `stores.csv` | 45 | Store metadata (Type A/B/C, Size) |

---

## 1. HDFS Upload

### HDFS Directory Hierarchy

```
/user/walmart_sales/
    raw/               <- Original CSV files (train.csv, test.csv, features.csv, stores.csv)
    cleaned/           <- Preprocessed Parquet files (date parsed, missing values handled)
        train/
        test/
        features/      (includes hasMarkDown1-5 indicator columns)
        stores/
    enriched/          <- Reserved for Spark output (merged, feature-engineered data)
```

### Prerequisites

```bash
# For raw CSV upload via Hadoop CLI:
# Ensure hadoop is in PATH and HDFS is running

# For Parquet cleaning (--with-cleaning):
pip install pyspark

# For WebHDFS method only:
pip install hdfs
```

### Usage

```bash
cd 7095_proj/

# Upload raw CSVs to HDFS only (Hadoop CLI)
python hdfs_upload/hdfs_upload.py --local-dir ./data

# Upload raw CSVs + clean and convert to Parquet
python hdfs_upload/hdfs_upload.py --local-dir ./data --with-cleaning

# Use WebHDFS instead of CLI
python hdfs_upload/hdfs_upload.py --method webhdfs --hdfs-host namenode-host --hdfs-port 9870

# Custom HDFS directory
python hdfs_upload/hdfs_upload.py --hdfs-base /user/group7/walmart_sales --local-dir ./data --with-cleaning
```

### What the Cleaning Step Does

When `--with-cleaning` is specified, PySpark processes each CSV:

- **Date parsing**: Converts string dates to proper date type
- **MarkDown handling**: Fills missing MarkDown1-5 values with 0.0
- **MarkDown indicators**: Adds `hasMarkDown1`-`hasMarkDown5` columns (1 = promotion active AND date >= 2011-11-01, 0 otherwise)
- **Output format**: Parquet (columnar, compressed, schema-enforced)

### Verification

```bash
hadoop fs -ls -R -h /user/walmart_sales/
hadoop fs -cat /user/walmart_sales/raw/stores.csv | head -5
```

---

## 2. MongoDB Schema & Data Load

### Prerequisites

```bash
pip install pymongo pandas
# MongoDB must be running (localhost:27017 by default)
```

### Database Design

**Database:** `walmart_sales`

#### Collections

| Collection | Documents | Description |
|-----------|-----------|-------------|
| `stores` | 45 | Store metadata: Store (unique), Type (A/B/C), Size |
| `features` | ~8,190 | Weekly features: (Store, Date) unique, Temperature, Fuel_Price, MarkDowns, CPI, Unemployment, hasMarkDown flags |
| `train_sales` | ~421,570 | Training sales: (Store, Dept, Date) unique, Weekly_Sales, IsHoliday |
| `test_sales` | ~115,064 | Test targets: (Store, Dept, Date) unique, IsHoliday |
| `merged_data` | ~421,570 | Joined dataset: train + features + stores + Year/Month/Week |

#### Schema Validation

All collections enforce **JSON Schema validation**:
- Required fields checked on insert/update
- Data types enforced: `int`, `double`, `bool`, `date`, `enum`
- MarkDown fields allow `null` (for rows without feature data after join)

#### Index Design

Indexes are optimized for the project's analytical query patterns:

| Query Pattern | Collection | Index |
|--------------|------------|-------|
| Avg sales by store type | `merged_data` | `(Type, Weekly_Sales DESC)` |
| Holiday vs non-holiday sales | `merged_data` | `(IsHoliday, Weekly_Sales DESC)` |
| Markdown impact on sales | `merged_data` | `(hasMarkDown1, Weekly_Sales DESC)` |
| Time-series trend analysis | `merged_data` | `(Date)`, `(Year, Month)` |
| Per-store drill-down | `merged_data` | `(Store, Dept, Date)` unique |
| Top sales ranking | `merged_data` | `(Weekly_Sales DESC)` |

### Usage

```bash
cd 7095_proj/

# Full pipeline: schema + load all data + merge + verify
python mongodb_load/mongodb_data_load.py --local-dir ./data

# Custom MongoDB connection
python mongodb_load/mongodb_data_load.py --mongo-uri "mongodb://user:pass@host:27017" --local-dir ./data

# Schema/indexes only (no data)
python mongodb_load/mongodb_schema.py

# Data only (skip schema creation)
python mongodb_load/mongodb_data_load.py --local-dir ./data --skip-schema
```

### Data Load Pipeline Steps

```
Step 1/4: Initialize schema (create collections + validation + indexes)
Step 2/4: Load raw CSVs into collections (stores, features, train_sales, test_sales)
Step 3/4: Create merged_data via aggregation pipeline ($lookup joins + $merge)
Step 4/4: Print verification summary with sample analytical queries
```

### Verification (Mongo Shell)

```javascript
use walmart_sales

// Check document counts
db.stores.countDocuments()        // -> 45
db.features.countDocuments()      // -> ~8190
db.train_sales.countDocuments()   // -> ~421570
db.test_sales.countDocuments()    // -> ~115064
db.merged_data.countDocuments()   // -> ~421570

// Analytical query: average sales by store type
db.merged_data.aggregate([
  { $group: { _id: "$Type", avg_sales: { $avg: "$Weekly_Sales" } } },
  { $sort: { avg_sales: -1 } }
])

// Analytical query: holiday impact
db.merged_data.aggregate([
  { $group: { _id: "$IsHoliday", avg_sales: { $avg: "$Weekly_Sales" } } }
])

// Analytical query: markdown promotion impact
db.merged_data.aggregate([
  { $group: { _id: "$hasMarkDown1", avg_sales: { $avg: "$Weekly_Sales" }, count: { $sum: 1 } } }
])

// Check indexes
db.merged_data.getIndexes()
```

---

## File Structure

```
hdfs_upload/
    hdfs_upload.py              # HDFS upload + optional Parquet cleaning

mongodb_load/
    mongodb_schema.py           # Schema definitions, validators, indexes
    mongodb_data_load.py        # Data loading + merge pipeline + verification

README_HDFS_MongoDB.md          # This file
```

## Dependencies

```
pymongo>=4.0
pandas>=1.5
pyspark>=3.3     # for HDFS Parquet cleaning
hdfs>=2.7        # for WebHDFS method only
```
