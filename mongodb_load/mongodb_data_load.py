"""
MongoDB Data Load Script for Walmart Sales Forecast Project
============================================================
Loads CSV data into MongoDB collections with preprocessing aligned to the
team's data preprocessing pipeline:
  - Date parsing
  - Missing MarkDown values filled with 0.0
  - hasMarkDown1-5 binary indicator columns
  - Merged dataset via aggregation pipeline (train + features + stores)
  - Temporal features (Year, Month, Week)

Supports both raw data loading (with preprocessing) and preprocessed data loading.

Usage:
    # Load raw data with preprocessing
    python mongodb_data_load.py --local-dir ./data
    python mongodb_data_load.py --mongo-uri "mongodb://user:pass@host:27017" --local-dir ./data

    # Load preprocessed data directly
    python mongodb_data_load.py --preprocessed --local-dir ./data/preprocessed
    python mongodb_data_load.py --preprocessed --local-dir ./data/preprocessed --skip-schema

Prerequisites:
    pip install pymongo pandas
"""

import os
import argparse
import logging
from datetime import datetime

import pandas as pd
from pymongo import MongoClient, UpdateOne

from mongodb_schema import DB_NAME, create_collections

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 5000

# MarkDown data is only available from this date onward
MARKDOWN_CUTOFF = datetime(2011, 11, 1)


def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime. Supports DD/MM/YYYY, YYYY-MM-DD, MM/DD/YYYY."""
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {date_str}")


def bulk_upsert(collection, operations: list):
    """Execute bulk upsert and clear the operations list."""
    if operations:
        result = collection.bulk_write(operations)
        return result.upserted_count + result.modified_count
    return 0


# ================================================================
# Load individual collections
# ================================================================

def load_stores(db, local_dir: str):
    """Load stores.csv -> stores collection."""
    filepath = os.path.join(local_dir, "stores.csv")
    df = pd.read_csv(filepath)
    logger.info(f"Loading {len(df)} stores from {filepath}")

    collection = db["stores"]
    operations = [
        UpdateOne(
            {"Store": int(row["Store"])},
            {"$set": {
                "Store": int(row["Store"]),
                "Type": str(row["Type"]),
                "Size": int(row["Size"]),
            }},
            upsert=True,
        )
        for _, row in df.iterrows()
    ]

    result = collection.bulk_write(operations)
    logger.info(
        f"  stores: {result.upserted_count} inserted, {result.modified_count} updated")


def load_features(db, local_dir: str):
    """
    Load features.csv -> features collection.
    Preprocessing:
      - Parse dates
      - Fill missing MarkDown values with 0.0
      - Add hasMarkDown1-5 binary indicators (1 if promotion exists AND date >= 2011-11-01)
    """
    filepath = os.path.join(local_dir, "features.csv")
    df = pd.read_csv(filepath)
    logger.info(f"Loading {len(df)} feature records from {filepath}")

    collection = db["features"]
    operations = []
    total = 0

    for _, row in df.iterrows():
        date_val = parse_date(str(row["Date"]))

        record = {
            "Store": int(row["Store"]),
            "Date": date_val,
            "Temperature": float(row["Temperature"]),
            "Fuel_Price": float(row["Fuel_Price"]),
            "CPI": float(row["CPI"]),
            "Unemployment": float(row["Unemployment"]),
            "IsHoliday": bool(row["IsHoliday"]),
        }

        # Process MarkDown columns with indicators
        for i in range(1, 6):
            md_col = f"MarkDown{i}"
            has_col = f"hasMarkDown{i}"
            md_val = row[md_col]

            if pd.notna(md_val) and date_val >= MARKDOWN_CUTOFF:
                record[md_col] = float(md_val)
                record[has_col] = 1
            else:
                record[md_col] = 0.0
                record[has_col] = 0

        operations.append(
            UpdateOne(
                {"Store": record["Store"], "Date": record["Date"]},
                {"$set": record},
                upsert=True,
            )
        )

        if len(operations) >= BATCH_SIZE:
            total += bulk_upsert(collection, operations)
            operations = []

    total += bulk_upsert(collection, operations)
    logger.info(
        f"  features: {total} documents written, {collection.count_documents({})} total")


def load_sales(db, local_dir: str, filename: str, collection_name: str):
    """Load train.csv or test.csv into the specified collection."""
    filepath = os.path.join(local_dir, filename)
    df = pd.read_csv(filepath)
    logger.info(
        f"Loading {len(df)} records from {filepath} -> '{collection_name}'")

    collection = db[collection_name]
    operations = []
    total = 0

    for _, row in df.iterrows():
        record = {
            "Store": int(row["Store"]),
            "Dept": int(row["Dept"]),
            "Date": parse_date(str(row["Date"])),
            "IsHoliday": bool(row["IsHoliday"]),
        }
        if "Weekly_Sales" in row.index and pd.notna(row["Weekly_Sales"]):
            record["Weekly_Sales"] = float(row["Weekly_Sales"])

        operations.append(
            UpdateOne(
                {"Store": record["Store"],
                    "Dept": record["Dept"], "Date": record["Date"]},
                {"$set": record},
                upsert=True,
            )
        )

        if len(operations) >= BATCH_SIZE:
            total += bulk_upsert(collection, operations)
            operations = []

    total += bulk_upsert(collection, operations)
    logger.info(
        f"  {collection_name}: {total} documents written, {collection.count_documents({})} total")


# ================================================================
# Load Preprocessed Data
# ================================================================

def load_preprocessed_merged_data(db, preprocessed_dir: str):
    """
    Load preprocessed merged_train.csv directly into merged_data collection.
    The preprocessed data already contains all merged features and temporal fields.
    """
    filepath = os.path.join(preprocessed_dir, "merged_train.csv")
    df = pd.read_csv(filepath)
    logger.info(
        f"Loading {len(df)} preprocessed merged records from {filepath}")

    collection = db["merged_data"]
    operations = []
    total = 0

    for _, row in df.iterrows():
        record = {
            "Store": int(row["Store"]),
            "Dept": int(row["Dept"]),
            "Date": parse_date(str(row["Date"])),
            "Weekly_Sales": float(row["Weekly_Sales"]),
            "IsHoliday": bool(row["IsHoliday_x"]) if "IsHoliday_x" in row else bool(row.get("IsHoliday", False)),
            # Features
            "Temperature": float(row["Temperature"]) if pd.notna(row.get("Temperature")) else None,
            "Fuel_Price": float(row["Fuel_Price"]) if pd.notna(row.get("Fuel_Price")) else None,
            "MarkDown1": float(row["MarkDown1"]) if pd.notna(row.get("MarkDown1")) else 0.0,
            "MarkDown2": float(row["MarkDown2"]) if pd.notna(row.get("MarkDown2")) else 0.0,
            "MarkDown3": float(row["MarkDown3"]) if pd.notna(row.get("MarkDown3")) else 0.0,
            "MarkDown4": float(row["MarkDown4"]) if pd.notna(row.get("MarkDown4")) else 0.0,
            "MarkDown5": float(row["MarkDown5"]) if pd.notna(row.get("MarkDown5")) else 0.0,
            "hasMarkDown1": int(row["hasMarkDown1"]) if pd.notna(row.get("hasMarkDown1")) else 0,
            "hasMarkDown2": int(row["hasMarkDown2"]) if pd.notna(row.get("hasMarkDown2")) else 0,
            "hasMarkDown3": int(row["hasMarkDown3"]) if pd.notna(row.get("hasMarkDown3")) else 0,
            "hasMarkDown4": int(row["hasMarkDown4"]) if pd.notna(row.get("hasMarkDown4")) else 0,
            "hasMarkDown5": int(row["hasMarkDown5"]) if pd.notna(row.get("hasMarkDown5")) else 0,
            "CPI": float(row["CPI"]) if pd.notna(row.get("CPI")) else None,
            "Unemployment": float(row["Unemployment"]) if pd.notna(row.get("Unemployment")) else None,
            # Store info
            "Type": str(row["Type"]),
            "Size": int(row["Size"]),
            # Temporal features
            "Year": int(row["Year"]),
            "Month": int(row["Month"]),
            "Week": int(row["Week"]),
        }

        operations.append(
            UpdateOne(
                {"Store": record["Store"],
                    "Dept": record["Dept"], "Date": record["Date"]},
                {"$set": record},
                upsert=True,
            )
        )

        if len(operations) >= BATCH_SIZE:
            total += bulk_upsert(collection, operations)
            operations = []

    total += bulk_upsert(collection, operations)
    logger.info(
        f"  merged_data: {total} documents written, {collection.count_documents({})} total")
    return total


def load_preprocessed_features(db, preprocessed_dir: str):
    """Load preprocessed features_cleaned.csv into features collection."""
    filepath = os.path.join(preprocessed_dir, "features_cleaned.csv")
    df = pd.read_csv(filepath)
    logger.info(
        f"Loading {len(df)} preprocessed feature records from {filepath}")

    collection = db["features"]
    operations = []
    total = 0

    for _, row in df.iterrows():
        date_val = parse_date(str(row["Date"]))

        record = {
            "Store": int(row["Store"]),
            "Date": date_val,
            "Temperature": float(row["Temperature"]) if pd.notna(row.get("Temperature")) else None,
            "Fuel_Price": float(row["Fuel_Price"]) if pd.notna(row.get("Fuel_Price")) else None,
            "CPI": float(row["CPI"]) if pd.notna(row.get("CPI")) else None,
            "Unemployment": float(row["Unemployment"]) if pd.notna(row.get("Unemployment")) else None,
            "IsHoliday": bool(row["IsHoliday"]),
        }

        # Process MarkDown columns with indicators
        for i in range(1, 6):
            md_col = f"MarkDown{i}"
            has_col = f"hasMarkDown{i}"
            record[md_col] = float(row[md_col]) if pd.notna(
                row.get(md_col)) else 0.0
            record[has_col] = int(row[has_col]) if pd.notna(
                row.get(has_col)) else 0

        operations.append(
            UpdateOne(
                {"Store": record["Store"], "Date": record["Date"]},
                {"$set": record},
                upsert=True,
            )
        )

        if len(operations) >= BATCH_SIZE:
            total += bulk_upsert(collection, operations)
            operations = []

    total += bulk_upsert(collection, operations)
    logger.info(
        f"  features: {total} documents written, {collection.count_documents({})} total")


def load_preprocessed_stores(db, preprocessed_dir: str):
    """Load preprocessed stores.csv into stores collection."""
    filepath = os.path.join(preprocessed_dir, "stores.csv")
    df = pd.read_csv(filepath)
    logger.info(f"Loading {len(df)} stores from {filepath}")

    collection = db["stores"]
    operations = [
        UpdateOne(
            {"Store": int(row["Store"])},
            {"$set": {
                "Store": int(row["Store"]),
                "Type": str(row["Type"]),
                "Size": int(row["Size"]),
            }},
            upsert=True,
        )
        for _, row in df.iterrows()
    ]

    result = collection.bulk_write(operations)
    logger.info(
        f"  stores: {result.upserted_count} inserted, {result.modified_count} updated")


# ================================================================
# Merged dataset via aggregation pipeline (for raw data)
# ================================================================

def create_merged_data(db):
    """
    Create merged_data collection via MongoDB aggregation pipeline.
    Joins: train_sales + features (on Store, Date) + stores (on Store)
    Adds: Year, Month, Week temporal features
    Includes: hasMarkDown1-5 indicators from features
    """
    logger.info("Creating merged dataset via aggregation pipeline...")

    pipeline = [
        # --- Join with features on (Store, Date) ---
        {
            "$lookup": {
                "from": "features",
                "let": {"store": "$Store", "date": "$Date"},
                "pipeline": [
                    {"$match": {"$expr": {"$and": [
                        {"$eq": ["$Store", "$$store"]},
                        {"$eq": ["$Date", "$$date"]},
                    ]}}},
                ],
                "as": "feat",
            }
        },
        {"$unwind": {"path": "$feat", "preserveNullAndEmptyArrays": True}},

        # --- Join with stores on Store ---
        {
            "$lookup": {
                "from": "stores",
                "localField": "Store",
                "foreignField": "Store",
                "as": "st",
            }
        },
        {"$unwind": {"path": "$st", "preserveNullAndEmptyArrays": True}},

        # --- Project all fields + temporal features ---
        {
            "$project": {
                "_id": 0,
                "Store": 1,
                "Dept": 1,
                "Date": 1,
                "Weekly_Sales": 1,
                "IsHoliday": 1,
                # Features
                "Temperature": {"$ifNull": ["$feat.Temperature", None]},
                "Fuel_Price": {"$ifNull": ["$feat.Fuel_Price", None]},
                "MarkDown1": {"$ifNull": ["$feat.MarkDown1", 0.0]},
                "MarkDown2": {"$ifNull": ["$feat.MarkDown2", 0.0]},
                "MarkDown3": {"$ifNull": ["$feat.MarkDown3", 0.0]},
                "MarkDown4": {"$ifNull": ["$feat.MarkDown4", 0.0]},
                "MarkDown5": {"$ifNull": ["$feat.MarkDown5", 0.0]},
                "hasMarkDown1": {"$ifNull": ["$feat.hasMarkDown1", 0]},
                "hasMarkDown2": {"$ifNull": ["$feat.hasMarkDown2", 0]},
                "hasMarkDown3": {"$ifNull": ["$feat.hasMarkDown3", 0]},
                "hasMarkDown4": {"$ifNull": ["$feat.hasMarkDown4", 0]},
                "hasMarkDown5": {"$ifNull": ["$feat.hasMarkDown5", 0]},
                "CPI": {"$ifNull": ["$feat.CPI", None]},
                "Unemployment": {"$ifNull": ["$feat.Unemployment", None]},
                # Store info
                "Type": "$st.Type",
                "Size": "$st.Size",
                # Temporal features
                "Year": {"$year": "$Date"},
                "Month": {"$month": "$Date"},
                "Week": {"$isoWeek": "$Date"},
            }
        },

        # --- Write to merged_data collection ---
        {"$merge": {
            "into": "merged_data",
            "on": ["Store", "Dept", "Date"],
            "whenMatched": "replace",
            "whenNotMatched": "insert",
        }},
    ]

    db["train_sales"].aggregate(pipeline, allowDiskUse=True)
    count = db["merged_data"].count_documents({})
    logger.info(f"  merged_data: {count} documents created")
    return count


# ================================================================
# Summary & verification
# ================================================================

def print_summary(db):
    """Print summary statistics and sample queries."""
    print("\n" + "=" * 65)
    print("  MongoDB Data Load Summary")
    print("=" * 65)

    for coll_name in ["stores", "features", "train_sales", "test_sales", "merged_data"]:
        count = db[coll_name].count_documents({})
        print(f"  {coll_name:15s} : {count:>10,} documents")

    print("-" * 65)

    # Sample: average sales by store type
    print("\n  [Sample Query] Average Weekly Sales by Store Type:")
    pipeline = [
        {"$group": {"_id": "$Type", "avg_sales": {"$avg": "$Weekly_Sales"}}},
        {"$sort": {"avg_sales": -1}},
    ]
    for doc in db["merged_data"].aggregate(pipeline):
        print(f"    Type {doc['_id']}: ${doc['avg_sales']:,.2f}")

    # Sample: holiday vs non-holiday
    print("\n  [Sample Query] Holiday vs Non-Holiday Average Sales:")
    pipeline = [
        {"$group": {"_id": "$IsHoliday", "avg_sales": {"$avg": "$Weekly_Sales"}}},
        {"$sort": {"_id": -1}},
    ]
    for doc in db["merged_data"].aggregate(pipeline):
        label = "Holiday" if doc["_id"] else "Non-Holiday"
        print(f"    {label}: ${doc['avg_sales']:,.2f}")

    # Sample record
    print("\n  [Sample] One merged_data document:")
    sample = db["merged_data"].find_one({}, {"_id": 0})
    if sample:
        for k, v in sample.items():
            print(f"    {k}: {v}")

    print("=" * 65)


# ================================================================
# Main pipeline
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Load Walmart sales data into MongoDB (schema + data + merge)"
    )
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017",
                        help="MongoDB connection URI (default: mongodb://localhost:27017)")
    parser.add_argument("--local-dir", default="./data",
                        help="Local directory containing CSV files (default: ./data)")
    parser.add_argument("--skip-schema", action="store_true",
                        help="Skip schema/index creation (if already initialized)")
    parser.add_argument("--preprocessed", action="store_true",
                        help="Load preprocessed data from local-dir (expects merged_train.csv, features_cleaned.csv, stores.csv)")
    args = parser.parse_args()

    logger.info(f"MongoDB URI   : {args.mongo_uri}")
    logger.info(f"Data directory: {args.local_dir}")
    logger.info(f"Preprocessed  : {args.preprocessed}")

    client = MongoClient(args.mongo_uri)
    db = client[DB_NAME]

    # Step 1: Initialize schema
    if not args.skip_schema:
        logger.info("Step 1/4: Initializing MongoDB schema and indexes...")
        create_collections(args.mongo_uri)
    else:
        logger.info("Step 1/4: Skipped (--skip-schema)")

    if args.preprocessed:
        # Load preprocessed data directly
        logger.info(
            "Step 2/4: Loading preprocessed CSV data into collections...")
        load_preprocessed_stores(db, args.local_dir)
        load_preprocessed_features(db, args.local_dir)
        load_preprocessed_merged_data(db, args.local_dir)
        # Note: test_sales is not available in preprocessed data
        logger.info(
            "  Note: test_sales not loaded (not available in preprocessed data)")
    else:
        # Step 2: Load raw data into collections
        logger.info("Step 2/4: Loading raw CSV data into collections...")
        load_stores(db, args.local_dir)
        load_features(db, args.local_dir)
        load_sales(db, args.local_dir, "train.csv", "train_sales")
        load_sales(db, args.local_dir, "test.csv", "test_sales")

        # Step 3: Create merged dataset
        logger.info(
            "Step 3/4: Creating merged dataset (train + features + stores)...")
        create_merged_data(db)

    # Step 4: Print summary
    logger.info("Step 4/4: Verification...")
    print_summary(db)

    client.close()
    logger.info("Data load pipeline complete!")


if __name__ == "__main__":
    main()
