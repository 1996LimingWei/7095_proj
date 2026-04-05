"""
MongoDB Schema Definition for Walmart Sales Forecast Project
=============================================================
Defines collections, JSON Schema validation rules, and indexes optimized
for the project's analytical queries:
  - Average sales by store type (A/B/C)
  - Sales during holiday vs non-holiday periods
  - Impact of markdown promotions on sales
  - Time-series trend analysis

Database: walmart_sales
Collections:
    - stores       : Store metadata (type, size)
    - features     : Weekly store-level economic/promotional features
    - train_sales  : Historical weekly sales per store-department
    - test_sales   : Prediction targets (no Weekly_Sales)
    - merged_data  : Preprocessed merged dataset (train + features + stores)
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import CollectionInvalid


DB_NAME = "walmart_sales"

# ================================================================
# JSON Schema Validators
# ================================================================

STORES_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["Store", "Type", "Size"],
        "properties": {
            "Store": {"bsonType": "int", "description": "Store number (1-45)"},
            "Type": {"enum": ["A", "B", "C"], "description": "Store type classification"},
            "Size": {"bsonType": "int", "description": "Store floor area in sq ft"},
        },
    }
}

FEATURES_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["Store", "Date", "IsHoliday"],
        "properties": {
            "Store": {"bsonType": "int"},
            "Date": {"bsonType": "date"},
            "Temperature": {"bsonType": ["double", "null"]},
            "Fuel_Price": {"bsonType": ["double", "null"]},
            "MarkDown1": {"bsonType": ["double", "null"]},
            "MarkDown2": {"bsonType": ["double", "null"]},
            "MarkDown3": {"bsonType": ["double", "null"]},
            "MarkDown4": {"bsonType": ["double", "null"]},
            "MarkDown5": {"bsonType": ["double", "null"]},
            "hasMarkDown1": {"bsonType": "int", "description": "1 if MarkDown1 promotion active"},
            "hasMarkDown2": {"bsonType": "int"},
            "hasMarkDown3": {"bsonType": "int"},
            "hasMarkDown4": {"bsonType": "int"},
            "hasMarkDown5": {"bsonType": "int"},
            "CPI": {"bsonType": ["double", "null"]},
            "Unemployment": {"bsonType": ["double", "null"]},
            "IsHoliday": {"bsonType": "bool"},
        },
    }
}

TRAIN_SALES_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["Store", "Dept", "Date", "Weekly_Sales", "IsHoliday"],
        "properties": {
            "Store": {"bsonType": "int"},
            "Dept": {"bsonType": "int"},
            "Date": {"bsonType": "date"},
            "Weekly_Sales": {"bsonType": "double"},
            "IsHoliday": {"bsonType": "bool"},
        },
    }
}

TEST_SALES_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["Store", "Dept", "Date", "IsHoliday"],
        "properties": {
            "Store": {"bsonType": "int"},
            "Dept": {"bsonType": "int"},
            "Date": {"bsonType": "date"},
            "IsHoliday": {"bsonType": "bool"},
        },
    }
}

MERGED_DATA_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["Store", "Dept", "Date", "Weekly_Sales"],
        "properties": {
            "Store": {"bsonType": "int"},
            "Dept": {"bsonType": "int"},
            "Date": {"bsonType": "date"},
            "Weekly_Sales": {"bsonType": "double"},
            "IsHoliday": {"bsonType": "bool"},
            # From features
            "Temperature": {"bsonType": ["double", "null"]},
            "Fuel_Price": {"bsonType": ["double", "null"]},
            "MarkDown1": {"bsonType": ["double", "null"]},
            "MarkDown2": {"bsonType": ["double", "null"]},
            "MarkDown3": {"bsonType": ["double", "null"]},
            "MarkDown4": {"bsonType": ["double", "null"]},
            "MarkDown5": {"bsonType": ["double", "null"]},
            "hasMarkDown1": {"bsonType": "int"},
            "hasMarkDown2": {"bsonType": "int"},
            "hasMarkDown3": {"bsonType": "int"},
            "hasMarkDown4": {"bsonType": "int"},
            "hasMarkDown5": {"bsonType": "int"},
            "CPI": {"bsonType": ["double", "null"]},
            "Unemployment": {"bsonType": ["double", "null"]},
            # From stores
            "Type": {"enum": ["A", "B", "C"]},
            "Size": {"bsonType": "int"},
            # Derived temporal features
            "Year": {"bsonType": "int"},
            "Month": {"bsonType": "int"},
            "Week": {"bsonType": "int"},
        },
    }
}


# ================================================================
# Collection Configs (schema + indexes)
#
# Index design rationale (aligned with Spark SQL analytical queries):
#   - avg sales by store type   -> (Type, Weekly_Sales)
#   - holiday period analysis   -> (IsHoliday, Weekly_Sales)
#   - markdown impact analysis  -> (hasMarkDown*, Weekly_Sales)
#   - time-series trends        -> (Date), (Year, Month)
#   - per-store/dept lookups    -> (Store, Dept, Date)
# ================================================================

COLLECTIONS = {
    "stores": {
        "validator": STORES_VALIDATOR,
        "indexes": [
            {"keys": [("Store", ASCENDING)], "unique": True},
            {"keys": [("Type", ASCENDING)]},
        ],
    },
    "features": {
        "validator": FEATURES_VALIDATOR,
        "indexes": [
            {"keys": [("Store", ASCENDING), ("Date", ASCENDING)],
             "unique": True},
            {"keys": [("Date", ASCENDING)]},
            {"keys": [("IsHoliday", ASCENDING)]},
            # For markdown presence queries
            {"keys": [("hasMarkDown1", ASCENDING)]},
        ],
    },
    "train_sales": {
        "validator": TRAIN_SALES_VALIDATOR,
        "indexes": [
            {"keys": [("Store", ASCENDING), ("Dept", ASCENDING),
                      ("Date", ASCENDING)], "unique": True},
            {"keys": [("Date", ASCENDING)]},
            {"keys": [("Store", ASCENDING)]},
            {"keys": [("Dept", ASCENDING)]},
            {"keys": [("Weekly_Sales", DESCENDING)]},
            # For holiday analysis query
            {"keys": [("IsHoliday", ASCENDING), ("Weekly_Sales", DESCENDING)]},
        ],
    },
    "test_sales": {
        "validator": TEST_SALES_VALIDATOR,
        "indexes": [
            {"keys": [("Store", ASCENDING), ("Dept", ASCENDING),
                      ("Date", ASCENDING)], "unique": True},
            {"keys": [("Date", ASCENDING)]},
        ],
    },
    "merged_data": {
        "validator": MERGED_DATA_VALIDATOR,
        "indexes": [
            # Primary key
            {"keys": [("Store", ASCENDING), ("Dept", ASCENDING),
                      ("Date", ASCENDING)], "unique": True},
            # Query: avg sales by store type
            {"keys": [("Type", ASCENDING), ("Weekly_Sales", DESCENDING)]},
            # Query: holiday vs non-holiday sales
            {"keys": [("IsHoliday", ASCENDING), ("Weekly_Sales", DESCENDING)]},
            # Query: markdown impact on sales
            {"keys": [("hasMarkDown1", ASCENDING),
                      ("Weekly_Sales", DESCENDING)]},
            # Query: time-series trend analysis
            {"keys": [("Date", ASCENDING)]},
            {"keys": [("Year", ASCENDING), ("Month", ASCENDING)]},
            # Query: per-store drill-down
            {"keys": [("Store", ASCENDING), ("Type", ASCENDING)]},
            # Query: top sales ranking
            {"keys": [("Weekly_Sales", DESCENDING)]},
        ],
    },
}


def create_collections(mongo_uri: str = "mongodb://localhost:27017"):
    """Create all collections with schema validation and indexes."""
    client = MongoClient(mongo_uri)
    db = client[DB_NAME]

    for coll_name, config in COLLECTIONS.items():
        # Create collection with validator
        try:
            db.create_collection(coll_name, validator=config["validator"])
            print(f"[+] Created collection: {coll_name}")
        except CollectionInvalid:
            db.command("collMod", coll_name, validator=config["validator"])
            print(
                f"[~] Updated validator for existing collection: {coll_name}")

        # Create indexes
        collection = db[coll_name]
        for idx in config["indexes"]:
            idx_name = collection.create_index(
                idx["keys"],
                unique=idx.get("unique", False),
            )
            print(f"    Index: {idx_name}")

    print(f"\nAll collections initialized in database '{DB_NAME}'")
    client.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Initialize MongoDB schema for Walmart Sales project")
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017",
                        help="MongoDB connection URI")
    args = parser.parse_args()

    create_collections(args.mongo_uri)
