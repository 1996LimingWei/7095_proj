"""
HDFS Upload Script for Walmart Sales Forecast Project
=====================================================
Uploads raw or preprocessed CSV data to HDFS and converts cleaned data to Parquet format
for downstream Spark processing.

HDFS Directory Hierarchy:
    /user/walmart_sales/
        raw/            <- Original CSV files
        cleaned/        <- Preprocessed Parquet files (missing values handled, dates parsed)
        enriched/       <- Merged & feature-engineered Parquet (produced by Spark later)
        preprocessed/   <- Preprocessed CSV files (from data/preprocessed/)

Usage:
    # Upload raw CSVs only
    python hdfs_upload.py --local-dir ./data

    # Upload raw CSVs + generate cleaned Parquet
    python hdfs_upload.py --local-dir ./data --with-cleaning

    # Upload preprocessed CSVs directly
    python hdfs_upload.py --preprocessed-dir ./data/preprocessed

    # Use WebHDFS instead of CLI
    python hdfs_upload.py --local-dir ./data --method webhdfs --hdfs-host namenode

Prerequisites:
    pip install pyspark pandas   (for --with-cleaning / pyspark method)
    pip install hdfs             (for --method webhdfs only)
"""

import os
import sys
import argparse
import subprocess
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Data files to upload (raw)
DATA_FILES = ["train.csv", "features.csv", "stores.csv", "test.csv"]

# Preprocessed data files to upload
PREPROCESSED_FILES = ["merged_train.csv", "features_cleaned.csv", "stores.csv"]

# Default HDFS base directory
DEFAULT_HDFS_BASE = "/user/walmart_sales"

# HDFS sub-directories
HDFS_RAW_DIR = "raw"
HDFS_CLEANED_DIR = "cleaned"
HDFS_ENRICHED_DIR = "enriched"
HDFS_PREPROCESSED_DIR = "preprocessed"


# ============================================================
# Method 1: Hadoop CLI (recommended for cluster environments)
# ============================================================

def upload_raw_via_cli(local_dir: str, hdfs_base: str):
    """Upload raw CSV files to HDFS using hadoop fs commands."""
    logger.info("=== Uploading raw CSVs via Hadoop CLI ===")

    raw_dir = f"{hdfs_base}/{HDFS_RAW_DIR}"

    # Create full directory hierarchy
    for subdir in [HDFS_RAW_DIR, HDFS_CLEANED_DIR, HDFS_ENRICHED_DIR]:
        path = f"{hdfs_base}/{subdir}"
        subprocess.run(["hadoop", "fs", "-mkdir", "-p", path], check=True)
    logger.info(f"Created HDFS directory hierarchy under {hdfs_base}/")

    for filename in DATA_FILES:
        local_path = os.path.join(local_dir, filename)
        hdfs_path = f"{raw_dir}/{filename}"

        if not os.path.exists(local_path):
            logger.warning(f"File not found: {local_path}, skipping")
            continue

        # Remove existing file to allow re-upload
        subprocess.run(["hadoop", "fs", "-rm", "-f",
                       hdfs_path], capture_output=True)

        logger.info(f"Uploading {local_path} -> {hdfs_path}")
        subprocess.run(
            ["hadoop", "fs", "-put", local_path, hdfs_path], check=True)
        logger.info(f"  Done: {filename}")

    # Verify
    logger.info("Verifying uploaded files:")
    result = subprocess.run(["hadoop", "fs", "-ls", "-h", raw_dir],
                            capture_output=True, text=True)
    print(result.stdout)


# ============================================================
# Method 2: WebHDFS REST API
# ============================================================

def upload_raw_via_webhdfs(local_dir: str, hdfs_base: str, hdfs_host: str, hdfs_port: int):
    """Upload raw CSV files via WebHDFS REST API."""
    try:
        from hdfs import InsecureClient
    except ImportError:
        logger.error("hdfs package not installed. Run: pip install hdfs")
        sys.exit(1)

    logger.info("=== Uploading raw CSVs via WebHDFS ===")
    client = InsecureClient(f"http://{hdfs_host}:{hdfs_port}", user='root')

    raw_dir = f"{hdfs_base}/{HDFS_RAW_DIR}"
    for subdir in [HDFS_RAW_DIR, HDFS_CLEANED_DIR, HDFS_ENRICHED_DIR]:
        client.makedirs(f"{hdfs_base}/{subdir}")
    logger.info(f"Created HDFS directory hierarchy under {hdfs_base}/")

    for filename in DATA_FILES:
        local_path = os.path.join(local_dir, filename)
        hdfs_path = f"{raw_dir}/{filename}"

        if not os.path.exists(local_path):
            logger.warning(f"File not found: {local_path}, skipping")
            continue

        logger.info(f"Uploading {local_path} -> {hdfs_path}")
        with open(local_path, "rb") as f:
            client.write(hdfs_path, f, overwrite=True)
        logger.info(f"  Done: {filename}")

    # Verify
    logger.info("Verifying uploaded files:")
    for entry in client.list(raw_dir):
        status = client.status(f"{raw_dir}/{entry}")
        size_mb = status["length"] / (1024 * 1024)
        print(f"  {entry}: {size_mb:.2f} MB")


# ============================================================
# Cleaning: preprocess CSVs and write Parquet to HDFS
# ============================================================

def clean_and_write_parquet(local_dir: str, hdfs_base: str):
    """
    Preprocess raw CSVs and write cleaned Parquet files to HDFS.
    Handles:
      - Date parsing to proper date type
      - Missing MarkDown values filled with 0.0
      - hasMarkDown1-5 binary indicator columns
      - Data type enforcement
    Produces Parquet files ready for Spark SQL / MLlib consumption.
    """
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
        from pyspark.sql.types import (
            StructType, StructField, IntegerType, DoubleType,
            StringType, BooleanType,
        )
    except ImportError:
        logger.error("pyspark not installed. Run: pip install pyspark")
        sys.exit(1)

    logger.info("=== Cleaning data and writing Parquet to HDFS ===")

    spark = SparkSession.builder \
        .appName("WalmartSales_HDFS_Clean") \
        .getOrCreate()

    cleaned_dir = f"{hdfs_base}/{HDFS_CLEANED_DIR}"

    # --- stores.csv ---
    stores_path = os.path.join(local_dir, "stores.csv")
    if os.path.exists(stores_path):
        df_stores = spark.read.csv(stores_path, header=True, inferSchema=True)
        df_stores = df_stores.select(
            F.col("Store").cast(IntegerType()),
            F.col("Type").cast(StringType()),
            F.col("Size").cast(IntegerType()),
        )
        df_stores.write.mode("overwrite").parquet(f"{cleaned_dir}/stores")
        logger.info(
            f"stores: {df_stores.count()} rows -> {cleaned_dir}/stores")

    # --- features.csv ---
    features_path = os.path.join(local_dir, "features.csv")
    if os.path.exists(features_path):
        df_features = spark.read.csv(
            features_path, header=True, inferSchema=True)

        # Parse date (handles DD/MM/YYYY format from Kaggle dataset)
        df_features = df_features.withColumn(
            "Date", F.to_date(F.col("Date"), "dd/MM/yyyy")
        )
        # Fallback: try yyyy-MM-dd if first format returns null
        df_features = df_features.withColumn(
            "Date",
            F.when(F.col("Date").isNull(), F.to_date(
                F.col("Date"), "yyyy-MM-dd"))
             .otherwise(F.col("Date"))
        )

        # MarkDown indicators: 1 if promotion value exists, 0 otherwise
        markdown_cutoff = "2011-11-01"
        for i in range(1, 6):
            col_name = f"MarkDown{i}"
            flag_name = f"hasMarkDown{i}"
            df_features = df_features.withColumn(
                flag_name,
                F.when(
                    (F.col(col_name).isNotNull()) & (
                        F.col("Date") >= F.lit(markdown_cutoff)),
                    F.lit(1)
                ).otherwise(F.lit(0))
            )
            # Fill missing MarkDown values with 0
            df_features = df_features.withColumn(
                col_name,
                F.when(F.col(col_name).isNull(), F.lit(
                    0.0)).otherwise(F.col(col_name))
            )

        df_features.write.mode("overwrite").parquet(f"{cleaned_dir}/features")
        logger.info(
            f"features: {df_features.count()} rows -> {cleaned_dir}/features")

    # --- train.csv ---
    train_path = os.path.join(local_dir, "train.csv")
    if os.path.exists(train_path):
        df_train = spark.read.csv(train_path, header=True, inferSchema=True)
        df_train = df_train.withColumn(
            "Date", F.to_date(F.col("Date"), "dd/MM/yyyy"))
        df_train = df_train.withColumn(
            "Date",
            F.when(F.col("Date").isNull(), F.to_date(
                F.col("Date"), "yyyy-MM-dd"))
             .otherwise(F.col("Date"))
        )
        df_train.write.mode("overwrite").parquet(f"{cleaned_dir}/train")
        logger.info(f"train: {df_train.count()} rows -> {cleaned_dir}/train")

    # --- test.csv ---
    test_path = os.path.join(local_dir, "test.csv")
    if os.path.exists(test_path):
        df_test = spark.read.csv(test_path, header=True, inferSchema=True)
        df_test = df_test.withColumn(
            "Date", F.to_date(F.col("Date"), "dd/MM/yyyy"))
        df_test = df_test.withColumn(
            "Date",
            F.when(F.col("Date").isNull(), F.to_date(
                F.col("Date"), "yyyy-MM-dd"))
             .otherwise(F.col("Date"))
        )
        df_test.write.mode("overwrite").parquet(f"{cleaned_dir}/test")
        logger.info(f"test: {df_test.count()} rows -> {cleaned_dir}/test")

    spark.stop()
    logger.info("Spark session stopped. Parquet cleaning complete.")


# ============================================================
# Upload Preprocessed Data
# ============================================================

def upload_preprocessed_via_cli(preprocessed_dir: str, hdfs_base: str):
    """Upload preprocessed CSV files to HDFS using hadoop fs commands."""
    logger.info("=== Uploading preprocessed CSVs via Hadoop CLI ===")

    preprocessed_hdfs_dir = f"{hdfs_base}/{HDFS_PREPROCESSED_DIR}"

    # Create directory
    subprocess.run(["hadoop", "fs", "-mkdir", "-p",
                   preprocessed_hdfs_dir], check=True)
    logger.info(f"Created HDFS directory: {preprocessed_hdfs_dir}")

    for filename in PREPROCESSED_FILES:
        local_path = os.path.join(preprocessed_dir, filename)
        hdfs_path = f"{preprocessed_hdfs_dir}/{filename}"

        if not os.path.exists(local_path):
            logger.warning(f"File not found: {local_path}, skipping")
            continue

        # Remove existing file to allow re-upload
        subprocess.run(["hadoop", "fs", "-rm", "-f",
                       hdfs_path], capture_output=True)

        logger.info(f"Uploading {local_path} -> {hdfs_path}")
        subprocess.run(
            ["hadoop", "fs", "-put", local_path, hdfs_path], check=True)
        logger.info(f"  Done: {filename}")

    # Verify
    logger.info("Verifying uploaded files:")
    result = subprocess.run(["hadoop", "fs", "-ls", "-h", preprocessed_hdfs_dir],
                            capture_output=True, text=True)
    print(result.stdout)


def upload_preprocessed_via_webhdfs(preprocessed_dir: str, hdfs_base: str, hdfs_host: str, hdfs_port: int):
    """Upload preprocessed CSV files via WebHDFS REST API."""
    try:
        from hdfs import InsecureClient
    except ImportError:
        logger.error("hdfs package not installed. Run: pip install hdfs")
        sys.exit(1)

    logger.info("=== Uploading preprocessed CSVs via WebHDFS ===")
    client = InsecureClient(f"http://{hdfs_host}:{hdfs_port}", user='root')

    preprocessed_hdfs_dir = f"{hdfs_base}/{HDFS_PREPROCESSED_DIR}"
    client.makedirs(preprocessed_hdfs_dir)
    logger.info(f"Created HDFS directory: {preprocessed_hdfs_dir}")

    for filename in PREPROCESSED_FILES:
        local_path = os.path.join(preprocessed_dir, filename)
        hdfs_path = f"{preprocessed_hdfs_dir}/{filename}"

        if not os.path.exists(local_path):
            logger.warning(f"File not found: {local_path}, skipping")
            continue

        logger.info(f"Uploading {local_path} -> {hdfs_path}")
        with open(local_path, "rb") as f:
            client.write(hdfs_path, f, overwrite=True)
        logger.info(f"  Done: {filename}")

    # Verify
    logger.info("Verifying uploaded files:")
    for entry in client.list(preprocessed_hdfs_dir):
        status = client.status(f"{preprocessed_hdfs_dir}/{entry}")
        size_mb = status["length"] / (1024 * 1024)
        print(f"  {entry}: {size_mb:.2f} MB")


# ============================================================
# Verification utility
# ============================================================

def verify_hdfs_structure(hdfs_base: str):
    """Print the full HDFS directory tree for verification."""
    logger.info(f"HDFS directory structure under {hdfs_base}/:")
    result = subprocess.run(
        ["hadoop", "fs", "-ls", "-R", "-h", hdfs_base],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(result.stdout)
    else:
        logger.warning(
            "Could not list HDFS directory (hadoop CLI may not be available)")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Upload Walmart sales data to HDFS (raw CSV + optional cleaned Parquet + preprocessed)"
    )
    parser.add_argument("--local-dir", default="./data",
                        help="Local directory containing raw CSV files (default: ./data)")
    parser.add_argument("--hdfs-base", default=DEFAULT_HDFS_BASE,
                        help=f"HDFS base directory (default: {DEFAULT_HDFS_BASE})")
    parser.add_argument("--method", choices=["cli", "webhdfs"], default="cli",
                        help="Upload method (default: cli)")
    parser.add_argument("--hdfs-host", default="localhost",
                        help="HDFS NameNode host for WebHDFS (default: localhost)")
    parser.add_argument("--hdfs-port", type=int, default=9870,
                        help="WebHDFS port (default: 9870)")
    parser.add_argument("--with-cleaning", action="store_true",
                        help="Also preprocess CSVs and write cleaned Parquet to HDFS")
    parser.add_argument("--preprocessed-dir",
                        help="Upload preprocessed CSV files from this directory (skips raw upload)")
    parser.add_argument('--skip-upload', action='store_true',
                        help='Skip raw CSV upload phase (use if already uploaded)')
    args = parser.parse_args()

    logger.info(f"Local data directory : {args.local_dir}")
    logger.info(f"HDFS base directory  : {args.hdfs_base}")
    logger.info(f"Upload method        : {args.method}")
    logger.info(f"With cleaning        : {args.with_cleaning}")
    logger.info(f"Preprocessed dir     : {args.preprocessed_dir}")

    # Upload preprocessed data (if specified)
    if args.preprocessed_dir:
        if args.method == "cli":
            upload_preprocessed_via_cli(args.preprocessed_dir, args.hdfs_base)
        elif args.method == "webhdfs":
            upload_preprocessed_via_webhdfs(
                args.preprocessed_dir, args.hdfs_base, args.hdfs_host, args.hdfs_port)
    else:
        # Step 1: Upload raw CSVs
        if not args.skip_upload:
            if args.method == "cli":
                upload_raw_via_cli(args.local_dir, args.hdfs_base)
            elif args.method == "webhdfs":
                upload_raw_via_webhdfs(
                    args.local_dir, args.hdfs_base, args.hdfs_host, args.hdfs_port)
        else:
            logger.info("Skipping raw CSV upload phase as requested.")

        # Step 2 (optional): Clean and write Parquet
        if args.with_cleaning:
            clean_and_write_parquet(args.local_dir, args.hdfs_base)

    # Step 3: Verify
    if args.method == "cli":
        verify_hdfs_structure(args.hdfs_base)

    logger.info("HDFS upload pipeline complete!")


if __name__ == "__main__":
    main()
