# Walmart Sales Forecasting - Quick Start Guide

A big data pipeline for retail sales forecasting using the Walmart Recruiting dataset with baseline machine learning models.

## Overview

This project implements a streamlined ML pipeline with data-driven feature selection. Features are automatically selected based on correlation analysis (removing low-correlation and redundant features), resulting in an optimal feature set (typically 10-15 features) while maintaining excellent performance with RandomForest (R² = 0.98).

## Prerequisites

- Python 3.8+
- Jupyter Notebook
- Docker and Docker Compose

## Quick Start

### Step 1: Data Preprocessing

Run the Jupyter notebook to generate preprocessed data:

```bash
jupyter notebook Data_preprocess_visualization.ipynb
```

This creates preprocessed files in `data/preprocessed/`:
- `merged_train.csv`
- `features_cleaned.csv`
- `stores.csv`

### Step 2: Start Services and Run ML Pipeline

```bash
# 1. Start all services (MongoDB, HDFS, and automatic data upload)
docker-compose up -d

# 2. Wait for data upload to complete (check logs)
docker logs -f walmart_data_upload

# 3. Once upload is complete, enter the tester container
docker exec -it walmart_tester /bin/bash

# 4. Run the ML pipeline (environment variables are already set!)
cd /app/machine_learning
python sales_forecasting.py

# 5. Exit the container when done
exit

# 6. Stop all services
docker-compose down
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| MongoDB | 27017 | Document database for analytics |
| HDFS NameNode | 9870 | HDFS web UI |
| HDFS DataNode | 9864 | HDFS data node |

## Output

Results are saved to `output/`:
- `feature_correlation_heatmap.png` - Computed correlation analysis for feature selection
- `model_comparison.csv` - Model performance metrics (MAE, RMSE, R², Accuracy, Precision, Recall, F1)
- `feature_importance_*.png` - Feature importance charts for tree-based models
- `predictions_vs_actual_XGBoost.png` - Prediction quality plot for best model
- `predictions.csv` - Test set predictions

## Model Performance (Data-Driven Feature Selection)

| Model | Test R² | Test MAE | Test Accuracy | Test F1 |
|-------|---------|----------|---------------|---------|
| Linear Regression | 0.0899 | 1524.37 | 0.68 | 0.68 |
| Ridge | 0.0899 | 1524.34 | 0.68 | 0.68 |
| **Random Forest** | **0.9780** | **569.34** | **0.97** | **0.98** |
| Gradient Boosting | 0.9092 | 1077.18 | 0.94 | 0.94 |
| XGBoost | 0.9678 | 739.07 | 0.96 | 0.96 |

## Features Used (Data-Driven Selection)

Features are **automatically selected** by the pipeline based on computed correlation analysis:

| Selection Criteria | Threshold | Description |
|-------------------|-----------|-------------|
| **Target Correlation** | \|corr\| >= 0.01 | Features must have meaningful correlation with Weekly_Sales |
| **Redundancy** | Inter-corr <= 0.95 | Remove highly correlated features (keep one with higher target correlation) |

## Key Insight

Random Forest achieves the best performance (R² = 0.978) with **data-driven feature selection**. The pipeline automatically:
1. Calculates correlations between all features and Weekly_Sales
2. Removes features with \|correlation\| < 0.01 (statistically insignificant)
3. Removes redundant features (inter-correlation > 0.95)
4. Selects optimal feature set (typically 10-15 features) based on thresholds, not arbitrary numbers

**Why This Approach:**
- **Transparent**: Correlation heatmap shows exactly why each feature was kept/removed
- **Data-driven**: Feature count determined by actual correlation values, not guesswork
- **Configurable**: Thresholds can be adjusted (e.g., stricter redundancy threshold = fewer features)
- **Reproducible**: Same data always produces same feature selection

This demonstrates that principled, threshold-based feature selection improves model interpretability without sacrificing accuracy.