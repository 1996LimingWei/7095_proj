# Walmart Sales Forecasting - Quick Start Guide

A big data pipeline for retail sales forecasting using the Walmart Recruiting dataset with baseline machine learning models.

## Overview

This project implements a straightforward ML pipeline using 16 baseline features (no lag or rolling features) achieving excellent performance with XGBoost and RandomForest models (R² ~0.97).

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
- `model_comparison.csv` - Model performance metrics (MAE, RMSE, R², Accuracy, Precision, Recall, F1)
- `feature_importance_*.png` - Feature importance charts for tree-based models
- `predictions_vs_actual_XGBoost.png` - Prediction quality plot for best model
- `predictions.csv` - Test set predictions

## Model Performance (Baseline 16 Features)

| Model | Test R² | Test MAE | Test Accuracy | Test F1 |
|-------|---------|----------|---------------|---------|
| Linear Regression | 0.0899 | 1524.37 | 0.68 | 0.68 |
| Ridge | 0.0899 | 1524.34 | 0.68 | 0.68 |
| Random Forest | **0.9780** | **569.34** | **0.97** | **0.98** |
| Gradient Boosting | 0.9092 | 1077.18 | 0.94 | 0.94 |
| **XGBoost** | **0.9678** | **739.07** | **0.96** | **0.96** |

## Features Used (Baseline Only - 16 Features)

- **Store Characteristics**: Store, Dept, Size, Type (encoded)
- **Temporal**: Year, Month, Week, Day, DayOfYear, Quarter, Season
- **Economic Indicators**: Temperature, Fuel_Price, CPI, Unemployment
- **Markdown/Promotions**: MarkDown1-5 and indicators
- **Holiday**: IsHoliday

## Key Insight

Random Forest and XGBoost achieve excellent performance (R² > 0.96) using only baseline features, demonstrating that complex temporal feature engineering (lag/rolling features) is not necessary for this dataset. The simple baseline approach is sufficient for a school project while maintaining high accuracy.

## Project Structure

```
.
├── Data_preprocess_visualization.ipynb   # Data preprocessing
├── machine_learning/
│   └── sales_forecasting.py              # ML pipeline (baseline features)
├── hdfs_upload/                          # HDFS upload scripts
├── mongodb_load/                         # MongoDB load scripts
├── docker-compose.yml                    # Docker services
└── data/
    └── preprocessed/                     # Preprocessed data output
```

## Team

7095 Team: Wei Liming, Ke Linyao, Huo Weijia, Xu Hao, Zhang Hongyang
