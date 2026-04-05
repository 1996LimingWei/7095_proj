# Walmart Sales Forecasting - Quick Start Guide

A big data pipeline for retail sales forecasting using the Walmart Recruiting dataset.

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
- `model_comparison.csv` - Model performance metrics
- `feature_importance_*.png` - Feature importance charts
- `predictions_vs_actual_*.png` - Prediction quality plots
- `predictions.csv` - Test set predictions

## Project Structure

```
.
├── Data_preprocess_visualization.ipynb   # Data preprocessing
├── machine_learning/
│   └── sales_forecasting.py              # ML pipeline
├── hdfs_upload/                          # HDFS upload scripts
├── mongodb_load/                         # MongoDB load scripts
├── docker-compose.yml                    # Docker services
└── data/
    └── preprocessed/                     # Preprocessed data output
```

## Team

7095 Team: Wei Liming, Ke Linyao, Huo Weijia, Xu Hao, Zhang Hongyang
