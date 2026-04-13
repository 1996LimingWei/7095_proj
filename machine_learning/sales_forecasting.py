"""
Walmart Sales Forecasting - Machine Learning Pipeline

This module implements a complete ML pipeline for retail sales forecasting using
the Walmart Recruiting dataset. It includes basic feature engineering, model training,
evaluation, and prediction generation.

Uses features dynamically selected through computed correlation analysis:
- Pipeline calculates correlations between all features and Weekly_Sales
- Removes features with |correlation| < threshold (default: 0.01)
- Removes redundant features (inter-correlation > threshold, default: 0.95)
- Result: Data-driven feature count (typically 10-15 features) for optimal model performance

Supports loading data from:
- HDFS (Hadoop Distributed File System)
- MongoDB (document database)

Environment Variables (can be set in .env file):
    DATA_SOURCE: 'hdfs' or 'mongodb' (default: 'mongodb')
    MONGO_URI: MongoDB connection URI
    DB_NAME: MongoDB database name
    HDFS_HOST: HDFS namenode host
    HDFS_PORT: HDFS WebHDFS port
    HDFS_BASE: Base HDFS directory

Author: 7095 Team
"""

import os
import sys
import warnings
from datetime import datetime
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score)
import xgboost as xgb

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from parent directory
    env_path = os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"[INFO] Loaded environment from {env_path}")
except ImportError:
    pass  # python-dotenv not installed, use system environment variables

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42


# ============================================================
# Data Loading from HDFS or MongoDB
# ============================================================

def load_data_from_hdfs(hdfs_base: str = '/user/walmart_sales') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load preprocessed data from HDFS.
    Requires hdfs library: pip install hdfs

    Args:
        hdfs_base: Base HDFS directory

    Returns:
        Tuple of (merged_train, features_cleaned, stores, test)
    """
    try:
        from hdfs import InsecureClient
    except ImportError:
        raise ImportError("hdfs library not installed. Run: pip install hdfs")

    # Connect to HDFS (default WebHDFS port 9870)
    hdfs_host = os.environ.get('HDFS_HOST', 'localhost')
    hdfs_port = int(os.environ.get('HDFS_PORT', '9870'))
    client = InsecureClient(f"http://{hdfs_host}:{hdfs_port}", user='root')

    preprocessed_dir = f"{hdfs_base}/preprocessed"

    # Read files from HDFS
    with client.read(f"{preprocessed_dir}/merged_train.csv") as reader:
        merged_train = pd.read_csv(reader)
    with client.read(f"{preprocessed_dir}/features_cleaned.csv") as reader:
        features_cleaned = pd.read_csv(reader)
    with client.read(f"{preprocessed_dir}/stores.csv") as reader:
        stores = pd.read_csv(reader)

    # Test data is not in preprocessed, load from local or raw
    data_dir = '../data'
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    print(f"[HDFS] Data loaded from HDFS: {hdfs_base}/preprocessed")
    return merged_train, features_cleaned, stores, test


def load_data_from_mongodb(mongo_uri: str = 'mongodb://localhost:27017',
                           db_name: str = 'walmart_sales') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load preprocessed data from MongoDB.
    Requires pymongo library: pip install pymongo

    Args:
        mongo_uri: MongoDB connection URI
        db_name: Database name

    Returns:
        Tuple of (merged_train, features_cleaned, stores, test)
    """
    try:
        from pymongo import MongoClient
    except ImportError:
        raise ImportError(
            "pymongo library not installed. Run: pip install pymongo")

    client = MongoClient(mongo_uri)
    db = client[db_name]

    # Load merged_data collection
    merged_train = pd.DataFrame(list(db['merged_data'].find({}, {'_id': 0})))

    # Load features collection
    features_cleaned = pd.DataFrame(list(db['features'].find({}, {'_id': 0})))

    # Load stores collection
    stores = pd.DataFrame(list(db['stores'].find({}, {'_id': 0})))

    # Test data not in MongoDB preprocessed collections
    data_dir = '../data'
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    client.close()
    print(f"[MongoDB] Data loaded from MongoDB: {db_name}")
    return merged_train, features_cleaned, stores, test


def load_data(data_source: str = 'mongodb',
              hdfs_base: str = '/user/walmart_sales',
              mongo_uri: str = 'mongodb://localhost:27017',
              db_name: str = 'walmart_sales') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load preprocessed data from HDFS or MongoDB.

    Args:
        data_source: 'hdfs' or 'mongodb' (default: 'mongodb')
        hdfs_base: Base HDFS directory (for hdfs source)
        mongo_uri: MongoDB connection URI (for mongodb source)
        db_name: MongoDB database name (for mongodb source)

    Returns:
        Tuple of (merged_train, features_cleaned, stores, test)

    Raises:
        ConnectionError: If unable to connect to the data source
        ValueError: If data_source is not 'hdfs' or 'mongodb'
    """
    print("=" * 60)
    print("DATA LOADING")
    print("=" * 60)
    print(f"Source: {data_source}")

    try:
        if data_source == 'hdfs':
            merged_train, features_cleaned, stores, test = load_data_from_hdfs(
                hdfs_base)
        elif data_source == 'mongodb':
            merged_train, features_cleaned, stores, test = load_data_from_mongodb(
                mongo_uri, db_name)
        else:
            raise ValueError(
                f"Unknown data_source: {data_source}. Use 'hdfs' or 'mongodb'")
    except Exception as e:
        print(f"ERROR: Failed to load data from {data_source}")
        print(f"Details: {str(e)}")
        raise ConnectionError(
            f"Could not load data from {data_source}. "
            f"Ensure the service is running and accessible.") from e

    print("=" * 60)
    print("DATA LOADING SUMMARY")
    print("=" * 60)
    print(f"Merged Train shape: {merged_train.shape}")
    print(f"Features Cleaned shape: {features_cleaned.shape}")
    print(f"Stores shape: {stores.shape}")
    print(f"Test shape: {test.shape}")
    print()

    return merged_train, features_cleaned, stores, test


def prepare_test_data(test: pd.DataFrame, features_cleaned: pd.DataFrame,
                      stores: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare test data by merging with features and stores, similar to training data.

    Args:
        test: Raw test dataframe
        features_cleaned: Cleaned features dataframe
        stores: Stores dataframe

    Returns:
        Prepared test dataframe
    """
    # Convert date columns to datetime
    test['Date'] = pd.to_datetime(test['Date'])
    features_cleaned['Date'] = pd.to_datetime(features_cleaned['Date'])

    # Merge test with features on Store and Date
    test_merged = pd.merge(test, features_cleaned, on=[
                           'Store', 'Date'], how='left')

    # Merge with stores on Store
    test_merged = pd.merge(test_merged, stores, on='Store', how='left')

    # IsHoliday column will be handled in feature_engineering
    # No rename needed, keep as IsHoliday

    print(f"Prepared Test shape: {test_merged.shape}")
    return test_merged


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from Date column.
    Matches notebook section 1.5.

    Args:
        df: DataFrame with 'Date' column

    Returns:
        DataFrame with additional temporal features
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    # Extract temporal features (matching notebook section 1.5)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)

    # Additional temporal features for better modeling
    df['Day'] = df['Date'].dt.day
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Quarter'] = df['Date'].dt.quarter

    # Season feature
    df['Season'] = df['Month'].map({12: 0, 1: 0, 2: 0,  # Winter
                                    3: 1, 4: 1, 5: 1,   # Spring
                                    6: 2, 7: 2, 8: 2,   # Summer
                                    9: 3, 10: 3, 11: 3})  # Fall

    return df


def create_lag_features(df: pd.DataFrame, group_cols: List[str],
                        target_col: str, lags: List[int]) -> pd.DataFrame:
    """
    Create lag features for time series data.

    Args:
        df: DataFrame sorted by date
        group_cols: Columns to group by (e.g., ['Store', 'Dept'])
        target_col: Target column to create lags for
        lags: List of lag periods

    Returns:
        DataFrame with lag features
    """
    df = df.copy()
    df = df.sort_values(by=group_cols + ['Date'])

    for lag in lags:
        lag_col = f'{target_col}_lag_{lag}'
        df[lag_col] = df.groupby(group_cols)[target_col].shift(lag)

    return df


def create_rolling_features(df: pd.DataFrame, group_cols: List[str],
                            target_col: str, windows: List[int]) -> pd.DataFrame:
    """
    Create rolling window statistics features.

    Args:
        df: DataFrame sorted by date
        group_cols: Columns to group by
        target_col: Target column
        windows: List of window sizes

    Returns:
        DataFrame with rolling features
    """
    df = df.copy()
    df = df.sort_values(by=group_cols + ['Date'])

    for window in windows:
        # Rolling mean
        df[f'{target_col}_rolling_mean_{window}'] = (
            df.groupby(group_cols)[target_col]
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        )

        # Rolling std
        df[f'{target_col}_rolling_std_{window}'] = (
            df.groupby(group_cols)[target_col]
            .transform(lambda x: x.rolling(window=window, min_periods=1).std())
        )

    return df


def analyze_feature_correlations(df: pd.DataFrame, output_dir: str = 'output',
                                 correlation_threshold: float = 0.95,
                                 min_target_correlation: float = 0.01) -> Tuple[pd.Series, List[str]]:
    """
    Analyze correlations and automatically select optimal features.

    Selection criteria:
    1. Remove features with |correlation with target| < min_target_correlation
    2. Remove redundant features (inter-correlation > correlation_threshold)
    3. Keep features with highest correlation to target when removing redundancies

    Args:
        df: DataFrame with features and Weekly_Sales target
        output_dir: Directory to save correlation heatmap
        correlation_threshold: Threshold for considering features redundant (default: 0.95)
        min_target_correlation: Minimum |correlation| with target to keep feature (default: 0.01)

    Returns:
        Tuple of (all_correlations, selected_feature_list)
    """
    print("\n" + "=" * 60)
    print("FEATURE CORRELATION ANALYSIS & SELECTION")
    print("=" * 60)
    print(f"Selection criteria:")
    print(f"  - Min |correlation with Weekly_Sales|: {min_target_correlation}")
    print(f"  - Max inter-feature correlation: {correlation_threshold}")

    # Get numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])

    if 'Weekly_Sales' not in numeric_df.columns:
        print("Warning: Weekly_Sales not found in numeric columns")
        return pd.Series(), []

    # Calculate correlation with Weekly_Sales
    target_correlations = numeric_df.corr()['Weekly_Sales'].drop(
        'Weekly_Sales', errors='ignore')
    target_correlations = target_correlations.sort_values(
        key=lambda x: abs(x), ascending=False)

    print("\nAll features by correlation with Weekly_Sales:")
    print("-" * 50)
    for feature, corr in target_correlations.items():
        marker = ""
        if abs(corr) < min_target_correlation:
            marker = " [LOW - will remove]"
        print(f"  {feature:20s}: {corr:7.4f}{marker}")

    # Step 1: Filter by target correlation threshold
    selected_features = target_correlations[abs(
        target_correlations) >= min_target_correlation].index.tolist()
    removed_low_corr = set(target_correlations.index) - set(selected_features)

    print(
        f"\nStep 1: Removed {len(removed_low_corr)} features with |correlation| < {min_target_correlation}")
    for feat in removed_low_corr:
        print(f"  - {feat}: {target_correlations[feat]:.4f}")

    # Step 2: Remove redundant features (high inter-correlation)
    if len(selected_features) > 1:
        feature_corr_matrix = numeric_df[selected_features].corr().abs()

        # Iterate through features and remove redundant ones
        features_to_remove = set()
        for i, feat_i in enumerate(selected_features):
            if feat_i in features_to_remove:
                continue
            for feat_j in selected_features[i+1:]:
                if feat_j in features_to_remove:
                    continue
                if feature_corr_matrix.loc[feat_i, feat_j] > correlation_threshold:
                    # Remove the one with lower target correlation
                    if abs(target_correlations[feat_i]) >= abs(target_correlations[feat_j]):
                        features_to_remove.add(feat_j)
                        print(
                            f"  - {feat_j} redundant with {feat_i} (corr={feature_corr_matrix.loc[feat_i, feat_j]:.3f})")
                    else:
                        features_to_remove.add(feat_i)
                        print(
                            f"  - {feat_i} redundant with {feat_j} (corr={feature_corr_matrix.loc[feat_i, feat_j]:.3f})")

        selected_features = [
            f for f in selected_features if f not in features_to_remove]
        print(
            f"\nStep 2: Removed {len(features_to_remove)} redundant features (inter-correlation > {correlation_threshold})")

    print(f"\nFinal selected features ({len(selected_features)} total):")
    for feat in selected_features:
        print(f"  - {feat}: {target_correlations[feat]:.4f}")

    # Generate correlation heatmap for selected features
    os.makedirs(output_dir, exist_ok=True)

    if len(selected_features) > 0:
        plot_features = selected_features + ['Weekly_Sales']
        plot_features = [f for f in plot_features if f in numeric_df.columns]
        corr_matrix = numeric_df[plot_features].corr()

        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', cmap='coolwarm',
                    center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title(f'Feature Correlation Heatmap ({len(selected_features)} Selected Features)',
                  fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(
            output_dir, 'feature_correlation_heatmap.png'), dpi=150)
        plt.close()

        print(
            f"\nCorrelation heatmap saved: {output_dir}/feature_correlation_heatmap.png")

    print("=" * 60)

    return target_correlations, selected_features


def feature_engineering(train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None,
                        output_dir: str = 'output',
                        correlation_threshold: float = 0.95,
                        min_target_correlation: float = 0.01) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], List[str]]:
    """
    Complete optimized feature engineering pipeline with data-driven feature selection.

    Features are automatically selected based on correlation analysis:
    - Remove features with |correlation with target| < min_target_correlation
    - Remove redundant features (inter-correlation > correlation_threshold)
    - Keep features with highest correlation to target when removing redundancies
    - Result: Dynamic feature count based on data, not hardcoded number

    Args:
        train_df: Training dataframe
        test_df: Optional test dataframe
        output_dir: Directory to save correlation analysis
        correlation_threshold: Threshold for considering features redundant
        min_target_correlation: Minimum |correlation| with target to keep feature

    Returns:
        Tuple of (train_features, test_features, feature_columns)
    """
    print("=" * 60)
    print("FEATURE ENGINEERING (DATA-DRIVEN SELECTION)")
    print("=" * 60)

    # Step 1: Extract temporal features
    print("Step 1: Extracting temporal features...")
    train_df = extract_temporal_features(train_df)
    if test_df is not None:
        test_df = extract_temporal_features(test_df)

    # Step 2: Handle categorical variables
    print("Step 2: Encoding categorical variables...")

    # Encode Store Type
    le_type = LabelEncoder()
    train_df['Type_encoded'] = le_type.fit_transform(train_df['Type'])
    if test_df is not None:
        # Handle unseen categories in test
        test_df['Type_encoded'] = test_df['Type'].map(
            {label: idx for idx, label in enumerate(le_type.classes_)}
        ).fillna(0).astype(int)

    # Convert boolean IsHoliday to int (keep only one, rename to IsHoliday)
    if 'IsHoliday_x' in train_df.columns:
        train_df['IsHoliday'] = train_df['IsHoliday_x'].astype(int)
        train_df = train_df.drop(
            columns=['IsHoliday_x', 'IsHoliday_y'], errors='ignore')
    elif 'IsHoliday_y' in train_df.columns:
        train_df['IsHoliday'] = train_df['IsHoliday_y'].astype(int)
        train_df = train_df.drop(columns=['IsHoliday_y'], errors='ignore')

    if test_df is not None:
        if 'IsHoliday_x' in test_df.columns:
            test_df['IsHoliday'] = test_df['IsHoliday_x'].astype(int)
            test_df = test_df.drop(
                columns=['IsHoliday_x', 'IsHoliday_y'], errors='ignore')
        elif 'IsHoliday_y' in test_df.columns:
            test_df['IsHoliday'] = test_df['IsHoliday_y'].astype(int)
            test_df = test_df.drop(columns=['IsHoliday_y'], errors='ignore')

    # Step 3: Analyze correlations and automatically select features
    print("\nStep 3: Analyzing correlations and selecting features...")
    correlations, feature_cols = analyze_feature_correlations(
        train_df, output_dir,
        correlation_threshold=correlation_threshold,
        min_target_correlation=min_target_correlation
    )

    # Feature selection is now data-driven based on:
    # 1. Correlation with Weekly_Sales (must be >= min_target_correlation)
    # 2. Inter-feature correlation (must be <= correlation_threshold)
    # The analyze_feature_correlations function returns the optimal feature set

    # Handle missing values
    print("\nStep 4: Handling missing values...")
    for col in feature_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna(0)

    # For test data: fill available features with 0
    if test_df is not None:
        for col in feature_cols:
            if col in test_df.columns:
                test_df[col] = test_df[col].fillna(0)
            else:
                # Add missing columns with 0 values
                test_df[col] = 0

    print(f"Feature engineering complete. Total features: {len(feature_cols)}")
    print(f"Train shape after FE: {train_df.shape}")
    if test_df is not None:
        print(f"Test shape after FE: {test_df.shape}")
    print()

    return train_df, test_df, feature_cols


def prepare_data_for_modeling(df: pd.DataFrame, feature_cols: List[str],
                              target_col: str = 'Weekly_Sales') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for modeling by selecting features and target.

    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        target_col: Target column name

    Returns:
        Tuple of (X, y)
    """
    # Remove rows with NaN in target
    df = df.dropna(subset=[target_col])

    # Select only available features
    available_features = [col for col in feature_cols if col in df.columns]
    X = df[available_features]
    y = df[target_col]

    return X, y


def evaluate_model(model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: pd.Series, y_test: pd.Series,
                   model_name: str) -> Dict:
    """
    Evaluate a trained model on train and test sets.

    Args:
        model: Trained model
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        model_name: Name of the model

    Returns:
        Dictionary of evaluation metrics
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate regression metrics
    metrics = {
        'Model': model_name,
        'Train_MAE': mean_absolute_error(y_train, y_train_pred),
        'Test_MAE': mean_absolute_error(y_test, y_test_pred),
        'Train_RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'Test_RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'Train_R2': r2_score(y_train, y_train_pred),
        'Test_R2': r2_score(y_test, y_test_pred)
    }

    # Calculate classification metrics (categorize as Low/Medium/High sales)
    quartile_25 = np.percentile(y_test, 25)
    quartile_75 = np.percentile(y_test, 75)

    y_test_cat = np.digitize(y_test.values, [quartile_25, quartile_75]) - 1
    y_pred_cat = np.digitize(y_test_pred, [quartile_25, quartile_75]) - 1

    y_test_cat = np.clip(y_test_cat, 0, 2)
    y_pred_cat = np.clip(y_pred_cat, 0, 2)

    metrics['Test_Accuracy'] = accuracy_score(y_test_cat, y_pred_cat)
    metrics['Test_Precision'] = precision_score(
        y_test_cat, y_pred_cat, average='weighted', zero_division=0)
    metrics['Test_Recall'] = recall_score(
        y_test_cat, y_pred_cat, average='weighted', zero_division=0)
    metrics['Test_F1'] = f1_score(
        y_test_cat, y_pred_cat, average='weighted', zero_division=0)

    return metrics


def train_models(X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_train: pd.Series, y_test: pd.Series) -> Dict:
    """
    Train multiple models and evaluate their performance.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target

    Returns:
        Dictionary of trained models and their metrics
    """
    print("=" * 60)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 60)

    models = {}
    results = []

    # 1. Linear Regression
    print("\nTraining Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['LinearRegression'] = lr
    results.append(evaluate_model(lr, X_train, X_test,
                   y_train, y_test, 'LinearRegression'))

    # 2. Ridge Regression
    print("Training Ridge Regression...")
    ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    ridge.fit(X_train, y_train)
    models['Ridge'] = ridge
    results.append(evaluate_model(
        ridge, X_train, X_test, y_train, y_test, 'Ridge'))

    # 3. Random Forest
    print("Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf
    results.append(evaluate_model(rf, X_train, X_test,
                   y_train, y_test, 'RandomForest'))

    # 4. Gradient Boosting
    print("Training Gradient Boosting...")
    gb = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE
    )
    gb.fit(X_train, y_train)
    models['GradientBoosting'] = gb
    results.append(evaluate_model(gb, X_train, X_test,
                   y_train, y_test, 'GradientBoosting'))

    # 5. XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    results.append(evaluate_model(xgb_model, X_train,
                   X_test, y_train, y_test, 'XGBoost'))

    # Display results
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print()

    return models, results_df


def plot_feature_importance(model, feature_names: List[str], model_name: str,
                            output_dir: str = 'output'):
    """
    Plot feature importance for tree-based models.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        model_name: Name of the model
        output_dir: Directory to save plots
    """
    if not hasattr(model, 'feature_importances_'):
        return

    os.makedirs(output_dir, exist_ok=True)

    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]  # Top 20 features

    # Plot
    plt.figure(figsize=(10, 8))
    plt.title(f'Top 20 Feature Importances - {model_name}')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, f'feature_importance_{model_name}.png'), dpi=150)
    plt.close()

    print(
        f"Feature importance plot saved: {output_dir}/feature_importance_{model_name}.png")


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                               model_name: str, output_dir: str = 'output'):
    """
    Plot predictions vs actual values.

    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, s=1)
    plt.plot([y_true.min(), y_true.max()], [
             y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Weekly Sales')
    plt.ylabel('Predicted Weekly Sales')
    plt.title(f'Predictions vs Actual - {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, f'predictions_vs_actual_{model_name}.png'), dpi=150)
    plt.close()


def generate_predictions(model, test_df: pd.DataFrame, feature_cols: List[str],
                         output_path: str = 'output/predictions.csv'):
    """
    Generate predictions for test data.

    Args:
        model: Trained model
        test_df: Test dataframe
        feature_cols: List of feature columns
        output_path: Path to save predictions
    """
    # Select available features
    available_features = [
        col for col in feature_cols if col in test_df.columns]
    X_test = test_df[available_features].fillna(0)

    # Generate predictions
    predictions = model.predict(X_test)

    # Create output dataframe
    output = pd.DataFrame({
        'Store': test_df['Store'],
        'Dept': test_df['Dept'],
        'Date': test_df['Date'],
        'Predicted_Weekly_Sales': predictions
    })

    # Save predictions
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Predictions saved: {output_path}")

    return output


def main():
    """
    Main execution function for the ML pipeline.

    Environment Variables:
        DATA_SOURCE: 'hdfs' or 'mongodb' (default: 'mongodb')
        HDFS_HOST: HDFS namenode host (default: 'localhost')
        HDFS_PORT: HDFS WebHDFS port (default: '9870')
        HDFS_BASE: Base HDFS directory (default: '/user/walmart_sales')
        MONGO_URI: MongoDB connection URI (default: 'mongodb://localhost:27017')
        DB_NAME: MongoDB database name (default: 'walmart_sales')
    """
    # Set paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, '..', 'output')

    # Read configuration from environment variables
    data_source = os.environ.get('DATA_SOURCE', 'mongodb')
    hdfs_base = os.environ.get('HDFS_BASE', '/user/walmart_sales')
    mongo_uri = os.environ.get('MONGO_URI', 'mongodb://localhost:27017')
    db_name = os.environ.get('DB_NAME', 'walmart_sales')

    print("\n" + "=" * 60)
    print("WALMART SALES FORECASTING - ML PIPELINE")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data Source: {data_source}")
    print()

    # Step 1: Load data from specified source
    print("Loading data...")
    merged_train, features_cleaned, stores, test = load_data(
        data_source=data_source,
        hdfs_base=hdfs_base,
        mongo_uri=mongo_uri,
        db_name=db_name
    )

    # Step 2: Prepare test data
    print("Preparing test data...")
    test_prepared = prepare_test_data(test, features_cleaned, stores)

    # Step 3: Feature Engineering
    train_fe, test_fe, feature_cols = feature_engineering(
        merged_train, test_prepared, output_dir)

    # Step 4: Prepare data for modeling
    print("Preparing data for modeling...")
    X, y = prepare_data_for_modeling(train_fe, feature_cols, 'Weekly_Sales')

    # Step 5: Train-test split
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print()

    # Step 6: Train and evaluate models
    models, results_df = train_models(X_train, X_test, y_train, y_test)

    # Step 7: Save results
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(
        output_dir, 'model_comparison.csv'), index=False)
    print(f"Results saved: {output_dir}/model_comparison.csv")

    # Step 8: Plot feature importance for best model (XGBoost)
    print("\nGenerating feature importance plots...")
    best_model_name = results_df.loc[results_df['Test_R2'].idxmax(), 'Model']
    print(f"Best model: {best_model_name}")

    for model_name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            plot_feature_importance(
                model, X_train.columns.tolist(), model_name, output_dir)

    # Step 10: Plot predictions vs actual for best model
    best_model = models[best_model_name]
    y_pred = best_model.predict(X_test)
    plot_predictions_vs_actual(
        y_test.values, y_pred, best_model_name, output_dir)

    # Step 11: Generate predictions on test set using best model
    print("\nGenerating predictions on test set...")
    if test_fe is not None:
        predictions = generate_predictions(
            best_model, test_fe, feature_cols,
            os.path.join(output_dir, 'predictions.csv')
        )
        print(f"\nPredictions summary:")
        print(predictions['Predicted_Weekly_Sales'].describe())

    print("COMPLETE")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


if __name__ == '__main__':
    main()
