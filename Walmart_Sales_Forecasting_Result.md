# Walmart Sales Forecasting - Model Results and Analysis

**Project:** Walmart Sales Forecast Big Data Pipeline  
**Dataset:** Walmart Recruiting - Store Sales Forecasting  
**Date:** April 2026

---

## Executive Summary

This document presents the results of our machine learning pipeline for Walmart weekly sales forecasting. We trained and evaluated five regression models on preprocessed data containing 421,570 records with 42 engineered features. **XGBoost achieved the best performance** with a Test R² of 0.992 and the lowest prediction error.

---

## Model Performance Comparison

### Overall Results

| Model | Train MAE | Test MAE | Train RMSE | Test RMSE | Train R² | Test R² |
|-------|-----------|----------|------------|-----------|----------|---------|
| LinearRegression | 1,497.56 | 1,524.53 | 4,105.78 | 4,125.85 | 0.9672 | 0.9674 |
| Ridge | 1,497.53 | 1,524.50 | 4,105.78 | 4,125.85 | 0.9672 | 0.9674 |
| RandomForest | 551.92 | 961.57 | 1,528.94 | 2,849.89 | 0.9955 | 0.9844 |
| GradientBoosting | 1,004.34 | 1,066.74 | 1,939.28 | 2,352.61 | 0.9927 | 0.9894 |
| **XGBoost** | **734.49** | **885.19** | **1,281.88** | **2,063.56** | **0.9968** | **0.9918** |

### Key Findings

1. **XGBoost is the Best Performer**
   - Lowest Test MAE (885.19) and Test RMSE (2,063.56)
   - Highest Test R² (0.9918), explaining 99.18% of variance
   - Good generalization with minimal overfitting (Train R² 0.9968 vs Test R² 0.9918)

2. **Tree-Based Models Outperform Linear Models**
   - RandomForest, GradientBoosting, and XGBoost all achieve R² > 0.98
   - Linear models (LinearRegression, Ridge) plateau at R² ~0.967
   - Non-linear relationships in sales data favor tree-based approaches

3. **RandomForest Shows Signs of Overfitting**
   - Largest gap between Train R² (0.9955) and Test R² (0.9844)
   - High Test RMSE (2,849.89) compared to XGBoost

---

## Feature Importance Analysis

### Top Predictive Features (Across All Models)

Based on feature importance charts from GradientBoosting, RandomForest, and XGBoost:

#### 1. **Weekly_Sales_rolling_mean_4** (Dominant Feature)
- **Importance:** ~75-95% across all tree models
- **Insight:** The 4-week rolling average of sales is by far the strongest predictor
- **Business Interpretation:** Recent sales history (last 4 weeks) is the best indicator of future sales

#### 2. **Weekly_Sales_rolling_std_4** (Secondary)
- **Importance:** ~2-5%
- **Insight:** Volatility in recent sales adds predictive power
- **Business Interpretation:** Sales stability/variability patterns help predict future performance

#### 3. **Weekly_Sales_lag_1 & Weekly_Sales_lag_2** (Tertiary)
- **Importance:** ~1-3% each
- **Insight:** Immediate past sales (1-2 weeks ago) provide additional signal
- **Business Interpretation:** Short-term trends and momentum matter

#### 4. **Store_Dept_Sales_Mean** (Contextual)
- **Importance:** ~1-2%
- **Insight:** Historical average for specific store-department combinations
- **Business Interpretation:** Each store-department has unique baseline performance

#### 5. **Dept_Sales_Mean** (Department Baseline)
- **Importance:** ~0.5-1%
- **Insight:** Department-level averages provide context
- **Business Interpretation:** Some departments consistently outperform others

#### 6. **IsHoliday_x** (Seasonal Factor)
- **Importance:** ~0.5-1%
- **Insight:** Holiday periods significantly impact sales
- **Business Interpretation:** Holiday promotions and shopping patterns are important

### Feature Categories by Importance

| Category | Top Features | Overall Impact |
|----------|--------------|----------------|
| **Temporal/Rolling** | Weekly_Sales_rolling_mean_4, Weekly_Sales_rolling_std_4, Weekly_Sales_lag_1, Weekly_Sales_lag_2 | **~95%** |
| **Aggregated Statistics** | Store_Dept_Sales_Mean, Dept_Sales_Mean, Store_Dept_Sales_Std | **~3%** |
| **Seasonal** | IsHoliday_x, Season, Week, Month | **~1%** |
| **External Factors** | MarkDown3, Temperature, CPI | **<1%** |

---

## Prediction Quality Analysis

### XGBoost Predictions vs Actual

The scatter plot of XGBoost predictions reveals:

- **Strong Linear Relationship:** Points cluster tightly around the diagonal line
- **Homoscedasticity:** Variance is relatively consistent across sales ranges
- **Outliers:** Some high-value sales (>300,000) show more dispersion
- **Accuracy:** Most predictions fall within ±50,000 of actual values

### Error Distribution

Based on Test RMSE values:

| Model | Avg Prediction Error | Relative to Mean Sales* |
|-------|---------------------|------------------------|
| XGBoost | ±2,063.56 | ~4.1% |
| GradientBoosting | ±2,352.61 | ~4.7% |
| RandomForest | ±2,849.89 | ~5.7% |
| Linear Models | ±4,125.85 | ~8.2% |

*Mean weekly sales approximately $50,000

---

## Business Insights and Conclusions

### 1. Sales Are Highly Predictable from Recent History

The dominance of rolling mean features (especially 4-week) indicates that **Walmart sales follow strong temporal patterns**. This suggests:
- Sales forecasting should prioritize recent performance data
- Short-term trends are more predictive than long-term historical averages
- Seasonal adjustments are secondary to recent momentum

### 2. Store-Department Specificity Matters

Store_Dept_Sales_Mean and Dept_Sales_Mean feature importance confirms that:
- Different departments have fundamentally different sales patterns
- Store-specific factors (location, size, demographics) impact performance
- One-size-fits-all forecasting is suboptimal

### 3. Promotional Impact (MarkDown) is Moderate

MarkDown3 appears in top features but with low importance (~0.5%), suggesting:
- Promotions have measurable but not dominant impact on sales
- Other factors (recent sales history, store characteristics) are more predictive
- Markdown effectiveness varies significantly by context

### 4. External Economic Factors Have Limited Direct Impact

Temperature, CPI, and Unemployment show minimal feature importance:
- These may have indirect effects captured through sales history
- Weekly sales are more driven by operational factors than macroeconomic conditions
- Store-specific dynamics outweigh broader economic trends

### 5. Holiday Effects Are Significant but Predictable

IsHoliday_x consistently appears in top features:
- Holiday periods create predictable sales spikes
- The model successfully captures these seasonal patterns
- Holiday forecasting should incorporate this feature

---

## Recommendations

### For Production Deployment

1. **Use XGBoost as Primary Model**
   - Best accuracy with good generalization
   - Balanced performance across all metrics
   - Efficient inference time

2. **Feature Engineering Priorities**
   - Focus on rolling window features (4-week, 8-week, 12-week)
   - Include lag features (1-week, 2-week, 4-week)
   - Maintain store-department aggregation features
   - Keep holiday indicators

3. **Data Collection Emphasis**
   - Prioritize accurate recent sales data
   - Ensure complete store-department mapping
   - Track promotional calendars (MarkDown data)
   - Monitor holiday schedules

### For Future Improvements

1. **Address High-Value Outliers**
   - Investigate prediction errors for sales >$300,000
   - Consider separate models for high-volume store-departments
   - Add outlier detection and special handling

2. **Explore Advanced Temporal Features**
   - Year-over-year comparisons
   - Fourier features for seasonality
   - Trend decomposition

3. **Ensemble Approaches**
   - Combine XGBoost with GradientBoosting
   - Weighted averaging based on validation performance
   - Stacking with meta-learner

4. **External Data Integration**
   - Weather data for temperature-sensitive departments
   - Local events and competitor activity
   - Economic indicators with proper lag structures

---

## Technical Notes

### Model Configuration

- **XGBoost:** n_estimators=200, max_depth=8, learning_rate=0.1
- **RandomForest:** n_estimators=100, max_depth=20
- **GradientBoosting:** n_estimators=100, max_depth=6
- **Train/Test Split:** 80/20 with random_state=42

### Feature Engineering Summary

- **Base Features:** Store, Dept, Size, Type, Temperature, Fuel_Price, CPI, Unemployment, MarkDown1-5
- **Temporal Features:** Year, Month, Week, Day, DayOfYear, Quarter, Season
- **Lag Features:** Weekly_Sales_lag_1, _lag_2, _lag_4, _lag_8
- **Rolling Features:** Weekly_Sales_rolling_mean_4/8/12, Weekly_Sales_rolling_std_4/8/12
- **Aggregated Features:** Store_Sales_Mean/Std/Min/Max, Dept_Sales_Mean/Std, Store_Dept_Sales_Mean/Std

---

## Conclusion

Our machine learning pipeline successfully achieved **99.18% accuracy (R²)** in forecasting Walmart weekly sales using XGBoost. The analysis reveals that recent sales history is the dominant predictive factor, with 4-week rolling averages providing the strongest signal. The model is suitable for production deployment and can support inventory management, staffing decisions, and promotional planning.

The big data pipeline architecture—combining HDFS for storage, MongoDB for analytics, and automated Docker deployment—provides a scalable foundation for real-time forecasting at Walmart's scale.

---

**Team:** 7095 Team  
**Members:** Wei Liming, Ke Linyao, Huo Weijia, Xu Hao, Zhang Hongyang
