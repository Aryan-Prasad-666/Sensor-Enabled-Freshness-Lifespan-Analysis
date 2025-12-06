# Machine Learning-Based Prediction and Visualization of Food Freshness
### Using Multi-Sensor Gas Readings and Advanced Time-Series Modeling  
**Python • scikit-learn • XGBoost**  
**License: MIT**

This project develops a complete machine learning system to predict the **remaining freshness lifespan of bananas** using hourly readings from multiple gas sensors (MQ2, MQ3, MQ4, MQ5, MQ9, MQ135) along with temperature and humidity.

The workflow includes:

- Data cleaning and interpolation  
- Advanced time-series feature engineering  
- Feature selection  
- Training five machine learning models  
- Evaluation using **Leave-One-Banana-Out (LOBO)** cross-validation  
- Visualizations of model behavior  

**XGBoost achieves the highest R² and lowest error**, demonstrating that gas-sensor patterns can reliably estimate the hours left before spoilage.

---
<p style="display: inline-block;">
  <img src="https://github.com/user-attachments/assets/056b7338-2832-498f-968c-d19034b00bbd" width="300" style="margin-right: 20px;"/>
  <img src="https://github.com/user-attachments/assets/b0697af4-b4c3-46aa-8d6f-fe99c427bb83" width="300"/>
</p>

---
## Project Overview

Bananas emit **ethylene and VOCs** as they ripen. MQ-series gas sensors detect these compounds.  
This system processes time-series data from nine bananas, engineers features capturing spoilage patterns, and trains machine learning models to forecast **Hours Left to Rot**.

### Key Highlights

- **172 engineered features**, including lags, rolling statistics, differences, and nonlinear feature interactions  
- **Feature reduction** using RandomForest importance + correlation filtering  
- **Strict LOBO-CV evaluation** for real-world generalization  
- **XGBoost** provides the best performance (R² = 0.46)

---

## Dataset

Nine CSV files:
Banana1_with_rot_hours.csv
Banana2_with_rot_hours.csv
...
Banana9_with_rot_hours.csv

### Columns

| Column | Description |
|--------|-------------|
| Hour | Time step (0 to rot) |
| MQ2–MQ135 | Gas sensor readings |
| Temp | Temperature (°C, imputed) |
| Hum | Humidity (% imputed) |
| Hours Left to Rot | Target value |

### Example Row
```
Hour,MQ2,MQ3,MQ4,MQ5,MQ9,MQ135,Temp,Hum,Hours Left to Rot
0,244,433,241,340,126,428,30.4,59.1,94
```


---

## Workflow

### 1. Data Cleaning

- Converts Temp/Hum to numeric  
- Fixes NaNs using forward fill → backward fill → linear interpolation  
- Ensures smooth time-series before feature engineering  

### 2. Feature Engineering

Base columns:
```
['Temp', 'Hum', 'MQ2', 'MQ3', 'MQ4', 'MQ5', 'MQ9', 'MQ135']
```

Generated features:

- **Lag features:** 1–5 hour lags  
- **Rolling statistics:** mean, std, min, max over windows 5/10/20  
- **Differences:** 1st and 2nd order  
- **Interactions:**  
  - Temp × Hum  
  - MQ3 × MQ135  
  - MQ2 ÷ MQ5  
  - MQ9 ÷ Hum  
  - Temp × MQ2  

Total engineered features: **172**.

### 3. Feature Selection

- RandomForest `SelectFromModel` (median threshold) → keeps ~86 features  
- Correlation filtering (> 0.9) removes redundant rolling/lags  

### 4. Modeling

Five models:

- Support Vector Regression (RBF)  
- Gradient Boosting Regressor  
- XGBoost Regressor  
- Random Forest Regressor  
- Linear Regression  

All wrapped in a `StandardScaler + Model` pipeline.

### 5. Evaluation

Uses **Leave-One-Banana-Out**:

1. Train on 8 bananas  
2. Test on 1 banana  
3. Repeat 9 times  

Metrics collected:

- R²  
- RMSE  
- MAE  
- MSE  

Visualizations include:

- Actual vs Predicted scatter plots  
- Residual distributions  
- Feature importance plots  
- Boxplots of metrics  

---

## Results

### LOBO-CV Performance

| Model | Avg R² | Avg RMSE | Avg MAE | Avg MSE |
|--------|--------|-----------|-----------|-----------|
| **XGBoost** | **0.458** | **24.79** | **20.82** | 681.06 |
| Gradient Boosting | 0.403 | 25.44 | 21.11 | 749.21 |
| Random Forest | 0.360 | 26.70 | 22.16 | 805.78 |
| Linear Regression | 0.226 | 28.91 | 24.75 | 892.18 |
| SVR | -0.032 | 34.98 | 28.72 | 1322.89 |

### Insights

- XGBoost captures nonlinear VOC patterns best  
- MAE ≈ 21 hours is practical for spoilage estimation  
- Linear and SVR models underperform due to high nonlinearity  

---

## Future Work

- Hyperparameter tuning (Optuna/Bayesian optimization)
- Ensemble stacking
- Real-time inference via Flask/Streamlit
- Edge device integration (Raspberry Pi + MQ sensors)
- Multi-fruit prediction
- Sensor + image multimodal fusion



