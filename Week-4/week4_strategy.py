
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Problem Definition & Synthetic Data Simulation
np.random.seed(42)
n_samples = 1000

# Synthesize operational logistics metrics
data = pd.DataFrame({
    'distance_km': np.random.uniform(5, 150, n_samples),
    'package_weight_kg': np.random.uniform(0.5, 30, n_samples),
    'traffic_index': np.random.uniform(1, 10, n_samples), # 1 = clear, 10 = severe congestion
    'driver_exp_years': np.random.uniform(0.5, 15, n_samples)
})

# Generate ground-truth target (Delivery Delay in minutes)
data['delay_minutes'] = (
    (data['distance_km'] * 0.4) + 
    (data['traffic_index'] * 5) - 
    (data['driver_exp_years'] * 1.2) + 
    np.random.normal(0, 5, n_samples)
).clip(lower=0)

# Feature and Target Split
X = data.drop('delay_minutes', axis=1)
y = data['delay_minutes']

# Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Model Selection & Implementation
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Model Predictions
y_pred = rf_model.predict(X_test)

# 3. Performance Evaluation & Validation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f} minutes")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} minutes")
print(f"R-squared Score (R²): {r2:.4f}")
