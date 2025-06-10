import pandas as pd
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

print("--- Starting Full Model Training Suite ---")

# --- 1. Load and Prepare Data ---
print("Loading data...")
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'smart_grid_data.csv')
df = pd.read_csv(data_path)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.set_index('Timestamp')

# Rename columns for consistency
df = df.rename(columns={
    'Predicted Load (kW)': 'load_kw',
    'Solar Power (kW)': 'solar_kw',
    'Electricity Price (USD/kWh)': 'price',
    'Power Consumption (kW)': 'power_kw',
    'Voltage Fluctuation (%)': 'voltage_fluctuation'
})

# Create a combined fault indicator (1 if either transformer fault or overload condition is True/1)
df['fault'] = ((df['Transformer Fault'] == True) | (df['Overload Condition'] == True)).astype(int)
print("Data loaded and fault indicator created successfully.")

# --- 2. Feature Engineering ---
def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    return df

df = create_features(df)
print("Features created.")

# --- 3. Generic Training Functions ---
def train_regressor_and_save(target_name, features, df, model_filename):
    print(f"\n--- Training regressor for: {target_name} ---")
    X = df[features]
    y = df[target_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    reg = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50, eval_metric='rmse', learning_rate=0.01, n_jobs=-1)
    reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    rmse = ((reg.predict(X_test) - y_test)**2).mean()**0.5
    print(f"Test RMSE for {target_name}: {rmse:.2f}")
    model_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'models', model_filename)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(reg, f)
    print(f"Model saved to: {model_path}")

def train_classifier_and_save(target_name, features, df, model_filename):
    print(f"\n--- Training classifier for: {target_name} ---")
    
    X = df[features]
    y = df[target_name]  # Already binary (0/1) from our fault indicator creation
    
    # For time series data, we keep shuffle=False but remove stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Print class distribution for monitoring
    print(f"Training set class distribution: {y_train.value_counts(normalize=True).to_dict()}")
    print(f"Test set class distribution: {y_test.value_counts(normalize=True).to_dict()}")
    
    clf = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        eval_metric='logloss',
        n_jobs=-1,
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])  # Handle class imbalance
    )
    
    # Updated fit call with correct parameter structure
    clf.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    
    y_pred_proba = clf.predict_proba(X_test)[:, 1]  # Probability of the '1' class
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Test AUC for {target_name}: {auc:.3f}")
    
    model_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'models', model_filename)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Model saved to: {model_path}")

# --- 4. Train All Models ---
# Regressor models
load_features = ['hour', 'dayofweek', 'month', 'year', 'dayofyear', 'Temperature (°C)', 'Humidity (%)']
solar_features = ['hour', 'dayofweek', 'month', 'dayofyear']
price_features = ['hour', 'dayofweek', 'load_kw']

train_regressor_and_save('load_kw', load_features, df, 'load_forecaster.pkl')
train_regressor_and_save('solar_kw', solar_features, df, 'solar_forecaster.pkl')
train_regressor_and_save('price', price_features, df, 'price_forecaster.pkl')

# Classifier model for faults
fault_features = [
    'Voltage (V)', 'Current (A)', 'power_kw', 'Reactive Power (kVAR)', 
    'Power Factor', 'Temperature (°C)', 'voltage_fluctuation'
]
train_classifier_and_save('fault', fault_features, df, 'fault_predictor.pkl')

print("\n--- All models have been trained and saved. ---") 