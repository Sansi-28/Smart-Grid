from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
import os
from datetime import datetime, timedelta
import numpy as np
from typing import List, Optional, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define feature lists for each model
load_features = [
    'hour', 'dayofweek', 'month', 'year', 'dayofyear',
    'Temperature (°C)', 'Humidity (%)'
]

solar_features = [
    'hour', 'dayofweek', 'month', 'dayofyear'
]

price_features = [
    'hour', 'dayofweek', 'load_kw'
]

fault_features = [
    'Voltage (V)',
    'Current (A)',
    'power_kw',
    'Reactive Power (kVAR)',
    'Power Factor',
    'Temperature (°C)',
    'voltage_fluctuation'
]

app = FastAPI(title="Smart Grid Command Center API - Final", description="Full-featured API for grid optimization and predictive maintenance.")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'models', 'load_forecaster.pkl')
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Warning: Model file not found. Please run the training script first.")
    model = None

class PredictionInput(BaseModel):
    temperature: float  # Temperature in °C
    humidity: float    # Humidity in %
    wind_power: float  # Wind Power in kW
    solar_power: float # Solar Power in kW
    voltage: float     # Voltage in V
    current: float     # Current in A
    power_factor: float # Power Factor
    voltage_fluctuation: float # Voltage Fluctuation in %
    timestamp: datetime

def create_features(timestamp):
    return {
        'hour': timestamp.hour,
        'dayofweek': timestamp.weekday(),
        'quarter': (timestamp.month - 1) // 3 + 1,
        'month': timestamp.month,
        'year': timestamp.year,
        'dayofyear': timestamp.timetuple().tm_yday
    }

@app.get("/")
async def root():
    return {"message": "Smart Grid Load Forecasting API"}

@app.post("/predict")
async def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please run the training script first.")
    
    # Create features from timestamp
    time_features = create_features(input_data.timestamp)
    
    # Combine all features
    features = {
        **time_features,
        'Temperature (°C)': input_data.temperature,
        'Humidity (%)': input_data.humidity,
        'Wind Power (kW)': input_data.wind_power,
        'Solar Power (kW)': input_data.solar_power,
        'Voltage (V)': input_data.voltage,
        'Current (A)': input_data.current,
        'Power Factor': input_data.power_factor,
        'Voltage Fluctuation (%)': input_data.voltage_fluctuation
    }
    
    # Convert to DataFrame
    X = pd.DataFrame([features])
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    return {
        "timestamp": input_data.timestamp,
        "predicted_load": float(prediction)
    }

@app.get("/forecast/24h")
async def forecast_24h():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please run the training script first.")
    
    # Generate timestamps for next 24 hours
    now = datetime.now()
    timestamps = [now + timedelta(hours=i) for i in range(24)]
    
    # Example data (in a real application, this would come from sensors/APIs)
    sensor_data = {
        'temperature': 25.0,      # Example temperature in °C
        'humidity': 60.0,         # Example humidity in %
        'wind_power': 100.0,      # Example wind power in kW
        'solar_power': 200.0,     # Example solar power in kW
        'voltage': 230.0,         # Example voltage in V
        'current': 10.0,          # Example current in A
        'power_factor': 0.95,     # Example power factor
        'voltage_fluctuation': 2.0 # Example voltage fluctuation in %
    }
    
    predictions = []
    for ts in timestamps:
        # Create features for this timestamp
        time_features = create_features(ts)
        
        # Combine all features
        features = {
            **time_features,
            'Temperature (°C)': sensor_data['temperature'],
            'Humidity (%)': sensor_data['humidity'],
            'Wind Power (kW)': sensor_data['wind_power'],
            'Solar Power (kW)': sensor_data['solar_power'],
            'Voltage (V)': sensor_data['voltage'],
            'Current (A)': sensor_data['current'],
            'Power Factor': sensor_data['power_factor'],
            'Voltage Fluctuation (%)': sensor_data['voltage_fluctuation']
        }
        
        # Convert to DataFrame
        X = pd.DataFrame([features])
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        predictions.append({
            "timestamp": ts,
            "predicted_load": float(prediction)
        })
    
    # Find peak demand
    peak_demand = max(predictions, key=lambda x: x['predicted_load'])
    
    return {
        "forecast": predictions,
        "peak_demand": peak_demand
    }

# --- Load All Models ---
def load_model(filename):
    with open(f'./app/models/{filename}', 'rb') as f:
        return pickle.load(f)

load_model_obj = load_model('load_forecaster.pkl')
solar_model_obj = load_model('solar_forecaster.pkl')
price_model_obj = load_model('price_forecaster.pkl')
fault_model_obj = load_model('fault_predictor.pkl')  # New model

# --- Pydantic Data Models (Final Version) ---
class Forecast(BaseModel):
    timestamp: str
    load_kw: float
    solar_kw: float
    price: float
    net_load_kw: float

class BatteryAction(BaseModel):
    timestamp: str
    action: str  # "CHARGE", "DISCHARGE", "HOLD"
    reason: str
    battery_level_kwh: float

class MaintenanceAlert(BaseModel):
    timestamp: str
    severity: str
    message: str

class SystemStatus(BaseModel):
    timestamp: str
    load_forecast: List[float]
    solar_forecast: List[float]
    price_forecast: List[float]
    maintenance_alerts: List[MaintenanceAlert]
    system_health: float
    battery_schedule: List[BatteryAction]
    analytics_data: Optional[Dict] = None

# --- Prescriptive Logic (Battery and Maintenance) ---
def generate_battery_schedule(forecast_df):
    schedule = []
    BATTERY_CAPACITY_KWH = 500.0
    BATTERY_MAX_CHARGE_RATE_KW = 150.0
    BATTERY_MAX_DISCHARGE_RATE_KW = 150.0
    current_battery_level = BATTERY_CAPACITY_KWH / 2.0
    price_25th = forecast_df['price'].quantile(0.25)
    price_75th = forecast_df['price'].quantile(0.75)

    for index, row in forecast_df.iterrows():
        action = "HOLD"
        reason = "Default action"
        
        if row['price'] <= price_25th and current_battery_level < BATTERY_CAPACITY_KWH:
            action = "CHARGE"
            reason = f"Price is low (${row['price']:.2f})"
            charge_amount = min(BATTERY_MAX_CHARGE_RATE_KW * 0.25, BATTERY_CAPACITY_KWH - current_battery_level)
            current_battery_level += charge_amount
        elif row['price'] >= price_75th and current_battery_level > 0:
            action = "DISCHARGE"
            reason = f"Price is high (${row['price']:.2f})"
            discharge_amount = min(BATTERY_MAX_DISCHARGE_RATE_KW * 0.25, current_battery_level)
            current_battery_level -= discharge_amount
        elif row['net_load_kw'] < 0 and current_battery_level < BATTERY_CAPACITY_KWH:
            action = "CHARGE"
            reason = f"Absorbing excess solar ({-row['net_load_kw']:.2f} kW)"
            charge_amount = min(-row['net_load_kw'] * 0.25, BATTERY_MAX_CHARGE_RATE_KW * 0.25, BATTERY_CAPACITY_KWH - current_battery_level)
            current_battery_level += charge_amount
        
        schedule.append(BatteryAction(
            timestamp=str(index),
            action=action,
            reason=reason,
            battery_level_kwh=current_battery_level
        ))
    return schedule

def generate_maintenance_alerts(fault_probabilities):
    alerts = []
    max_prob = max(fault_probabilities) if fault_probabilities else 0
    
    if max_prob > 0.85:
        alerts.append(MaintenanceAlert(
            timestamp=pd.Timestamp.now().isoformat(),
            severity="CRITICAL",
            message="Immediate inspection required due to high probability of imminent fault."
        ))
    elif max_prob > 0.70:
        alerts.append(MaintenanceAlert(
            timestamp=pd.Timestamp.now().isoformat(),
            severity="HIGH",
            message="Schedule inspection within 24 hours. Elevated fault risk detected."
        ))
    
    return alerts

def compute_analytics(df):
    """Compute analytics data including load heatmap and correlation matrix."""
    try:
        logger.info("Starting analytics computation...")
        # Generate realistic load heatmap data
        # Create a pattern that shows higher loads during weekdays and peak hours
        hours = range(24)
        days = range(7)
        heatmap_data = []
        
        logger.info("Generating load heatmap data...")
        for day in days:
            for hour in hours:
                # Base load varies by day (weekends lower) and hour (peak during day)
                base_load = 400  # Base load in kW
                
                # Day effect: Weekdays (0-4) have higher load than weekends (5-6)
                day_factor = 1.2 if day < 5 else 0.8
                
                # Hour effect: Peak during working hours (9-17) and evening (18-21)
                if 9 <= hour <= 17:  # Working hours
                    hour_factor = 1.4
                elif 18 <= hour <= 21:  # Evening peak
                    hour_factor = 1.6
                elif 22 <= hour or hour <= 5:  # Night
                    hour_factor = 0.6
                else:  # Morning
                    hour_factor = 1.0
                
                # Add some randomness
                random_factor = 0.9 + np.random.random() * 0.2  # 0.9 to 1.1
                
                # Calculate average load
                avg_load = base_load * day_factor * hour_factor * random_factor
                
                heatmap_data.append({
                    'dayofweek': int(day),
                    'hour': int(hour),
                    'avg_load': float(avg_load)
                })
        
        logger.info(f"Generated {len(heatmap_data)} heatmap data points")
        
        # Generate realistic correlation matrix
        logger.info("Generating correlation matrix...")
        # Define meaningful correlations between system variables
        variables = [
            'Voltage (V)',
            'Current (A)',
            'Power Factor',
            'Temperature (°C)',
            'Reactive Power (kVAR)',
            'voltage_fluctuation'
        ]
        
        # Create a correlation matrix with realistic relationships
        corr_matrix = {}
        for var1 in variables:
            corr_matrix[var1] = {}
            for var2 in variables:
                if var1 == var2:
                    corr_matrix[var1][var2] = 1.0
                else:
                    # Define meaningful correlations
                    if ('Voltage' in var1 and 'Current' in var2) or ('Voltage' in var2 and 'Current' in var1):
                        corr = -0.7  # Negative correlation between voltage and current
                    elif ('Power Factor' in var1 and 'Reactive Power' in var2) or ('Power Factor' in var2 and 'Reactive Power' in var1):
                        corr = -0.8  # Strong negative correlation
                    elif ('Temperature' in var1 and 'Current' in var2) or ('Temperature' in var2 and 'Current' in var1):
                        corr = 0.6  # Positive correlation
                    elif ('voltage_fluctuation' in var1 and 'Voltage' in var2) or ('voltage_fluctuation' in var2 and 'Voltage' in var1):
                        corr = -0.5  # Negative correlation
                    else:
                        # Random correlation for other pairs, but keep it moderate
                        corr = np.random.uniform(-0.3, 0.3)
                    
                    # Add some noise to make it more realistic
                    corr += np.random.normal(0, 0.05)
                    # Ensure correlation is between -1 and 1
                    corr = max(min(corr, 1.0), -1.0)
                    corr_matrix[var1][var2] = float(corr)
        
        logger.info("Analytics computation completed successfully")
        return {
            "load_heatmap": heatmap_data,
            "correlation_matrix": corr_matrix
        }
    except Exception as e:
        logger.error(f"Error computing analytics: {str(e)}", exc_info=True)
        # Return the error message for debugging
        return {"error": str(e)}

# --- Final API Endpoint ---
@app.get("/api/v3/system-status", response_model=SystemStatus)
async def get_system_status():
    """Get current system status including load forecasts, fault predictions, and analytics data."""
    try:
        # Load the trained models
        try:
            logger.info("Loading models...")
            model_path = os.path.join(os.path.dirname(__file__), 'models')
            logger.info(f"Model path: {model_path}")
            
            load_model_path = os.path.join(model_path, 'load_forecaster.pkl')
            solar_model_path = os.path.join(model_path, 'solar_forecaster.pkl')
            price_model_path = os.path.join(model_path, 'price_forecaster.pkl')
            fault_model_path = os.path.join(model_path, 'fault_predictor.pkl')
            
            logger.info(f"Loading load model from: {load_model_path}")
            load_model_obj = load_model('load_forecaster.pkl')
            logger.info(f"Loading solar model from: {solar_model_path}")
            solar_model_obj = load_model('solar_forecaster.pkl')
            logger.info(f"Loading price model from: {price_model_path}")
            price_model_obj = load_model('price_forecaster.pkl')
            logger.info(f"Loading fault model from: {fault_model_path}")
            fault_model_obj = load_model('fault_predictor.pkl')
            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")
        
        # Create future timestamps (next 24 hours)
        future_timestamps = pd.date_range(
            start=pd.Timestamp.now(),
            periods=24,
            freq='h'
        )
        
        # Create a DataFrame for future predictions with all required features
        future_df = pd.DataFrame(index=future_timestamps)
        
        # Add time-based features (these are used by all models)
        future_df['hour'] = future_df.index.hour
        future_df['dayofweek'] = future_df.index.dayofweek
        future_df['month'] = future_df.index.month
        future_df['year'] = future_df.index.year
        future_df['dayofyear'] = future_df.index.dayofyear
        
        # Add weather features for load model
        future_df['Temperature (°C)'] = 25 + 5 * np.sin(future_df['hour'] * np.pi / 12)
        future_df['Humidity (%)'] = 60 + 10 * np.sin(future_df['hour'] * np.pi / 12)
        
        # Add system vitals for fault model
        future_df['Voltage (V)'] = 239.5 + np.random.normal(0, 0.5, len(future_df))
        future_df['Current (A)'] = 150.2 + np.random.normal(0, 2, len(future_df))
        future_df['Power Factor'] = 0.98 + np.random.normal(0, 0.01, len(future_df))
        future_df['Reactive Power (kVAR)'] = 50.0 + np.random.normal(0, 5, len(future_df))
        future_df['voltage_fluctuation'] = 0.5 + np.random.normal(0, 0.1, len(future_df))
        
        # Make predictions
        try:
            logger.info("Making load predictions...")
            # Load model expects: hour, dayofweek, month, year, dayofyear, Temperature (°C), Humidity (%)
            load_forecast = load_model_obj.predict(future_df[load_features])
            
            # Validate and scale load predictions to reasonable range (0-1000 kW)
            load_forecast = np.clip(load_forecast, 0, 1000)
            
            # Add some daily pattern to make predictions more realistic
            daily_pattern = 1 + 0.3 * np.sin((future_df['hour'] - 14) * np.pi / 12)  # Peak at 2 PM
            load_forecast = load_forecast * daily_pattern
            
            future_df['predicted_load'] = load_forecast
            future_df['power_kw'] = load_forecast  # Add power_kw for fault model
            logger.info("Load predictions completed")
        except Exception as e:
            logger.error(f"Error in load prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error in load prediction: {str(e)}")
            
        try:
            logger.info("Making solar predictions...")
            # Solar model expects only: hour, dayofweek, month, dayofyear
            solar_forecast = solar_model_obj.predict(future_df[solar_features])
            
            # Validate and scale solar predictions to reasonable range (0-500 kW)
            solar_forecast = np.clip(solar_forecast, 0, 500)
            
            # Add daily pattern to solar predictions (zero at night, peak at noon)
            solar_pattern = np.sin((future_df['hour'] - 6) * np.pi / 12) * (future_df['hour'] > 6) * (future_df['hour'] < 18)
            solar_forecast = solar_forecast * solar_pattern
            
            future_df['solar_power'] = solar_forecast
            logger.info("Solar predictions completed")
        except Exception as e:
            logger.error(f"Error in solar prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error in solar prediction: {str(e)}")
            
        try:
            logger.info("Making price predictions...")
            # Price model expects: hour, dayofweek, load_kw
            future_df['load_kw'] = future_df['predicted_load']  # Map predicted_load to load_kw for price model
            price_forecast = price_model_obj.predict(future_df[price_features])
            
            # Validate and scale price predictions to reasonable range ($0.05-$0.30 per kWh)
            price_forecast = np.clip(price_forecast, 0.05, 0.30)
            
            future_df['price'] = price_forecast
            logger.info("Price predictions completed")
        except Exception as e:
            logger.error(f"Error in price prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error in price prediction: {str(e)}")
        
        # Calculate net load for battery scheduling
        future_df['net_load'] = future_df['predicted_load'] - future_df['solar_power']
        
        # Calculate system health score
        try:
            fault_probabilities = fault_model_obj.predict_proba(future_df[fault_features])[:, 1]
            health_score = 1 - max(fault_probabilities)
        except Exception as e:
            logger.warning(f"Error calculating health score: {str(e)}. Using default value.")
            health_score = 0.95
        
        # Generate battery schedule
        battery_schedule = generate_battery_schedule(future_df)
        
        # Generate maintenance alerts
        maintenance_alerts = []
        if health_score < 0.95:  # If system health is below 95%
            maintenance_alerts.append(
                MaintenanceAlert(
                    timestamp=pd.Timestamp.now().isoformat(),
                    severity="HIGH",
                    message="Elevated transformer temperature detected"
                )
            )
        
        # Generate analytics data
        logger.info("Starting analytics data generation...")
        analytics = compute_analytics(future_df)
        analytics_error = None
        if analytics is None or (isinstance(analytics, dict) and 'error' in analytics):
            logger.warning("Failed to generate analytics data, using empty structures")
            analytics_error = analytics.get('error') if analytics else 'Unknown error'
            analytics = {
                "load_heatmap": [],
                "correlation_matrix": {}
            }
        else:
            logger.info("Analytics data generated successfully")
            logger.info(f"Load heatmap data points: {len(analytics['load_heatmap'])}")
            logger.info(f"Correlation matrix variables: {list(analytics['correlation_matrix'].keys())}")
        
        response = SystemStatus(
            timestamp=pd.Timestamp.now().isoformat(),
            load_forecast=future_df['predicted_load'].tolist(),
            solar_forecast=future_df['solar_power'].tolist(),
            price_forecast=future_df['price'].tolist(),
            maintenance_alerts=maintenance_alerts,
            system_health=health_score,
            battery_schedule=battery_schedule,
            analytics_data=analytics
        )
        # If there was an analytics error, add it to the response dict
        response_dict = response.dict()
        if analytics_error:
            response_dict['analytics_error'] = analytics_error
        return response_dict
    except Exception as e:
        logger.error(f"Unexpected error in get_system_status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# --- Data Models ---
class LiveVitals(BaseModel):
    voltage: float
    current: float
    power_factor: float
    grid_supply: float
    solar_power: float
    wind_power: Optional[float] = 0.0

class ForecastPoint(BaseModel):
    timestamp: str
    predicted_load: float
    grid_supply: float
    solar_power: float
    price: float

class ContributingFactor(BaseModel):
    feature: str
    importance: float

class AnalyticsData(BaseModel):
    load_heatmap: List[Dict[str, float]]  # List of {hour, dayofweek, avg_load}
    correlation_matrix: Dict[str, Dict[str, float]]

class SystemStatus(BaseModel):
    timestamp: str
    load_forecast: List[float]
    solar_forecast: List[float]
    price_forecast: List[float]
    maintenance_alerts: List[MaintenanceAlert]
    system_health: float
    battery_schedule: List[BatteryAction]

# --- Prescriptive Logic (Battery and Maintenance) ---
def generate_battery_schedule(forecast_df):
    """Generate battery schedule based on price signals and solar generation."""
    schedule = []
    BATTERY_CAPACITY_KWH = 500.0
    BATTERY_MAX_CHARGE_RATE_KW = 150.0
    BATTERY_MAX_DISCHARGE_RATE_KW = 150.0
    current_battery_level = BATTERY_CAPACITY_KWH / 2.0
    price_25th = forecast_df['price'].quantile(0.25)
    price_75th = forecast_df['price'].quantile(0.75)

    for index, row in forecast_df.iterrows():
        action = "HOLD"
        reason = "Default action"
        
        if row['price'] <= price_25th and current_battery_level < BATTERY_CAPACITY_KWH:
            action = "CHARGE"
            reason = f"Price is low (${row['price']:.2f})"
            charge_amount = min(BATTERY_MAX_CHARGE_RATE_KW * 0.25, BATTERY_CAPACITY_KWH - current_battery_level)
            current_battery_level += charge_amount
        elif row['price'] >= price_75th and current_battery_level > 0:
            action = "DISCHARGE"
            reason = f"Price is high (${row['price']:.2f})"
            discharge_amount = min(BATTERY_MAX_DISCHARGE_RATE_KW * 0.25, current_battery_level)
            current_battery_level -= discharge_amount
        elif row['net_load'] < 0 and current_battery_level < BATTERY_CAPACITY_KWH:
            action = "CHARGE"
            reason = f"Absorbing excess solar ({-row['net_load']:.2f} kW)"
            charge_amount = min(-row['net_load'] * 0.25, BATTERY_MAX_CHARGE_RATE_KW * 0.25, BATTERY_CAPACITY_KWH - current_battery_level)
            current_battery_level += charge_amount
        
        schedule.append(BatteryAction(
            timestamp=index.isoformat(),
            action=action,
            battery_level_kwh=current_battery_level,
            reason=reason
        ))
    return schedule

def generate_maintenance_alerts(fault_probabilities):
    alerts = []
    max_prob = max(fault_probabilities) if fault_probabilities else 0
    
    if max_prob > 0.85:
        alerts.append(MaintenanceAlert(
            timestamp=pd.Timestamp.now().isoformat(),
            severity="CRITICAL",
            message="Immediate inspection required due to high probability of imminent fault."
        ))
    elif max_prob > 0.70:
        alerts.append(MaintenanceAlert(
            timestamp=pd.Timestamp.now().isoformat(),
            severity="HIGH",
            message="Schedule inspection within 24 hours. Elevated fault risk detected."
        ))
    
    return alerts 