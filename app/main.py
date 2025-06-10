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
    end_time: str  # Added for visualization bands

class MaintenanceAlert(BaseModel):
    timestamp: str
    priority: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    message: str
    fault_probability: float
    evidence: Optional[Dict[str, float]] = None  # Added for detailed diagnostics

class SystemStatus(BaseModel):
    timestamp: str
    forecasts: List[Forecast]  # Changed to match new dashboard expectations
    battery_schedule: List[BatteryAction]
    maintenance_alerts: List[MaintenanceAlert]
    system_health: float
    component_vitals: Optional[Dict[str, List[float]]] = None  # Added for future use

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
        end_time = index + timedelta(minutes=15)  # 15-minute intervals
        
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
            battery_level_kwh=current_battery_level,
            end_time=str(end_time)
        ))
    return schedule

def generate_maintenance_alerts(fault_probabilities, forecast_df):
    alerts = []
    max_prob = max(fault_probabilities) if fault_probabilities else 0
    
    if max_prob > 0.85:
        alerts.append(MaintenanceAlert(
            timestamp=pd.Timestamp.now().isoformat(),
            priority="CRITICAL",
            message="Immediate inspection required due to high probability of imminent fault.",
            fault_probability=max_prob,
            evidence={
                "voltage_fluctuation": float(forecast_df['voltage_fluctuation'].mean()),
                "temperature": float(forecast_df['Temperature (°C)'].mean()),
                "power_factor": float(forecast_df['Power Factor'].mean())
            }
        ))
    elif max_prob > 0.70:
        alerts.append(MaintenanceAlert(
            timestamp=pd.Timestamp.now().isoformat(),
            priority="HIGH",
            message="Schedule inspection within 24 hours. Elevated fault risk detected.",
            fault_probability=max_prob,
            evidence={
                "voltage_fluctuation": float(forecast_df['voltage_fluctuation'].mean()),
                "temperature": float(forecast_df['Temperature (°C)'].mean()),
                "power_factor": float(forecast_df['Power Factor'].mean())
            }
        ))
    
    return alerts

# --- Final API Endpoint ---
@app.get("/api/v3/system-status", response_model=SystemStatus)
async def get_system_status():
    """Get current system status including load forecasts and fault predictions."""
    try:
        # Load the trained models
        try:
            logger.info("Loading models...")
            load_model_obj = load_model('load_forecaster.pkl')
            solar_model_obj = load_model('solar_forecaster.pkl')
            price_model_obj = load_model('price_forecaster.pkl')
            fault_model_obj = load_model('fault_predictor.pkl')
            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")
        
        # Create future timestamps (next 24 hours)
        future_timestamps = pd.date_range(
            start=pd.Timestamp.now(),
            periods=24,
            freq='h'
        )
        
        # Create a DataFrame for future predictions with all required features
        future_df = pd.DataFrame(index=future_timestamps)
        
        # Add all features needed for predictions with exact names matching training
        logger.info("Creating feature DataFrame...")
        future_df['Voltage (V)'] = 230.0
        future_df['Current (A)'] = 100.0
        future_df['power_kw'] = 50.0
        future_df['Reactive Power (kVAR)'] = 20.0
        future_df['Power Factor'] = 0.95
        future_df['Temperature (°C)'] = 25.0
        future_df['voltage_fluctuation'] = 0.02
        
        # Add time-based features needed for load and solar predictions
        future_df['hour'] = future_df.index.hour
        future_df['dayofweek'] = future_df.index.dayofweek
        future_df['month'] = future_df.index.month
        future_df['year'] = future_df.index.year
        future_df['dayofyear'] = future_df.index.dayofyear
        future_df['Humidity (%)'] = 50.0
        
        # Log available features
        logger.info(f"Available features: {future_df.columns.tolist()}")
        
        # Verify all required features are present
        missing_features = set(fault_features) - set(future_df.columns)
        if missing_features:
            error_msg = f"Missing features in DataFrame: {missing_features}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Make predictions with error handling for each model
        try:
            logger.info("Making load predictions...")
            load_forecast = load_model_obj.predict(future_df[load_features])
            logger.info("Load predictions completed")
            
            # Add load_kw to the DataFrame for price prediction
            future_df['load_kw'] = load_forecast
        except Exception as e:
            logger.error(f"Error in load prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error in load prediction: {str(e)}")
            
        try:
            logger.info("Making solar predictions...")
            solar_forecast = solar_model_obj.predict(future_df[solar_features])
            logger.info("Solar predictions completed")
            future_df['solar_kw'] = solar_forecast
        except Exception as e:
            logger.error(f"Error in solar prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error in solar prediction: {str(e)}")
            
        try:
            logger.info("Making price predictions...")
            price_forecast = price_model_obj.predict(future_df[price_features])
            logger.info("Price predictions completed")
            future_df['price'] = price_forecast
        except Exception as e:
            logger.error(f"Error in price prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error in price prediction: {str(e)}")
        
        # Calculate net load for battery optimization
        future_df['net_load_kw'] = future_df['load_kw'] - future_df['solar_kw']
        
        # Create forecasts list
        forecasts = []
        for i, ts in enumerate(future_timestamps):
            forecasts.append(Forecast(
                timestamp=str(ts),
                load_kw=float(load_forecast[i]),
                solar_kw=float(solar_forecast[i]),
                price=float(price_forecast[i]),
                net_load_kw=float(load_forecast[i] - solar_forecast[i])
            ))
        
        # Generate battery schedule
        battery_schedule = generate_battery_schedule(future_df)
        
        # Make fault predictions
        try:
            logger.info("Making fault predictions...")
            fault_probabilities = fault_model_obj.predict_proba(future_df[fault_features])[:, 1]
            logger.info("Fault predictions completed")
        except Exception as e:
            logger.error(f"Error in fault prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error in fault prediction: {str(e)}")
        
        # Create maintenance alerts based on fault probabilities
        maintenance_alerts = generate_maintenance_alerts(list(fault_probabilities), future_df)
        
        # Calculate system health score
        health_score = 100 - (fault_probabilities.mean() * 100)
        
        logger.info("Successfully generated all predictions and alerts")
        
        return SystemStatus(
            timestamp=pd.Timestamp.now().isoformat(),
            forecasts=forecasts,
            battery_schedule=battery_schedule,
            maintenance_alerts=maintenance_alerts,
            system_health=health_score
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_system_status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}") 