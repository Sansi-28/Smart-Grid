from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import pandas as pd
import pickle
import os
from datetime import datetime, timedelta
import numpy as np
import logging
import json
import joblib

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
    'hour', 'dayofweek', 'power_kw'
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
class SystemVitals(BaseModel):
    """Real-time system vitals"""
    timestamp: str
    voltage: float = Field(..., description="Current voltage in volts")
    current: float = Field(..., description="Current in amperes")
    power_factor: float = Field(..., description="Power factor")
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., description="Humidity percentage")
    grid_power: float = Field(..., description="Current grid power in kW")
    solar_power: float = Field(..., description="Current solar generation in kW")
    battery_power: float = Field(..., description="Current battery power in kW")
    battery_soc: float = Field(..., description="Battery state of charge percentage")
    load_power: float = Field(..., description="Current load power in kW")
    grid_price: float = Field(..., description="Current grid price in $/kWh")

class Forecast(BaseModel):
    """Energy forecast data point"""
    timestamp: str
    predicted_load: float = Field(..., description="Predicted load in kW")
    solar_power: float = Field(..., description="Predicted solar generation in kW")
    grid_supply: float = Field(..., description="Predicted grid supply in kW")
    price: float = Field(..., description="Predicted price in $/kWh")
    net_load_kw: float = Field(..., description="Net load (load - solar) in kW")

class BatteryAction(BaseModel):
    """Battery optimization action"""
    timestamp: str
    action: str  # 'charge' or 'discharge'
    power: float
    start_time: str
    end_time: str
    confidence: float

class MaintenanceAlert(BaseModel):
    """Maintenance alert"""
    timestamp: str
    component: str = Field(..., description="Component requiring maintenance")
    severity: str = Field(..., description="Severity level: 'critical', 'warning', or 'info'")
    description: str = Field(..., description="Alert description")
    probability: float = Field(..., description="Probability of failure (0-1)")
    recommended_action: str = Field(..., description="Recommended maintenance action")
    impact: str = Field(..., description="Potential impact if not addressed")
    evidence: Dict[str, Any] = Field(..., description="Supporting evidence for the alert")

class SystemAnalytics(BaseModel):
    """System analytics and metrics"""
    timestamp: str
    daily_stats: Dict[str, float] = Field(..., description="Daily statistics")
    weekly_stats: Dict[str, float] = Field(..., description="Weekly statistics")
    monthly_stats: Dict[str, float] = Field(..., description="Monthly statistics")
    cost_metrics: Dict[str, float] = Field(..., description="Cost-related metrics")
    efficiency_metrics: Dict[str, float] = Field(..., description="Efficiency metrics")
    reliability_metrics: Dict[str, float] = Field(..., description="Reliability metrics")
    environmental_metrics: Dict[str, float] = Field(..., description="Environmental impact metrics")

class SystemStatus(BaseModel):
    """Complete system status response"""
    timestamp: str
    forecasts: List[Forecast]
    battery_schedule: List[BatteryAction]
    maintenance_alerts: List[MaintenanceAlert]
    system_health: float
    live_vitals: Dict[str, float] = Field(default_factory=dict)
    analytics_data: Dict[str, Any] = Field(default_factory=dict)

# --- Prescriptive Logic (Battery and Maintenance) ---
def generate_battery_schedule(df: pd.DataFrame) -> List[BatteryAction]:
    schedule = []
    for index, row in df.iterrows():
        net_load = row['power_kw'] - row['solar_power']
        price = row['price']
        
        # Simple battery optimization logic
        if price > 0.5 and net_load > 0:  # High price, high load - discharge
            action = "discharge"
            power = min(net_load, 5.0)  # Max 5kW discharge
        elif price < 0.3 and net_load < 0:  # Low price, excess solar - charge
            action = "charge"
            power = min(abs(net_load), 5.0)  # Max 5kW charge
        else:
            action = "idle"
            power = 0.0
            
        end_time = (pd.to_datetime(index) + pd.Timedelta(hours=1)).isoformat()
        
        schedule.append(BatteryAction(
            timestamp=str(index),
            action=action,
            power=float(power),
            start_time=str(index),
            end_time=end_time,
            confidence=0.9
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
        # Load models
        logger.info("Loading models...")
        load_model = joblib.load('app/models/load_forecaster.pkl')
        solar_model = joblib.load('app/models/solar_forecaster.pkl')
        price_model = joblib.load('app/models/price_forecaster.pkl')
        fault_model = joblib.load('app/models/fault_predictor.pkl')
        logger.info("All models loaded successfully")
        
        # Create feature DataFrame
        logger.info("Creating feature DataFrame...")
        future_timestamps = pd.date_range(
            start=pd.Timestamp.now(),
            periods=24,
            freq='H'
        )
        
        future_df = pd.DataFrame(index=future_timestamps)
        future_df['hour'] = future_df.index.hour
        future_df['dayofweek'] = future_df.index.dayofweek
        future_df['month'] = future_df.index.month
        future_df['year'] = future_df.index.year
        future_df['dayofyear'] = future_df.index.dayofyear
        
        # Add static features (using last known values)
        future_df['Voltage (V)'] = 230.0  # Nominal voltage
        future_df['Current (A)'] = 10.0   # Nominal current
        future_df['power_kw'] = 2.0       # Base load
        future_df['Reactive Power (kVAR)'] = 0.5
        future_df['Power Factor'] = 0.95
        future_df['Temperature (°C)'] = 25.0
        future_df['voltage_fluctuation'] = 0.02
        future_df['Humidity (%)'] = 60.0
        
        logger.info(f"Available features: {future_df.columns.tolist()}")
        
        # Make predictions
        logger.info("Making load predictions...")
        load_forecast = load_model.predict(future_df[load_features])
        logger.info("Load predictions completed")
        
        logger.info("Making solar predictions...")
        solar_forecast = solar_model.predict(future_df[solar_features])
        logger.info("Solar predictions completed")
        
        logger.info("Making price predictions...")
        price_forecast = price_model.predict(future_df[price_features])
        logger.info("Price predictions completed")
        
        # Update DataFrame with predictions
        future_df = future_df.copy()
        future_df['power_kw'] = load_forecast
        future_df['solar_power'] = solar_forecast
        future_df['price'] = price_forecast
        
        # Generate battery schedule
        logger.info("Generating battery schedule...")
        battery_schedule = generate_battery_schedule(future_df)
        logger.info("Battery schedule generated")
        
        # Create forecasts list
        logger.info("Creating forecasts...")
        forecasts = []
        for i, ts in enumerate(future_df.index):
            forecasts.append(Forecast(
                timestamp=str(ts),
                predicted_load=float(load_forecast[i]),
                grid_supply=float(load_forecast[i] - solar_forecast[i]),
                solar_power=float(solar_forecast[i]),
                price=float(price_forecast[i]),
                net_load_kw=float(load_forecast[i] - solar_forecast[i])
            ))
        logger.info("Forecasts created")
        
        # Generate maintenance alerts
        logger.info("Making fault predictions...")
        fault_probabilities = fault_model.predict_proba(future_df[fault_features])[:, 1]
        logger.info("Fault predictions completed")
        
        maintenance_alerts = []
        for i, ts in enumerate(future_df.index):
            if fault_probabilities[i] > 0.7:
                maintenance_alerts.append(MaintenanceAlert(
                    timestamp=str(ts),
                    component="Grid",
                    severity="high",
                    description="High probability of grid instability",
                    evidence={
                        "fault_probability": float(fault_probabilities[i]),
                        "voltage": float(future_df.iloc[i]['Voltage (V)']),
                        "current": float(future_df.iloc[i]['Current (A)'])
                    }
                ))
        
        # Calculate system health score
        health_score = 1.0 - np.mean(fault_probabilities)
        
        # Create live vitals and analytics data
        live_vitals = {
            "voltage": float(future_df.iloc[-1]['Voltage (V)']),
            "current": float(future_df.iloc[-1]['Current (A)']),
            "power": float(future_df.iloc[-1]['power_kw']),
            "temperature": float(future_df.iloc[-1]['Temperature (°C)'])
        }
        
        analytics_data = {
            "peak_load": float(np.max(load_forecast)),
            "avg_price": float(np.mean(price_forecast)),
            "solar_generation": float(np.sum(solar_forecast)),
            "battery_cycles": len([a for a in battery_schedule if a.action != "idle"])
        }
        
        logger.info("Successfully generated all predictions and alerts")
        
        return SystemStatus(
            timestamp=pd.Timestamp.now().isoformat(),
            forecasts=forecasts,
            battery_schedule=battery_schedule,
            maintenance_alerts=maintenance_alerts,
            system_health=float(health_score),
            live_vitals=live_vitals,
            analytics_data=analytics_data
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in get_system_status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced API Models
class ContributingFactor(BaseModel):
    feature: str
    importance: float

class LiveVitals(BaseModel):
    voltage: float
    current: float
    power_factor: float
    grid_supply: float
    solar_power: float
    wind_power: Optional[float] = 0.0

class Forecast(BaseModel):
    timestamp: datetime
    predicted_load: float
    grid_supply: float
    solar_power: float
    price: float

class BatteryAction(BaseModel):
    timestamp: datetime
    action: str
    reason: str
    power: Optional[float] = None

class AnalyticsData(BaseModel):
    load_heatmap: List[Dict[str, float]]
    correlation_matrix: Dict[str, Dict[str, float]]

class SystemStatus(BaseModel):
    timestamp: datetime
    live_vitals: LiveVitals
    forecasts: List[Forecast]
    battery_schedule: List[BatteryAction]
    maintenance_alerts: List[MaintenanceAlert]
    analytics_data: AnalyticsData
    system_health: float

def generate_analytics_data(df: pd.DataFrame) -> AnalyticsData:
    """Generate analytics data including heatmap and correlation matrix."""
    # Create load heatmap data
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['dayofweek'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    heatmap_data = df.groupby(['hour', 'dayofweek'])['power_kw'].mean().reset_index()
    
    # Create correlation matrix
    numeric_cols = ['power_kw', 'solar_kw', 'price', 'voltage', 'current', 'power_factor']
    corr_matrix = df[numeric_cols].corr()
    
    return AnalyticsData(
        load_heatmap=heatmap_data.to_dict('records'),
        correlation_matrix=corr_matrix.to_dict()
    )

def get_contributing_factors(fault_prob: float, features: Dict[str, float]) -> List[ContributingFactor]:
    """Generate contributing factors for maintenance alerts."""
    # Sort features by their absolute values
    sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
    total_importance = sum(abs(v) for v in features.values())
    
    return [
        ContributingFactor(
            feature=k,
            importance=abs(v) / total_importance
        )
        for k, v in sorted_features[:3]  # Top 3 contributing factors
    ]

@app.get("/api/v3/system-status", response_model=SystemStatus)
async def get_system_status():
    try:
        # Get current timestamp
        current_time = datetime.now()
        
        # Generate forecast data
        forecast_hours = pd.date_range(start=current_time, periods=24, freq='H')
        forecast_data = []
        
        for timestamp in forecast_hours:
            # Create features for prediction
            features = {
                'hour': timestamp.hour,
                'dayofweek': timestamp.dayofweek,
                'month': timestamp.month,
                'year': timestamp.year,
                'dayofyear': timestamp.dayofyear,
                'Temperature (°C)': 25.0,  # Example temperature
                'Humidity (%)': 60.0,      # Example humidity
                'power_kw': 3000.0         # Example load
            }
            
            # Make predictions
            load_pred = load_model_obj.predict(pd.DataFrame([features]))[0]
            solar_pred = solar_model_obj.predict(pd.DataFrame([features]))[0]
            price_pred = price_model_obj.predict(pd.DataFrame([features]))[0]
            
            forecast_data.append(Forecast(
                timestamp=timestamp,
                predicted_load=load_pred,
                grid_supply=max(0, load_pred - solar_pred),
                solar_power=solar_pred,
                price=price_pred
            ))
        
        # Generate battery schedule
        battery_schedule = []
        for forecast in forecast_data:
            if forecast.price < 0.1:  # Low price period
                battery_schedule.append(BatteryAction(
                    timestamp=forecast.timestamp,
                    action="CHARGE",
                    reason="Low electricity price"
                ))
            elif forecast.price > 0.2 and forecast.predicted_load > 4000:  # High price and high load
                battery_schedule.append(BatteryAction(
                    timestamp=forecast.timestamp,
                    action="DISCHARGE",
                    reason="High demand and price"
                ))
        
        # Generate maintenance alerts
        maintenance_alerts = []
        current_features = {
            'voltage': 239.5,
            'current': 150.2,
            'power_kw': 3500.0,
            'power_factor': 0.98,
            'voltage_fluctuation': 0.02
        }
        
        fault_prob = fault_model_obj.predict_proba(pd.DataFrame([current_features]))[0][1]
        if fault_prob > 0.8:
            maintenance_alerts.append(MaintenanceAlert(
                priority="CRITICAL",
                message="Immediate inspection required on Transformer 4B",
                fault_probability=fault_prob,
                contributing_factors=get_contributing_factors(fault_prob, current_features)
            ))
        elif fault_prob > 0.6:
            maintenance_alerts.append(MaintenanceAlert(
                priority="HIGH",
                message="Schedule maintenance for Transformer 4B",
                fault_probability=fault_prob,
                contributing_factors=get_contributing_factors(fault_prob, current_features)
            ))
        
        # Generate live vitals
        live_vitals = LiveVitals(
            voltage=current_features['voltage'],
            current=current_features['current'],
            power_factor=current_features['power_factor'],
            grid_supply=current_features['power_kw'],
            solar_power=forecast_data[0].solar_power
        )
        
        # Generate analytics data
        df = pd.read_csv('data/smart_grid_data.csv')
        analytics_data = generate_analytics_data(df)
        
        return SystemStatus(
            timestamp=current_time,
            live_vitals=live_vitals,
            forecasts=forecast_data,
            battery_schedule=battery_schedule,
            maintenance_alerts=maintenance_alerts,
            analytics_data=analytics_data,
            system_health=1.0 - max([alert.fault_probability for alert in maintenance_alerts] + [0.0])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 