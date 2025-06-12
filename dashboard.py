import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz

# Set page config
st.set_page_config(
    page_title="Smart Grid Dashboard",
    layout="wide"
)

# Constants
API_URL = "http://localhost:8001/api/v3/system-status"

# Title and description
st.title("Smart Grid Dashboard")
st.markdown("""
This dashboard provides real-time load forecasting for the next 24 hours,
helping grid operators make informed decisions about power distribution.
""")

def create_forecast_plot(data):
    """Create a plotly figure for the forecast data."""
    # Create DataFrame from the response format
    timestamps = pd.date_range(
        start=pd.Timestamp(data['timestamp']),
        periods=len(data['load_forecast']),
        freq='h'
    )
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'Load Forecast (kW)': data['load_forecast'],
        'Solar Forecast (kW)': data['solar_forecast'],
        'Price Forecast ($/kWh)': data['price_forecast']
    })
    
    # Create the plot
    fig = go.Figure()
    
    # Add load forecast
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['Load Forecast (kW)'],
        name='Load Forecast',
        line=dict(color='blue')
    ))
    
    # Add solar forecast
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['Solar Forecast (kW)'],
        name='Solar Forecast',
        line=dict(color='green')
    ))
    
    # Add price forecast on secondary y-axis
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['Price Forecast ($/kWh)'],
        name='Price Forecast',
        line=dict(color='red'),
        yaxis='y2'
    ))
    
    # Update layout
    fig.update_layout(
        title='24-Hour Load, Solar, and Price Forecast',
        xaxis_title='Time',
        yaxis_title='Power (kW)',
        yaxis2=dict(
            title='Price ($/kWh)',
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def create_maintenance_plot(data):
    """Create a plotly figure for maintenance alerts."""
    if not data['maintenance_alerts']:
        return None
        
    # Create DataFrame from maintenance alerts
    df = pd.DataFrame(data['maintenance_alerts'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create the plot
    fig = go.Figure()
    
    # Add alerts as scatter points
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=[1] * len(df),  # Constant y-value for all points
        mode='markers+text',
        marker=dict(
            size=15,
            color=['red' if s == 'high' else 'orange' for s in df['severity']]
        ),
        text=df['message'],
        name='Maintenance Alerts'
    ))
    
    # Update layout
    fig.update_layout(
        title='Maintenance Alerts',
        xaxis_title='Time',
        yaxis=dict(showticklabels=False, range=[0, 2]),
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig

def create_battery_plot(data):
    """Create a plotly figure for battery optimization schedule."""
    if not data['battery_schedule']:
        return None
        
    # Create DataFrame from battery schedule
    df = pd.DataFrame([{
        'timestamp': pd.to_datetime(action['timestamp']),
        'battery_level': action['battery_level_kwh'],
        'action': action['action'],
        'reason': action['reason']
    } for action in data['battery_schedule']])
    
    # Create the plot
    fig = go.Figure()
    
    # Add battery level line
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['battery_level'],
        name='Battery Level',
        line=dict(color='green'),
        mode='lines+markers',
        marker=dict(
            size=10,
            color=['red' if a == 'DISCHARGE' else 'blue' if a == 'CHARGE' else 'gray' 
                  for a in df['action']]
        )
    ))
    
    # Update layout
    fig.update_layout(
        title='Battery Optimization Schedule',
        xaxis_title='Time',
        yaxis_title='Battery Level (kWh)',
        hovermode='x unified',
        showlegend=True,
        annotations=[
            dict(
                x=row['timestamp'],
                y=row['battery_level'],
                text=row['action'],
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40
            ) for _, row in df.iterrows()
        ]
    )
    
    return fig

# Main dashboard layout
try:
    # Fetch data from API
    response = requests.get(API_URL)
    response.raise_for_status()
    data = response.json()
    
    # System Health and Alerts Section
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="System Health Score",
            value=f"{data['system_health']:.1f}%",
            delta=None
        )
    
    # Display maintenance alerts
    if data['maintenance_alerts']:
        with col2:
            st.warning(f"⚠️ {len(data['maintenance_alerts'])} Maintenance Alerts")
            for alert in data['maintenance_alerts']:
                st.error(f"{alert['severity'].upper()}: {alert['message']}")
    
    # Forecast Plot Section
    st.subheader("Load and Solar Forecast")
    forecast_plot = create_forecast_plot(data)
    if forecast_plot:
        st.plotly_chart(forecast_plot, use_container_width=True)
    
    # Battery Optimization Section
    st.subheader("Battery Optimization")
    battery_plot = create_battery_plot(data)
    if battery_plot:
        st.plotly_chart(battery_plot, use_container_width=True)
        
        # Display battery schedule details
        st.write("### Battery Schedule Details")
        battery_df = pd.DataFrame([{
            'Time': pd.to_datetime(action['timestamp']).strftime('%Y-%m-%d %H:%M'),
            'Action': action['action'],
            'Battery Level (kWh)': f"{action['battery_level_kwh']:.1f}",
            'Reason': action['reason']
        } for action in data['battery_schedule']])
        st.dataframe(battery_df, use_container_width=True)
    
    # Maintenance Plot Section
    if data['maintenance_alerts']:
        st.subheader("Maintenance Alerts Timeline")
        maintenance_plot = create_maintenance_plot(data)
        if maintenance_plot:
            st.plotly_chart(maintenance_plot, use_container_width=True)
    
except requests.exceptions.RequestException as e:
    st.error(f"Error fetching data from API: {str(e)}")
    st.info("Make sure the API server is running on port 8001")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please check the API response format and try again")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Smart Grid Load Forecasting Dashboard | Powered by XGBoost and FastAPI</p>
</div>
""", unsafe_allow_html=True) 