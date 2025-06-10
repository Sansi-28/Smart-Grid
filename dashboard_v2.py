import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
from datetime import datetime, timedelta
import json

# Page config
st.set_page_config(
    page_title="Smart Grid Mission Control v2.0",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        background-color: #262730;
        border-radius: 4px 4px 0 0;
        gap: 1rem;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
    }
    .metric-card {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1E88E5;
    }
    .alert-card {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid;
    }
    .critical { border-color: #EF5350; }
    .warning { border-color: #FFA726; }
    .info { border-color: #29B6F6; }
</style>
""", unsafe_allow_html=True)

# Constants
API_URL = "http://localhost:8002/api/v3/system-status"
REFRESH_INTERVAL = 60  # seconds

def fetch_system_status():
    """Fetch system status from API"""
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching system status: {str(e)}")
        return None

def create_vitals_display(vitals):
    """Create display for system vitals"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Grid Power", f"{vitals['grid_power']:.1f} kW")
        st.metric("Solar Power", f"{vitals['solar_power']:.1f} kW")
    with col2:
        st.metric("Battery Power", f"{vitals['battery_power']:.1f} kW")
        st.metric("Battery SoC", f"{vitals['battery_soc']:.1f}%")
    with col3:
        st.metric("Load Power", f"{vitals['load_power']:.1f} kW")
        st.metric("Grid Price", f"${vitals['grid_price']:.2f}/kWh")
    with col4:
        st.metric("Voltage", f"{vitals['voltage']:.1f} V")
        st.metric("Current", f"{vitals['current']:.1f} A")

def create_forecast_chart(forecasts):
    """Create interactive forecast chart using Altair"""
    df = pd.DataFrame(forecasts)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create base chart
    base = alt.Chart(df).encode(
        x='timestamp:T',
        tooltip=['timestamp:T', 'predicted_load:Q', 'solar_power:Q', 'grid_supply:Q', 'price:Q']
    )
    
    # Create layers
    load_line = base.mark_line(color='#1E88E5').encode(
        y=alt.Y('predicted_load:Q', title='Power (kW)'),
        tooltip=['predicted_load:Q']
    )
    
    solar_line = base.mark_line(color='#43A047').encode(
        y='solar_power:Q',
        tooltip=['solar_power:Q']
    )
    
    grid_line = base.mark_line(color='#EF5350').encode(
        y='grid_supply:Q',
        tooltip=['grid_supply:Q']
    )
    
    # Create price chart
    price_chart = alt.Chart(df).mark_line(color='#FFA726').encode(
        x='timestamp:T',
        y=alt.Y('price:Q', title='Price ($/kWh)'),
        tooltip=['price:Q']
    )
    
    # Combine charts
    chart = alt.layer(load_line, solar_line, grid_line).properties(
        height=300,
        title='Energy Forecast'
    )
    
    price_chart = price_chart.properties(
        height=150,
        title='Price Forecast'
    )
    
    return alt.vconcat(chart, price_chart)

def create_battery_schedule_chart(schedule):
    """Create battery schedule visualization"""
    df = pd.DataFrame(schedule)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    chart = alt.Chart(df).mark_bar().encode(
        x='timestamp:T',
        y=alt.Y('power:Q', title='Battery Power (kW)'),
        color=alt.Color('action_type:N', scale=alt.Scale(
            domain=['charge', 'discharge', 'idle'],
            range=['#43A047', '#EF5350', '#9E9E9E']
        )),
        tooltip=['timestamp:T', 'power:Q', 'action_type:N', 'reason:N', 'expected_savings:Q']
    ).properties(
        height=300,
        title='Battery Schedule'
    )
    
    return chart

def create_maintenance_alerts(alerts):
    """Display maintenance alerts"""
    if not alerts:
        st.info("No maintenance alerts at this time")
        return
    
    for alert in alerts:
        severity_class = alert['severity'].lower()
        st.markdown(f"""
        <div class="alert-card {severity_class}">
            <h4>{alert['component']} - {alert['severity'].upper()}</h4>
            <p>{alert['description']}</p>
            <p><strong>Probability:</strong> {alert['probability']:.1%}</p>
            <p><strong>Recommended Action:</strong> {alert['recommended_action']}</p>
            <p><strong>Impact:</strong> {alert['impact']}</p>
        </div>
        """, unsafe_allow_html=True)

def create_analytics_display(analytics):
    """Display system analytics"""
    # Daily Stats
    st.subheader("Daily Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Energy Consumed", f"{analytics['daily_stats']['total_energy_consumed']:.1f} kWh")
    with col2:
        st.metric("Solar Generated", f"{analytics['daily_stats']['total_solar_generated']:.1f} kWh")
    with col3:
        st.metric("Peak Load", f"{analytics['daily_stats']['peak_load']:.1f} kW")
    with col4:
        st.metric("Avg Price", f"${analytics['daily_stats']['average_price']:.2f}/kWh")
    
    # Cost Metrics
    st.subheader("Cost Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Daily Cost", f"${analytics['cost_metrics']['daily_cost']:.2f}")
    with col2:
        st.metric("Daily Savings", f"${analytics['cost_metrics']['daily_savings']:.2f}")
    with col3:
        st.metric("Battery Savings", f"${analytics['cost_metrics']['battery_savings']:.2f}")
    
    # Efficiency Metrics
    st.subheader("Efficiency Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Solar Utilization", f"{analytics['efficiency_metrics']['solar_utilization']:.1%}")
    with col2:
        st.metric("Grid Efficiency", f"{analytics['efficiency_metrics']['grid_efficiency']:.1%}")
    with col3:
        st.metric("Battery Efficiency", f"{analytics['efficiency_metrics']['battery_efficiency']:.1%}")
    
    # Environmental Impact
    st.subheader("Environmental Impact")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Carbon Reduction", f"{analytics['environmental_metrics']['carbon_reduction']:.1f} kg CO2")
    with col2:
        st.metric("Renewable Percentage", f"{analytics['environmental_metrics']['renewable_percentage']:.1f}%")

def main():
    st.title("⚡ Smart Grid Mission Control v2.0")
    
    # Fetch data
    data = fetch_system_status()
    if not data:
        st.error("Unable to fetch system status. Please check the API connection.")
        return
    
    # System Health Score
    health_score = data['system_health']
    health_color = "green" if health_score >= 80 else "orange" if health_score >= 60 else "red"
    st.markdown(f"""
    <div style='text-align: center; padding: 1rem; background-color: #262730; border-radius: 0.5rem; margin-bottom: 1rem;'>
        <h2 style='color: {health_color};'>System Health: {health_score:.1f}%</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "🎯 Energy Flow & Economics",
        "🔧 System Health & Reliability",
        "📊 Data Explorer"
    ])
    
    with tab1:
        st.header("Energy Flow & Economics")
        
        # System Vitals
        st.subheader("Live System Vitals")
        create_vitals_display(data['live_vitals'])
        
        # Forecasts
        st.subheader("Energy Forecasts")
        st.altair_chart(create_forecast_chart(data['forecasts']), use_container_width=True)
        
        # Battery Schedule
        st.subheader("Battery Optimization")
        st.altair_chart(create_battery_schedule_chart(data['battery_schedule']), use_container_width=True)
        
        # Cost Analysis
        st.subheader("Cost Analysis")
        create_analytics_display(data['analytics_data'])
    
    with tab2:
        st.header("System Health & Reliability")
        
        # Maintenance Alerts
        st.subheader("Maintenance Alerts")
        create_maintenance_alerts(data['maintenance_alerts'])
        
        # Reliability Metrics
        st.subheader("Reliability Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("System Health", f"{data['analytics_data']['reliability_metrics']['system_health']:.1f}%")
        with col2:
            st.metric("Voltage Stability", f"{data['analytics_data']['reliability_metrics']['voltage_stability']:.1f}%")
        with col3:
            st.metric("Power Quality", f"{data['analytics_data']['reliability_metrics']['power_quality']:.1f}%")
    
    with tab3:
        st.header("Data Explorer")
        
        # Raw Data View
        st.subheader("Raw Data")
        data_tab1, data_tab2, data_tab3 = st.tabs(["Forecasts", "Battery Schedule", "Analytics"])
        
        with data_tab1:
            st.dataframe(pd.DataFrame(data['forecasts']))
        with data_tab2:
            st.dataframe(pd.DataFrame(data['battery_schedule']))
        with data_tab3:
            st.json(data['analytics_data'])

if __name__ == "__main__":
    main()
