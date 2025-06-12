import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Grid Mission Control",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for a Professional Look ---
st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    h1, h2, h3 {
        color: #00A9E0; /* Electric Blue for headers */
    }
    .stMetric {
        background-color: #1a1c24;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #2c303b;
    }
    .stMetricValue {
        color: #FAFAFA;
    }
    .stMetricDelta {
        color: #b0b3b8 !important;
    }
    .stButton>button {
        border-radius: 20px;
        border: 1px solid #00A9E0;
        background-color: transparent;
        color: #00A9E0;
    }
    .stButton>button:hover {
        border: 1px solid #FAFAFA;
        color: #FAFAFA;
    }
</style>
""", unsafe_allow_html=True)

# --- API Communication ---
API_URL = "http://127.0.0.1:8002/api/v3/system-status"

@st.cache_data(ttl=60)  # Cache data for 60 seconds
def get_data():
    """Fetches data from the API and caches it."""
    try:
        response = requests.get(API_URL, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: Could not connect to Mission Control API. Details: {e}")
        return None

# --- Helper function to reconstruct a forecasts DataFrame from new fields ---
def get_forecast_df(api_data):
    """Reconstruct a forecasts DataFrame from load_forecast, solar_forecast, price_forecast (and a simulated grid_supply)."""
    if not api_data or not all(k in api_data for k in ('load_forecast', 'solar_forecast', 'price_forecast')):
         st.error("API data missing required forecast fields.")
         return pd.DataFrame()
    # Assume that load_forecast, solar_forecast, price_forecast are lists of equal length (e.g. 24 hourly values).
    n = len(api_data['load_forecast'])
    # (In a real dashboard, you might also have a list of timestamps; here we simulate a timestamp (e.g. hourly) for demo.)
    timestamps = [pd.Timestamp(api_data['timestamp']) + pd.Timedelta(hours=i) for i in range(n)]
    # (Simulate grid_supply as (predicted_load – solar_power) clipped to non-negative.)
    grid_supply = [max(0, l - s) for l, s in zip(api_data['load_forecast'], api_data['solar_forecast'])]
    df = pd.DataFrame({
         'timestamp': timestamps,
         'predicted_load': api_data['load_forecast'],
         'grid_supply': grid_supply,
         'solar_power': api_data['solar_forecast'],
         'price': api_data['price_forecast']
    })
    return df

# --- Main Dashboard Logic ---
api_data = get_data()

if not api_data:
    st.warning("Awaiting data from the Command Center API...")
    st.stop()

# ================================================================================================
# SECTION 1: THE COMMAND BRIDGE (Situational Awareness)
# ================================================================================================

# --- Determine System Status ---
alerts = api_data.get('maintenance_alerts', [])
status = "OPERATIONAL"
status_color = "#28a745"  # Green
if any(alert['severity'] == 'CRITICAL' for alert in alerts):
    status = "CRITICAL ALERT"
    status_color = "#dc3545"  # Red
elif any(alert['severity'] == 'HIGH' for alert in alerts):
    status = "ELEVATED RISK"
    status_color = "#ffc107"  # Amber

st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: center; padding: 10px; background-color: #1a1c24; border-radius: 10px; border: 1px solid {status_color}; margin-bottom: 20px;">
    <h1 style="color: #FAFAFA; margin: 0; font-size: 2.5em;">Mission Control</h1>
    <div>
        <span style="color: #FAFAFA; font-weight: bold; margin-right: 15px; font-size: 1.2em;">System Status:</span>
        <span style="color: {status_color}; font-weight: bold; font-size: 1.2em;">● {status}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- KPIs ---
live_vitals = api_data.get('live_vitals', {})
# (Reconstruct a forecasts DataFrame using our helper.)
forecast_df = get_forecast_df(api_data)
# (If forecast_df is empty, you might want to display a fallback or error message.)
if forecast_df.empty:
     st.error("Could not reconstruct forecast data. Please check API data.")
     st.stop()
# (Convert timestamp to datetime if it isn't already.)
forecast_df['timestamp'] = pd.to_datetime(forecast_df['timestamp'])

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label="Current Grid Load", 
        value=f"{live_vitals.get('grid_supply', 0):.1f} kW"
    )
with col2:
    peak_load = forecast_df['predicted_load'].max()
    peak_time = forecast_df.loc[forecast_df['predicted_load'].idxmax()]['timestamp']
    st.metric(
        label="Predicted 24h Peak", 
        value=f"{peak_load:.0f} kW", 
        delta=f"at {peak_time.strftime('%H:%M')}"
    )
with col3:
    renewable_gen = live_vitals.get('solar_power', 0) + live_vitals.get('wind_power', 0)
    st.metric(
        label="Current Renewable Input", 
        value=f"{renewable_gen:.1f} kW",
        help="Sum of Solar and Wind generation."
    )
with col4:
    health_score = api_data.get('system_health', 1.0)
    st.metric(
        label="System Health Score", 
        value=f"{health_score:.2%}",
        delta="Lower is worse", delta_color="inverse"
    )

st.markdown("---")

# ================================================================================================
# SECTION 2: THE 24-HOUR OPERATIONAL PLAN
# ================================================================================================
st.header("The 24-Hour Operational Plan")

op_col1, op_col2 = st.columns([3, 2])

with op_col1:
    # --- Main Forecast Chart ---
    fig = go.Figure()
    # Layer 1: Stacked area for supply
    fig.add_trace(go.Scatter(
        x=forecast_df['timestamp'], 
        y=forecast_df['grid_supply'], 
        mode='lines', 
        line_width=0, 
        stackgroup='one', 
        name='Grid Supply', 
        fillcolor='rgba(255, 193, 7, 0.4)'
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df['timestamp'], 
        y=forecast_df['solar_power'], 
        mode='lines', 
        line_width=0, 
        stackgroup='one', 
        name='Solar Power', 
        fillcolor='rgba(40, 167, 69, 0.5)'
    ))
    
    # Layer 2: Total Predicted Load line
    fig.add_trace(go.Scatter(
        x=forecast_df['timestamp'], 
        y=forecast_df['predicted_load'], 
        mode='lines', 
        name='Total Predicted Load', 
        line=dict(color='#00A9E0', width=4)
    ))

    # Layer 3: Price on secondary axis
    fig.add_trace(go.Scatter(
        x=forecast_df['timestamp'], 
        y=forecast_df['price'], 
        mode='lines', 
        name='Electricity Price', 
        line=dict(color='#FAFAFA', width=2, dash='dot'), 
        yaxis='y2'
    ))

    # Layer 4: Battery action bands
    battery_df = pd.DataFrame(api_data['battery_schedule'])
    battery_df['timestamp'] = pd.to_datetime(battery_df['timestamp'])
    for _, row in battery_df.iterrows():
        color = 'rgba(40, 167, 69, 0.2)' if row['action'] == 'CHARGE' else 'rgba(255, 193, 7, 0.2)'
        fig.add_vrect(
            x0=row['timestamp'], 
            x1=row['timestamp'] + pd.Timedelta(minutes=15), 
            fillcolor=color, 
            layer="below", 
            line_width=0
        )
    
    fig.update_layout(
        title=dict(
            text="Energy Landscape: Supply, Demand & Price Forecast",
            font=dict(size=24, color="#FAFAFA")
        ),
        xaxis=dict(
            title=dict(
                text="Time",
                font=dict(size=14, color="#FAFAFA")
            ),
            tickfont=dict(size=12, color="#FAFAFA"),
            gridcolor="#333333"
        ),
        yaxis=dict(
            title=dict(
                text="kW",
                font=dict(size=14, color="#FAFAFA")
            ),
            tickfont=dict(size=12, color="#FAFAFA"),
            gridcolor="#333333"
        ),
        yaxis2=dict(
            title=dict(
                text="Price ($/kWh)",
                font=dict(size=14, color="#FAFAFA")
            ),
            tickfont=dict(size=12, color="#FAFAFA"),
            gridcolor="#333333",
            overlaying="y",
            side="right"
        ),
        plot_bgcolor="#1a1c24",
        paper_bgcolor="#1a1c24",
        legend=dict(
            font=dict(size=12, color="#FAFAFA"),
            bgcolor="#1a1c24",
            bordercolor="#333333"
        ),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

with op_col2:
    st.subheader("Battery Action Plan")
    st.dataframe(
        battery_df[['timestamp', 'action', 'reason']], 
        use_container_width=True, 
        hide_index=True
    )

st.markdown("---")

# ================================================================================================
# SECTION 3: DIAGNOSTIC ANALYTICS CENTER
# ================================================================================================
st.header("Diagnostic Analytics Center")

diag_col1, diag_col2 = st.columns(2)

with diag_col1:
    st.subheader("Typical Load Patterns")
    # --- Heatmap Section ---
    analytics_data = api_data.get('analytics_data')
    if analytics_data and 'load_heatmap' in analytics_data:
        heatmap_data = pd.DataFrame(analytics_data['load_heatmap'])
        pivot = heatmap_data.pivot(index='dayofweek', columns='hour', values='avg_load')
        day_map = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
        pivot = pivot.rename(index=day_map)
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='Viridis'
        ))
        fig.update_layout(
            title='Average Load by Hour and Day of Week', 
            yaxis_title="Day", 
            xaxis_title="Hour of Day", 
            plot_bgcolor='#1a1c24', 
            paper_bgcolor='#1a1c24', 
            font_color='#FAFAFA'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No load heatmap analytics data available from the API.")

with diag_col2:
    st.subheader("System Variable Correlations")
    # --- Correlation Matrix Section ---
    if analytics_data and 'correlation_matrix' in analytics_data:
        corr_df = pd.DataFrame(analytics_data['correlation_matrix'])
        fig = ff.create_annotated_heatmap(
            z=corr_df.values,
            x=list(corr_df.columns),
            y=list(corr_df.index),
            colorscale='Blues',
            showscale=True,
            font_colors=['white','black']
        )
        fig.update_layout(
            title='Correlation Matrix of Key Variables', 
            plot_bgcolor='#1a1c24', 
            paper_bgcolor='#1a1c24', 
            font_color='#FAFAFA'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No correlation analytics data available from the API.")

st.markdown("---")

# ================================================================================================
# SECTION 4: SYSTEM HEALTH & RELIABILITY
# ================================================================================================
st.header("System Health & Reliability")

health_col1, health_col2 = st.columns([1, 2])

with health_col1:
    st.subheader("Live Vitals")
    # --- Live Vitals Gauges ---
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = live_vitals.get('voltage', 0),
        domain = {'x': [0, 1], 'y': [0.6, 1]},
        title = {'text': "Voltage (V)"},
        gauge = {
            'axis': {'range': [210, 250]}, 
            'bar': {'color': "#00A9E0"}, 
            'steps': [
                {'range': [210, 220], 'color': 'yellow'}, 
                {'range': [240, 250], 'color': 'yellow'}
            ]
        }
    ))
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = live_vitals.get('power_factor', 0),
        domain = {'x': [0, 1], 'y': [0, 0.4]},
        title = {'text': "Power Factor"},
        gauge = {
            'axis': {'range': [0.8, 1]}, 
            'bar': {'color': "#00A9E0"}, 
            'steps': [{'range': [0.8, 0.9], 'color': 'yellow'}]
        }
    ))
    fig.update_layout(
        paper_bgcolor = "#1a1c24", 
        height=300, 
        margin=dict(l=20, r=20, t=40, b=20), 
        font = {'color': "white"}
    )
    st.plotly_chart(fig, use_container_width=True)

with health_col2:
    st.subheader("System Health & Alerts")
    
    # System Health Score
    health_score = api_data.get('system_health', 1.0)
    st.metric("System Health Score", f"{health_score:.1%}")
    
    # Get alerts from API response
    alerts = api_data.get('maintenance_alerts', [])
    
    # Alert Status
    if alerts:
        critical_alerts = sum(1 for alert in alerts if alert['severity'].upper() == 'CRITICAL')
        high_risk_alerts = sum(1 for alert in alerts if alert['severity'].upper() == 'HIGH')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Critical Alerts", critical_alerts, delta=None, 
                     delta_color="inverse" if critical_alerts > 0 else "off")
        with col2:
            st.metric("High Risk Alerts", high_risk_alerts, delta=None,
                     delta_color="inverse" if high_risk_alerts > 0 else "off")
        
        # Maintenance Alert Visualization
        st.subheader("Predictive Maintenance Triage")
        
        # Convert alerts to DataFrame for plotting
        alert_df = pd.DataFrame(alerts)
        alert_df['timestamp'] = pd.to_datetime(alert_df['timestamp'])
        alert_df['severity'] = alert_df['severity'].str.upper()  # Normalize severity to uppercase
        
        # Define severity levels and colors
        severity_levels = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
        severity_colors = {
            'CRITICAL': '#ff4b4b',  # Red
            'HIGH': '#ffa726',      # Orange
            'MEDIUM': '#ffeb3b',    # Yellow
            'LOW': '#66bb6a'        # Green
        }
        
        # Create scatter plot
        fig = go.Figure()
        
        # Add scatter points for each severity level
        for severity in severity_levels:
            mask = alert_df['severity'] == severity
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=alert_df[mask]['timestamp'],
                    y=[severity] * mask.sum(),
                    mode='markers',
                    name=severity.capitalize(),
                    marker=dict(
                        color=severity_colors[severity],
                        size=12,
                        symbol='circle'
                    ),
                    hovertemplate=(
                        "<b>%{x}</b><br>" +
                        "Severity: %{y}<br>" +
                        "Message: " + alert_df[mask]['message'].iloc[0] + "<br>" +
                        "<extra></extra>"
                    )
                ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="Maintenance Alerts Over Time",
                font=dict(size=16, color='white')
            ),
            xaxis=dict(
                title=dict(
                    text="Time",
                    font=dict(color='white')
                ),
                gridcolor='rgba(128, 128, 128, 0.2)',
                tickfont=dict(color='white')
            ),
            yaxis=dict(
                title=dict(
                    text="Alert Severity",
                    font=dict(color='white')
                ),
                categoryorder='array',
                categoryarray=severity_levels,
                gridcolor='rgba(128, 128, 128, 0.2)',
                tickfont=dict(color='white')
            ),
            plot_bgcolor='#1a1c24',
            paper_bgcolor='#1a1c24',
            showlegend=True,
            legend=dict(
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(255,255,255,0.2)',
                borderwidth=1,
                font=dict(color='white')
            ),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Alert Details
        st.subheader("Alert Details")
        for alert in alerts:
            severity = alert['severity'].upper()
            severity_color = severity_colors.get(severity, '#ffffff')
            st.markdown(f"""
                <div style='padding: 10px; border-left: 4px solid {severity_color}; background-color: rgba(255,255,255,0.05); margin: 5px 0;'>
                    <p style='margin: 0;'><strong>Time:</strong> {alert['timestamp']}</p>
                    <p style='margin: 0;'><strong>Severity:</strong> {severity.capitalize()}</p>
                    <p style='margin: 0;'><strong>Message:</strong> {alert['message']}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No active maintenance alerts")

# Section 5: Battery Optimization
with st.container():
    st.header("Battery Optimization")
    
    # Battery Schedule
    battery_schedule = api_data.get('battery_schedule', [])
    if battery_schedule:
        # Convert schedule to DataFrame
        schedule_df = pd.DataFrame(battery_schedule)
        schedule_df['timestamp'] = pd.to_datetime(schedule_df['timestamp'])
        
        # Create battery schedule visualization
        fig = go.Figure()
        
        # Add battery level line
        fig.add_trace(go.Scatter(
            x=schedule_df['timestamp'],
            y=schedule_df['battery_level_kwh'],
            mode='lines',
            name='Battery Level',
            line=dict(color='#00ff00', width=2)
        ))
        
        # Add charging/discharging markers
        charge_mask = schedule_df['action'] == 'CHARGE'
        discharge_mask = schedule_df['action'] == 'DISCHARGE'
        
        if charge_mask.any():
            fig.add_trace(go.Scatter(
                x=schedule_df[charge_mask]['timestamp'],
                y=schedule_df[charge_mask]['battery_level_kwh'],
                mode='markers',
                name='Charging',
                marker=dict(
                    color='#00ff00',
                    size=8,
                    symbol='triangle-up'
                )
            ))
        
        if discharge_mask.any():
            fig.add_trace(go.Scatter(
                x=schedule_df[discharge_mask]['timestamp'],
                y=schedule_df[discharge_mask]['battery_level_kwh'],
                mode='markers',
                name='Discharging',
                marker=dict(
                    color='#ff0000',
                    size=8,
                    symbol='triangle-down'
                )
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="Battery Schedule",
                font=dict(size=16, color='white')
            ),
            xaxis=dict(
                title=dict(
                    text="Time",
                    font=dict(color='white')
                ),
                gridcolor='rgba(128, 128, 128, 0.2)',
                tickfont=dict(color='white')
            ),
            yaxis=dict(
                title=dict(
                    text="Battery Level (kWh)",
                    font=dict(color='white')
                ),
                range=[0, 500],  # Based on BATTERY_CAPACITY_KWH from API
                gridcolor='rgba(128, 128, 128, 0.2)',
                tickfont=dict(color='white')
            ),
            plot_bgcolor='#1a1c24',
            paper_bgcolor='#1a1c24',
            showlegend=True,
            legend=dict(
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(255,255,255,0.2)',
                borderwidth=1,
                font=dict(color='white')
            ),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Schedule Details
        st.subheader("Schedule Details")
        for entry in battery_schedule:
            action_color = '#00ff00' if entry['action'] == 'CHARGE' else '#ff0000' if entry['action'] == 'DISCHARGE' else '#ffffff'
            st.markdown(f"""
                <div style='padding: 10px; border-left: 4px solid {action_color}; background-color: rgba(255,255,255,0.05); margin: 5px 0;'>
                    <p style='margin: 0;'><strong>Time:</strong> {entry['timestamp']}</p>
                    <p style='margin: 0;'><strong>Action:</strong> {entry['action']}</p>
                    <p style='margin: 0;'><strong>Battery Level:</strong> {entry['battery_level_kwh']:.1f} kWh</p>
                    <p style='margin: 0;'><strong>Reason:</strong> {entry['reason']}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No battery schedule available") 