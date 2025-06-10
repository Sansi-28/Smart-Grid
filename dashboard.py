import streamlit as st
import requests
import pandas as pd
import altair as alt
from datetime import datetime, timedelta
import pytz

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Grid Command Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for Dark Theme and Styling ---
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
    .stMetricValue {
        color: #28a745; /* Green for metric values */
    }
    .stMetricLabel {
        font-size: 1.1em;
    }
    /* Customize tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a1c24;
    }
</style>
""", unsafe_allow_html=True)

# --- Constants ---
API_URL = "http://localhost:8001/api/v3/system-status"

# --- Helper Functions ---
def get_data():
    """Fetches data from the API and caches it."""
    try:
        response = requests.get(API_URL, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: Could not connect to the Command Center API. Please ensure the backend is running. Details: {e}")
        return None

# --- Main Dashboard Logic ---
api_data = get_data()

# Global Header
if api_data:
    alerts = api_data.get('maintenance_alerts', [])
    status = "OPERATIONAL"
    status_color = "#28a745"  # Green
    if any(alert['priority'] == 'CRITICAL' for alert in alerts):
        status = "CRITICAL ALERT"
        status_color = "#dc3545"  # Red
    elif any(alert['priority'] == 'HIGH' for alert in alerts):
        status = "ELEVATED RISK"
        status_color = "#ffc107"  # Amber

    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px; background-color: #1a1c24; border-radius: 5px; margin-bottom: 20px;">
        <h2 style="color: #FAFAFA; margin: 0;">Smart Grid Command Center</h2>
        <div>
            <span style="color: #FAFAFA; font-weight: bold; margin-right: 15px;">System Status:</span>
            <span style="color: {status_color}; font-weight: bold;">● {status}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

if not api_data:
    st.stop()

# Data Processing
forecast_df = pd.DataFrame(api_data['forecasts'])
forecast_df['timestamp'] = pd.to_datetime(forecast_df['timestamp'])

battery_df = pd.DataFrame(api_data['battery_schedule'])
battery_df['timestamp'] = pd.to_datetime(battery_df['timestamp'])
battery_df['end_time'] = pd.to_datetime(battery_df['end_time'])

# --- Dashboard Tabs (The "Missions") ---
tab1, tab2, tab3 = st.tabs([
    "⚡  Energy Flow & Economics", 
    "🛡️  System Health & Reliability", 
    "📈  Data Explorer"
])

# =================================================================================================
# Mission 1: Energy Flow & Economics
# =================================================================================================
with tab1:
    st.header("The 24-Hour Energy Landscape")
    
    col1, col2 = st.columns([3, 2])  # 60% for chart, 40% for insights

    with col1:
        # Create the multi-layered chart
        base = alt.Chart(forecast_df).encode(
            x=alt.X('timestamp:T', title='Time', axis=alt.Axis(format='%H:%M'))
        )

        # Shaded bands for battery actions
        charge_band = alt.Chart(battery_df[battery_df['action'] == 'CHARGE']).mark_rect(opacity=0.3, color='#28a745').encode(
            x='timestamp:T',
            x2='end_time:T',
            tooltip=['timestamp:T', 'end_time:T', 'action:N', 'reason:N', 'battery_level_kwh:Q']
        )
        discharge_band = alt.Chart(battery_df[battery_df['action'] == 'DISCHARGE']).mark_rect(opacity=0.3, color='#ffc107').encode(
            x='timestamp:T',
            x2='end_time:T',
            tooltip=['timestamp:T', 'end_time:T', 'action:N', 'reason:N', 'battery_level_kwh:Q']
        )

        # Main forecast lines
        load_line = base.mark_line(color='#00A9E0', strokeWidth=3).encode(
            y=alt.Y('load_kw:Q', title='Power (kW)'),
            tooltip=['timestamp:T', 'load_kw:Q']
        )
        solar_area = base.mark_area(color='#28a745', opacity=0.5).encode(
            y=alt.Y('solar_kw:Q', title='Solar Generation (kW)'),
            tooltip=['timestamp:T', 'solar_kw:Q']
        )
        price_line = base.mark_line(color='#ffc107', strokeDash=[5,5]).encode(
            y=alt.Y('price:Q', title='Price ($/kWh)', axis=alt.Axis(titleColor='#ffc107')),
            tooltip=['timestamp:T', 'price:Q']
        )

        # Combine all layers
        final_chart = alt.layer(
            charge_band, discharge_band, load_line, solar_area, price_line
        ).resolve_scale(
            y='independent'
        ).properties(
            title='24-Hour Energy Forecast',
            height=400
        ).interactive()
        
        st.altair_chart(final_chart, use_container_width=True)

    with col2:
        st.subheader("Optimization & Insights")
        
        # --- KPIs ---
        kpi1, kpi2, kpi3 = st.columns(3)
        peak_demand_row = forecast_df.loc[forecast_df['load_kw'].idxmax()]
        renewable_share = forecast_df['solar_kw'].sum() / forecast_df['load_kw'].sum()
        
        kpi1.metric(label="Peak Demand", value=f"{peak_demand_row['load_kw']:.0f} kW", 
                   delta=f"at {peak_demand_row['timestamp'].strftime('%H:%M')}", 
                   delta_color="inverse")
        kpi2.metric(label="Renewable Share", value=f"{renewable_share:.1%}")
        kpi3.metric(label="System Health", value=f"{api_data['system_health']:.1f}%")

        # --- Smart Charge Module ---
        st.subheader("Smart Charge Plan")
        next_action_row = battery_df[battery_df['timestamp'] > datetime.now()].iloc[0] if not battery_df[battery_df['timestamp'] > datetime.now()].empty else battery_df.iloc[-1]
        
        action_color = "#FAFAFA"
        if next_action_row['action'] == "CHARGE": action_color = "#28a745"
        if next_action_row['action'] == "DISCHARGE": action_color = "#ffc107"
        
        st.markdown(f"""
        **Next Action @ {next_action_row['timestamp'].strftime('%H:%M')}:** 
        <span style="color: {action_color}; font-size: 1.5em; font-weight: bold;">{next_action_row['action']}</span>
        <br>
        <small><i>Reason: {next_action_row['reason']}</i></small>
        """, unsafe_allow_html=True)

        with st.expander("View Full Action Log"):
            st.dataframe(battery_df[['timestamp', 'action', 'reason', 'battery_level_kwh']], 
                        use_container_width=True, 
                        hide_index=True)

# =================================================================================================
# Mission 2: System Health & Reliability
# =================================================================================================
with tab2:
    st.header("Predictive Maintenance Triage")

    alerts = api_data.get('maintenance_alerts', [])
    
    if not alerts:
        st.success("## ✅ SYSTEM NOMINAL")
        st.write("No predictive alerts. All components are operating within expected parameters.")
    else:
        for alert in alerts:
            if alert['priority'] == 'CRITICAL':
                st.error(f"## 🚨 {alert['priority']}")
                st.subheader(alert['message'])
                st.metric(label="Fault Probability", value=f"{alert['fault_probability']:.1%}")
                if alert.get('evidence'):
                    st.write("**Evidence:**")
                    for metric, value in alert['evidence'].items():
                        st.write(f"- {metric}: {value:.2f}")
            elif alert['priority'] == 'HIGH':
                st.warning(f"## ⚠️ {alert['priority']}")
                st.subheader(alert['message'])
                st.metric(label="Fault Probability", value=f"{alert['fault_probability']:.1%}")
                if alert.get('evidence'):
                    st.write("**Evidence:**")
                    for metric, value in alert['evidence'].items():
                        st.write(f"- {metric}: {value:.2f}")

    st.info("💡 **Vitals & Fault Forecast:** In a full deployment, this section would include live charts of component vitals (voltage, temperature) and a forward-looking fault probability forecast chart, providing deeper diagnostic context.")

# =================================================================================================
# Mission 3: Data Explorer
# =================================================================================================
with tab3:
    st.header("Explore the Forecast Data")

    st.write("Select metrics from the 24-hour forecast to plot and analyze.")
    
    # Let user select which columns to plot
    all_metrics = forecast_df.columns.drop('timestamp').tolist()
    selected_metrics = st.multiselect(
        'Select metrics to display:',
        options=all_metrics,
        default=['load_kw', 'net_load_kw', 'price']
    )
    
    if selected_metrics:
        explorer_df = forecast_df[['timestamp'] + selected_metrics]
        explorer_df_melted = explorer_df.melt(id_vars=['timestamp'], var_name='Metric', value_name='Value')

        explorer_chart = alt.Chart(explorer_df_melted).mark_line().encode(
            x=alt.X('timestamp:T', title='Time'),
            y=alt.Y('Value:Q', title='Value', scale=alt.Scale(zero=False)),
            color='Metric:N',
            tooltip=['timestamp:T', 'Metric:N', 'Value:Q']
        ).interactive()
        
        st.altair_chart(explorer_chart, use_container_width=True)
    else:
        st.info("Please select at least one metric to plot.")

    st.subheader("What-If Simulation")
    st.info("💡 **Next Step:** A future version of this dashboard could include controls here to adjust parameters like battery capacity or electricity prices and re-run the optimization to see the financial impact, demonstrating the true power of the prescriptive models.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Smart Grid Command Center | Powered by XGBoost and FastAPI</p>
</div>
""", unsafe_allow_html=True) 