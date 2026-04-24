import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import random
from test_backend import run_backend
import tempfile
import os 


st.set_page_config(
    page_title="IAF C4I: MAINTENANCE SYSTEM",
    layout="wide",
    initial_sidebar_state="collapsed"
)
#css
st.markdown("""
    <style>
    .stApp {
        background-color: #050505;
        color: #00ff41;
        font-family: 'Courier New', Courier, monospace;
    }
    div[data-testid="stHeader"], div[data-testid="stToolbar"] {
        background-color: #050505;
        color: #00ff41;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #00ff41 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        border-bottom: 1px solid #00ff41;
        padding-bottom: 5px;
    }
    .hud-box {
        border: 2px solid #00ff41;
        background-color: rgba(0, 20, 0, 0.5);
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.2);
    }
    .stat-number {
        font-size: 32px;
        font-weight: bold;
    }
    .stButton>button {
        background-color: transparent;
        color: #00ff41;
        border: 1px solid #00ff41;
        border-radius: 0;
        text-transform: uppercase;
        font-family: 'Courier New', monospace;
    }
    .stButton>button:hover {
        background-color: #00ff41;
        color: #000000;
        border: 1px solid #00ff41;
    }
    .status-critical { color: #ff3333; font-weight: bold; }
    .status-warning { color: #ffcc00; font-weight: bold; }
    .status-ready { color: #00ff41; font-weight: bold; }
            
    @keyframes textBlink {
    0% { opacity: 1; }
    50% { opacity: 0.2; }
    100% { opacity: 1; }
    }

    .blink-text {
        animation: textBlink 1s infinite;
    }

    
            
    </style>
""", unsafe_allow_html=True)
# File Uploading
uploaded_file = st.file_uploader("Upload Engine Test Data File", type=["csv"])
if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_path = tmp.name

    df = run_backend(temp_path)


    if os.path.exists("outputs/final_output.csv"):

        lstm_df = pd.read_csv("outputs/final_output.csv")

        st.subheader("LSTM Health Prediction")

        st.dataframe(lstm_df)

        import plotly.express as px

        fig = px.bar(
            lstm_df,
            x="unit_number",
            y="Predicted_RUL",
            color="Health_Status",
            title="Engine RUL Prediction",
            color_discrete_map={
                "Safe": "#00ff41",      # green
                "Warning": "#ffdd57",   # yellow
                "Critical": "#ff3333"   # red
            }
        )

        st.plotly_chart(fig)

    st.success("File processed successfully!")

    if os.path.exists("outputs/test_anomaly_results.csv"):
        backend_df = pd.read_csv("outputs/test_anomaly_results.csv")

        st.write("Backend Output Preview")
        st.dataframe(backend_df.head())

else:
    st.warning("⚠ Please upload engine test data to view fleet status.")
#data loading and processing
@st.cache_data
def load_data(uploaded_file):

    if uploaded_file is not None:
        try:
            test_df = pd.read_csv(uploaded_file)

            # Ensure required columns exist
            required_cols = ["unit_number", "time_in_cycles"]

            for col in required_cols:
                if col not in test_df.columns:
                    st.error(f"Missing required column: {col}")
                    return pd.DataFrame()

            # Convert numeric columns safely
            test_df["unit_number"] = pd.to_numeric(test_df["unit_number"], errors="coerce")
            test_df["time_in_cycles"] = pd.to_numeric(test_df["time_in_cycles"], errors="coerce")

            return test_df

        except Exception as e:
            st.error(f"Error loading file: {e}")
            return pd.DataFrame()

    return pd.DataFrame()


test_df = load_data(uploaded_file)
# Process data for dashboard
def get_fleet_status(test_df, lstm_df):
    fleet_data = []

    if test_df.empty or lstm_df.empty:
        return []

    latest_cycles = test_df.groupby('unit_number').last().reset_index()

    # Map LSTM output
    lstm_map = lstm_df.set_index("unit_number").to_dict("index")

    for idx, row in latest_cycles.iterrows():
        unit_id = int(row['unit_number'])
        cycles = int(row['time_in_cycles']) if not pd.isna(row['time_in_cycles']) else 0

        lstm_info = lstm_map.get(unit_id, {})

        # 🔥 Convert LSTM → UI labels
        raw_status = lstm_info.get("Health_Status", "Unknown")

        if raw_status == "Safe":
            status = "READY"
        elif raw_status == "Warning":
            status = "WARNING"
        elif raw_status == "Critical":
            status = "CRITICAL"
        else:
            status = "UNKNOWN"

        rul = lstm_info.get("Predicted_RUL", 0)

        # Better scaling
        health_score = max(0, min(100, int(rul * 0.4)))

        fleet_data.append({
            "id": f"ENG-{unit_id:03d}",
            "real_id": unit_id,
            "type": "Turbofan",
            "status": status,
            "health": health_score,
            "cycles": cycles,
            "rul": rul,  # ✅ NEW (use later if needed)
            "s11": row['sensor_measurement_11'],
            "s4": row['sensor_measurement_4']
        })

    return sorted(fleet_data, key=lambda x: x['health'])

if os.path.exists("outputs/final_output.csv"):
    lstm_df = pd.read_csv("outputs/final_output.csv")
    fleet_data = get_fleet_status(test_df, lstm_df)
else:
    st.error("LSTM output not found")
    st.stop()

if test_df.empty:
    st.stop()

if 'selected_unit' not in st.session_state:
    st.session_state.selected_unit = None
if 'roster_limit' not in st.session_state:
    st.session_state.roster_limit = 10  
# helper functions
def show_detail_view(unit_real_id):
    is_mock = test_df.empty
    current_status = next((item for item in fleet_data if item["real_id"] == unit_real_id), None)

    if current_status:
        with st.container():
            st.markdown(f"### [ENGINE TELEMETRY: {current_status['id']}]")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"""
                <div class="hud-box">
                    <p><b>UNIT ID:</b> {current_status['id']}</p>
                    <p><b>CYCLES:</b> <span style="font-size:24px; color:#fff">{current_status['cycles']}</span></p>
                    <p><b>STATUS:</b> <span class="status-{current_status['status'].lower()}">{current_status['status']}</span></p>
                </div>
                """, unsafe_allow_html=True)
                if current_status['status'] == "CRITICAL":
                    st.markdown("""
                    <div style="
                        border: 2px solid #ff3333;
                        background-color: rgba(40, 0, 0, 0.9);
                        padding: 15px;
                        margin-top: 10px;
                    ">
                        <p class="blink-text" style="color:#ff3333; font-weight:bold; font-size:16px;">
                            ⚠ DIAGNOSTIC ALERT
                        </p>
                        <p style="color:#ff4444;">
                            HPC Degradation Detected.
                        </p>
                        <p style="color:#ff6666;">
                            <b>ACTION:</b> Schedule immediate overhaul.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            with col2:
                if is_mock:
                    cycles = np.arange(0, 100)
                    s11_vals = [540 + random.uniform(-1, 1) for _ in cycles]
                    s4_vals = [1400 + (x * 0.5) for x in cycles]
                else:
                    history = test_df[test_df['unit_number'] == unit_real_id]
                    cycles = history['time_in_cycles']
                    s11_vals = history['sensor_measurement_11']
                    s4_vals = history['sensor_measurement_4']

                st.markdown("#### HPC STATIC PRESSURE (SENSOR 11) HISTORY")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=cycles, y=s11_vals, mode='lines',
                                         line=dict(color='#00ff41', width=2), name='HPC Pressure'))
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,20,0,0.5)',
                    font=dict(color='#00ff41', family="Courier New"),
                    margin=dict(l=0, r=0, t=10, b=0), height=250,
                    xaxis=dict(showgrid=True, gridcolor='#004d1a', title='Cycles'),
                    yaxis=dict(showgrid=True, gridcolor='#004d1a', title='Ps30 (psia)'),
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### LPT OUTLET TEMP (SENSOR 4)")
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=cycles,y=s4_vals,mode='lines',line=dict(color='#ffcc00', width=2),name='LPT Temp'))
                fig2.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,20,0,0.5)',
                    font=dict(color='#00ff41', family="Courier New"),
                    margin=dict(l=0, r=0, t=10, b=0), height=150,
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='#004d1a'),
                )
                st.plotly_chart(fig2, use_container_width=True)

            if st.button("<< RETURN TO FLEET VIEW"):
                st.session_state.selected_unit = None
                st.rerun()
# main dashboard layout
st.markdown("## AI DRIVEN AIRCRAFT MAINTENANCE PREDICTION SYSTEM &nbsp;&nbsp;&nbsp;")

if st.session_state.selected_unit:
    show_detail_view(st.session_state.selected_unit)

else:
    col_left, col_center, col_right = st.columns([0.8, 1, 1.2])
    with col_left:
        total_units = len(fleet_data)
        critical_count = len([x for x in fleet_data if x['status'] == 'CRITICAL'])
        warning_count = len([x for x in fleet_data if x['status'] == 'WARNING'])
        ready_count = len([x for x in fleet_data if x['status'] == 'READY'])
        avg_health = int(sum(item['health'] for item in fleet_data) / total_units) if total_units > 0 else 0

        st.markdown("### FLEET STATUS")
        st.markdown(f"""
        <div class="hud-box">
            <div style="display:flex; justify-content:space-around; text-align:center;">
                <div><div class="stat-number">{ready_count}</div><small>READY</small></div>
                <div><div class="stat-number status-warning">{warning_count}</div><small>WARNING</small></div>
                <div><div class="stat-number status-critical">{critical_count}</div><small>CRITICAL</small></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
   # radar 
    with col_center:
        st.markdown("### ACTIVE MONITORING")
        st.write("")

        critical_units = [u for u in fleet_data if u['status'] == 'CRITICAL']
        warning_units = [u for u in fleet_data if u['status'] == 'WARNING']
        ready_units = [u for u in fleet_data if u['status'] == 'READY']
        display_units = critical_units[:20] + warning_units[:20]+ ready_units[:20] 

        r_values, theta_values, colors, hover_texts, custom_ids = [], [], [], [], []
        for unit in display_units:
            r_values.append(random.uniform(0.2, 0.9))
            theta_values.append(random.uniform(0, 360))
            colors.append('#ff3333' if unit['status'] == 'CRITICAL' else ('#ffcc00' if unit['status'] == 'WARNING' else '#00ff41'))
            hover_texts.append(f"<b>{unit['id']}</b><br>Cycles: {unit['cycles']}<br>Status: {unit['status']}")
            custom_ids.append(unit['real_id'])

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=r_values, theta=theta_values, mode='markers',
            marker=dict(color=colors, size=10, line=dict(color='white', width=1), opacity=0.9),
            hovertext=hover_texts, hoverinfo="text", customdata=custom_ids,
            name="Engines"
        ))

        sweep_width = 25
        fig.add_trace(go.Scatterpolar(
            r=[0, 1.2, 1.2, 0],
            theta=[0, -sweep_width, sweep_width, 0],
            mode='lines',
            fill='toself',
            fillcolor='rgba(0, 255, 65, 0.25)',
            line=dict(color='rgba(0, 255, 65, 0.1)', width=1),
            hoverinfo='skip', showlegend=False
        ))

        frames = []
        for angle in range(0, 360, 4):
            new_thetas = [angle, angle - sweep_width, angle + sweep_width, angle]
            frames.append(go.Frame(
                data=[go.Scatterpolar(r=[0, 1.2, 1.2, 0], theta=new_thetas)],
                traces=[1],
                layout=go.Layout(transition={'duration': 0})
            ))
        fig.frames = frames

        fig.update_layout(
            polar=dict(
                bgcolor='#050505',
                radialaxis=dict(visible=True, range=[0, 1], showticklabels=False,
                                linecolor='#00ff41', gridcolor='#004d1a'),
                angularaxis=dict(visible=True, showticklabels=False,
                                 linecolor='#00ff41', gridcolor='#004d1a', rotation=90),
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            height=320,
            dragmode='select',
            transition={'duration': 0, 'easing': 'linear'},
        )

        event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")

        if event and len(event.selection["points"]) > 0:
            points = event.selection["points"]
            engine_points = [p for p in points if "customdata" in p]
            if engine_points:
                st.session_state.selected_unit = engine_points[0]["customdata"]
                st.rerun()

        st.markdown(f"""
            <div style="text-align: center; font-size: 12px; color: #00ff41;">
                DETECTED ENGINES: {len(display_units)} <br> [SCAN SUCCESSFUL]
            </div>
        """, unsafe_allow_html=True)
    # Engine Roster 
    with col_right:
        st.markdown("### ENGINE ROSTER")
        st.markdown("""
        <div style="display: flex; justify-content: space-between; border-bottom: 2px solid #00ff41; padding: 5px; font-weight: bold;">
            <span style="width: 25%;">UNIT ID</span>
            <span style="width: 25%;">CYCLES</span>
            <span style="width: 25%;">STATUS</span>
            <span style="width: 20%;">ACTION</span>
        </div>
        """, unsafe_allow_html=True)

        for i, ac in enumerate(fleet_data[:st.session_state.roster_limit]):
            c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
            with c1: st.write(f"**{ac['id']}**")
            with c2: st.write(f"{ac['cycles']}")
            with c3:
                st.markdown(
                    f'<span class="status-{ac["status"].lower()}">{ac["status"]}</span>',
                    unsafe_allow_html=True
                )
            with c4:
                if st.button("VIEW", key=f"btn_{i}_{ac['real_id']}"):
                    st.session_state.selected_unit = ac['real_id']
                    st.rerun()
                    st.markdown("<hr style='margin:0;border-color:#003300;'>", unsafe_allow_html=True)

        if st.session_state.roster_limit < len(fleet_data):
            if st.button("SHOW MORE"):
                st.session_state.roster_limit += 10
                st.rerun()