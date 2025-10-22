"""
Rolls-Royce Digital Twin — Streamlit App (single-file)
Author: Generated for Chandrika Joshi
Description:
 - Advanced, futuristic-themed Streamlit dashboard that simulates:
    - RUL (Remaining Useful Life) predictions using synthetic / placeholder model
    - Real-time sensor visualizations
    - LangChain-like autonomous decision agent (rule-based stub)
    - GPT-based maintenance report generation (OpenAI -- optional; user provides key)
 - Run: 
    pip install -r requirements.txt
    streamlit run Rolls-Royce_DigitalTwin_Streamlit_App.py

Requirements (example):
streamlit
pandas
numpy
scikit-learn
altair
plotly
openai  # optional for GPT report
pillow
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
import base64
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Rolls-Royce — Digital Twin", layout="wide", initial_sidebar_state="expanded")

# --- Helper: inject CSS for futuristic theme ---
st.markdown("""
<style>
:root{
  --accent:#00ffd5;
  --accent2:#7f5af0;
  --bg:#05060a;
  --card:#0b0d12;
  --muted:#9aa3b2;
}
[data-testid="stAppViewContainer"] {background: linear-gradient(180deg, rgba(5,6,10,1) 0%, rgba(8,10,18,1) 60%);}
[data-testid="stHeader"]{background:transparent}
section.main > div {padding-top: 0rem}
.css-1v3fvcr {padding-top: 0rem}

.header {
  padding: 18px 24px;
  border-radius: 12px;
  background: linear-gradient(90deg, rgba(127,90,240,0.12), rgba(0,255,213,0.06));
  box-shadow: 0 6px 24px rgba(0,0,0,0.5);
  backdrop-filter: blur(6px);
}
.h1 {font-size:34px; font-weight:700; color: white; margin: 0}
.h2 {font-size:14px; color: var(--muted); margin: 0}
.card {background: rgba(255,255,255,0.02); padding: 18px; border-radius: 14px;}
.small-muted{color:var(--muted); font-size:13px}
.badge {background: linear-gradient(90deg,var(--accent),var(--accent2)); padding:6px 10px; color:#071226; border-radius:999px; font-weight:700}
</style>

<div class="header">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div>
      <div class="h1">Rolls‑Royce Digital Twin</div>
      <div class="h2">Predictive maintenance · Agentic AI · Real-time dashboards</div>
    </div>
    <div style="text-align:right">
      <div class="badge">IntelligentEngine Demo</div>
      <div style="height:6px"></div>
      <div class="small-muted">Project by: Chandrika Joshi</div>
    </div>
  </div>
</div>

""", unsafe_allow_html=True)

# --- Sidebar controls ---
st.sidebar.title("Controls & Simulation")
use_real_model = st.sidebar.checkbox("Use demo RUL model (fast)", value=True)
simulate = st.sidebar.checkbox("Autorefresh / simulate live stream", value=True)
refresh_rate = st.sidebar.slider("Simulation refresh (seconds)", 1, 10, 3)
seed = st.sidebar.number_input("Random seed", 0, 9999, 42)
np.random.seed(seed)

st.sidebar.markdown("---")
openai_key = st.sidebar.text_input("OpenAI API key (optional)", type="password")
st.sidebar.markdown("---")
st.sidebar.markdown("**Export**: Download current data or report")

# --- Top row: Overview cards ---
col1, col2, col3, col4 = st.columns([2,1,1,1])

with col1:
    st.markdown("""
    <div class='card'>
      <h3 style='color:white;margin:0'>Fleet Health Overview</h3>
      <p class='small-muted'>At-a-glance status of simulated engines and predicted failures</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.metric("Avg RUL (cycles)", "—", delta=None, label_visibility='visible')

with col3:
    st.metric("Engines at-risk", "—", delta=None)

with col4:
    st.metric("Anomalies detected", "—", delta=None)

st.markdown("---")

# --- Simulate sensor / CMAPSS-like data ---
@st.cache_data
def generate_synthetic_cmapss(n_engines=12, seq_len=200):
    # Each engine has multiple sensors; this is simplified/synthetic
    engines = []
    for eid in range(n_engines):
        baseline = np.random.uniform(0.0, 1.0, size=(6,))
        life = np.linspace(1.0, 0.0, seq_len)
        for t in range(seq_len):
            sensors = baseline + np.random.normal(0, 0.02, size=(6,)) + (1 - life[t]) * np.random.uniform(0, 0.5, size=(6,))
            row = {
                'engine_id': f'E{eid:03d}',
                'cycle': t+1,
                'RUL_true': seq_len - t,
            }
            for i, s in enumerate(sensors, start=1):
                row[f'sensor_{i}'] = float(s)
            engines.append(row)
    return pd.DataFrame(engines)

# create dataset
DATA = generate_synthetic_cmapss(n_engines=8, seq_len=220)

# --- Build a fast demo RUL model (trained on synthetic env) ---
@st.cache_data
def train_demo_rul_model(df):
    # construct simple features: last sensor values and cycle
    X = []
    y = []
    grouped = df.groupby('engine_id')
    for name, group in grouped:
        group = group.sort_values('cycle')
        # sample random steps to train regression
        sample_idx = np.random.choice(group.index, size=int(len(group)*0.6), replace=False)
        for idx in sample_idx:
            row = group.loc[idx]
            features = [row[f'sensor_{i}'] for i in range(1,7)] + [row['cycle']]
            X.append(features)
            y.append(row['RUL_true'])
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    model = RandomForestRegressor(n_estimators=100, random_state=1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return model, rmse

model, model_rmse = train_demo_rul_model(DATA)

# --- RUL prediction function ---

def predict_rul_for_engine(model, latest_row):
    features = np.array([[latest_row[f'sensor_{i}'] for i in range(1,7)] + [latest_row['cycle']]])
    pred = model.predict(features)[0]
    return max(0.0, float(pred))

# --- Simple agentic decision maker (LangChain-style stub) ---

def agentic_decision(engine_row, rul_pred):
    # rule-based decisions for demo
    actions = []
    if rul_pred < 30:
        actions.append('Schedule immediate maintenance — high priority')
    elif rul_pred < 80:
        actions.append('Plan maintenance within next 30 cycles')
    else:
        actions.append('Continue monitoring — low priority')

    # detect sensor anomalies (very simple)
    sensors = [engine_row[f'sensor_{i}'] for i in range(1,7)]
    if any(s > 1.2 for s in sensors):
        actions.append('Sensor anomaly: inspect sensor cluster 1-3')
    if np.std(sensors) > 0.18:
        actions.append('High variance in sensors — check for intermittent faults')

    # add confidence
    confidence = 0.9 if rul_pred > 50 else 0.7
    return actions, confidence

# --- UI: Fleet selector & live engine card ---
colA, colB = st.columns([2,3])
with colA:
    st.subheader("Fleet — Engines")
    engines = sorted(DATA['engine_id'].unique())
    selected_engine = st.selectbox("Select engine", engines)
    engine_df = DATA[DATA['engine_id']==selected_engine].sort_values('cycle')
    latest = engine_df.iloc[-1]
    st.markdown(f"**Engine ID:** {selected_engine}  ")
    st.markdown(f"**Current cycle:** {int(latest['cycle'])}")
    st.markdown(f"**True RUL (sim):** {int(latest['RUL_true'])} cycles")

    # small futuristic image
    st.markdown("<div style='text-align:center'><img src='https://images.unsplash.com/photo-1502877338535-766e1452684a?w=1200&q=80' width=220 style='border-radius:8px;box-shadow:0 8px 28px rgba(0,0,0,0.6)'></div>", unsafe_allow_html=True)

with colB:
    st.subheader("Predicted RUL & Agent Actions")
    if use_real_model:
        rul_pred = predict_rul_for_engine(model, latest)
    else:
        rul_pred = float(max(0.0, latest['RUL_true'] + np.random.normal(0, 10)))
    st.metric(label="Predicted RUL (cycles)", value=f"{rul_pred:.1f}")
    actions, confidence = agentic_decision(latest, rul_pred)
    st.markdown("**Agent decisions:**")
    for a in actions:
        st.info(a)
    st.markdown(f"**Agent confidence:** {confidence*100:.0f}%")

# --- Charts: sensor timeseries & RUL forecast ---
st.markdown('---')

left, right = st.columns([2,1])
with left:
    st.subheader('Sensor Time-series (last 60 cycles)')
    last60 = engine_df.tail(60)
    sensor_cols = [f'sensor_{i}' for i in range(1,7)]
    df_melt = last60.melt(id_vars=['cycle'], value_vars=sensor_cols, var_name='sensor', value_name='value')
    chart = alt.Chart(df_melt).mark_line().encode(x='cycle', y='value', color='sensor').interactive().properties(height=320)
    st.altair_chart(chart, use_container_width=True)

with right:
    st.subheader('RUL Forecast (toy)')
    # toy forecast line: predicted declines linearly from current pred to 0
    cycles = np.arange(int(latest['cycle']), int(latest['cycle']) + int(rul_pred) + 1)
    forecast_rul = np.linspace(rul_pred, 0, len(cycles))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cycles, y=forecast_rul, mode='lines+markers', name='Forecast RUL'))
    fig.add_trace(go.Scatter(x=[latest['cycle']], y=[rul_pred], mode='markers', name='Now'))
    fig.update_layout(height=320, xaxis_title='Cycle', yaxis_title='Remaining Life (cycles)', template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

# --- Fleet heatmap / status ---
st.markdown('---')
st.subheader('Fleet Status Grid')

fleet_summary = []
for eid in engines:
    last = DATA[DATA['engine_id']==eid].sort_values('cycle').iloc[-1]
    pred = predict_rul_for_engine(model, last)
    risk = 'LOW'
    if pred < 30:
        risk='CRITICAL'
    elif pred < 80:
        risk='MEDIUM'
    fleet_summary.append({'engine_id':eid, 'pred_rul':pred, 'risk':risk})
fleet_df = pd.DataFrame(fleet_summary)

# color-coded table
def color_risk(row):
    color = 'green' if row['risk']=='LOW' else ('orange' if row['risk']=='MEDIUM' else 'red')
    return f"<div style='padding:8px;border-radius:8px;background:{color};color:#071226;font-weight:700;text-align:center'>{row['risk']}</div>"

cols = st.columns(len(engines))
for i, eid in enumerate(engines):
    with cols[i]:
        row = fleet_df[fleet_df['engine_id']==eid].iloc[0]
        st.markdown(f"<div class='card' style='text-align:center'><h4 style='margin:6px'>{eid}</h4><p style='margin:0'>RUL: {row['pred_rul']:.0f}</p>{color_risk(row)}</div>", unsafe_allow_html=True)

# --- Generate GPT maintenance report (optional) ---
st.markdown('---')
st.subheader('Generate Maintenance Report (GPT)')
report_engine = st.selectbox('Choose engine for report', engines, key='report_engine')
report_level = st.selectbox('Report detail level', ['Summary','Detailed','Investigation-ready'])
if st.button('Generate report'):
    # prepare prompt
    last = DATA[DATA['engine_id']==report_engine].sort_values('cycle').iloc[-1]
    pred = predict_rul_for_engine(model, last)
    actions, confidence = agentic_decision(last, pred)
    prompt = (
        f"You are a maintenance engineer assistant. Provide a {report_level} maintenance report\n\n"
        f"Engine ID: {report_engine}\nCycle: {int(last['cycle'])}\nPredicted RUL: {pred:.1f} cycles\n"
        f"Sensor readings: " + ", ".join([f"s{i}={last[f'sensor_{i}']:.3f}" for i in range(1,7)]) + "\n"
        f"Agent decisions: " + "; ".join(actions) + "\n"
        f"Confidence: {confidence:.2f}"
    )

    if openai_key:
        import openai
        openai.api_key = openai_key
        try:
            res = openai.ChatCompletion.create(
                model='gpt-4o-mini',
                messages=[{'role':'system','content':'You are a helpful maintenance engineer.'},{'role':'user','content':prompt}],
                temperature=0.2,
                max_tokens=500
            )
            report_text = res['choices'][0]['message']['content']
        except Exception as e:
            report_text = f"[Failed to call OpenAI: {e}]\n\nPrompt used:\n" + prompt
    else:
        # fallback: small templated report
        report_text = f"Maintenance Report (auto-generated)\nEngine: {report_engine}\nCycle: {int(last['cycle'])}\nPredicted RUL: {pred:.1f}\nDecisions:\n"
        for a in actions:
            report_text += f" - {a}\n"
        report_text += "\nNotes:\n - This is a demo report. Connect your OpenAI API key in the sidebar for a full GPT-generated narrative."

    st.code(report_text, language='text')
    # offer download
    b = report_text.encode('utf-8')
    st.download_button('Download report (.txt)', data=b, file_name=f'{report_engine}_report.txt')

# --- Export current dataset ---
csv = DATA.to_csv(index=False).encode('utf-8')
st.download_button('Download simulated dataset (CSV)', csv, 'digital_twin_data.csv', 'text/csv')

# --- Auto-simulate refresh ---
if simulate:
    st.rerun()

# --- Footer credits ---
st.markdown("""
<div style='padding:18px;margin-top:18px;border-radius:12px;background:transparent;color:#9aa3b2'>
Built with ❤️ — Inspired by Rolls‑Royce IntelligentEngine. Project by Chandrika Joshi. 
</div>
""", unsafe_allow_html=True)
