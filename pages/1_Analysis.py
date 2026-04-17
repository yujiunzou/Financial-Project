import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import (FEATURES, FEATURE_LABELS, INDUSTRY_PM, RED_FLAGS,
                   compute_features_all_years, compute_beneish)

st.set_page_config(page_title="Stock Analysis", page_icon="🔎", layout="wide")
st.title("🔎 Single Stock Analysis")
st.markdown("Enter a stock ticker to get a complete fraud risk breakdown.")

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    rf  = joblib.load("fraud_model.pkl")
    try:
        lr = joblib.load("logistic_model.pkl")
    except Exception:
        lr = None
    return rf, lr

try:
    rf_model, lr_model = load_models()
except Exception:
    st.error("⚠️ `fraud_model.pkl` not found. Place it in the root folder.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Input")
    ticker_input = st.text_input("Stock Ticker", value="AAPL").upper().strip()
    industry_sel = st.selectbox("Industry", list(INDUSTRY_PM.keys()))
    run_btn = st.button("🚀 Analyze", use_container_width=True)
    st.divider()
    st.caption("Try known fraud cases: ENRN (Enron), WCOM (WorldCom)")

if not run_btn:
    st.info("👈 Enter a ticker and click **Analyze**.")
    st.stop()

industry_pm = INDUSTRY_PM[industry_sel]
industry_avg = {f: 0.0 for f in FEATURES}
industry_avg.update({
    'roa': 0.08, 'profit_margin': industry_pm,
    'current_ratio': 1.5, 'debt_ratio': 0.5,
    'asset_turnover': 0.7, 'ocf_ratio': 0.1,
})

with st.spinner(f"Fetching data for **{ticker_input}**…"):
    try:
        feat_all = compute_features_all_years(ticker_input, industry_pm)
        m_score, beneish_components = compute_beneish(ticker_input)
    except Exception as e:
        st.error(str(e)); st.stop()

feat_latest = feat_all.iloc[-1][FEATURES].fillna(0)
feat_df = pd.DataFrame([feat_latest])

# ── Predictions ───────────────────────────────────────────────────────────────
rf_prob  = rf_model.predict_proba(feat_df)[0][1]
rf_label = rf_model.predict(feat_df)[0]
lr_prob  = lr_model.predict_proba(feat_df)[0][1] if lr_model else None
beneish_flag = int(m_score > -1.78)

# Ensemble score
scores = [rf_prob, beneish_flag]
if lr_prob is not None:
    scores.append(lr_prob)
ensemble_score = float(np.mean(scores))

# ── Top summary ───────────────────────────────────────────────────────────────
st.subheader(f"Results for: {ticker_input}")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Beneish M-Score", f"{m_score:.2f}", delta="⚠️ Risk" if beneish_flag else "✅ Safe",
          delta_color="inverse")
c2.metric("RF Fraud Probability", f"{rf_prob:.1%}")
if lr_prob is not None:
    c3.metric("LR Fraud Probability", f"{lr_prob:.1%}")
else:
    c3.metric("LR Fraud Probability", "N/A")
c4.metric("Ensemble Score", f"{ensemble_score:.1%}",
          delta="⚠️ HIGH RISK" if ensemble_score > 0.5 else "✅ LOW RISK",
          delta_color="inverse")

if ensemble_score > 0.5:
    st.error("🚨 **High manipulation risk.** Multiple indicators flag this company as potentially engaging in earnings manipulation.")
else:
    st.success("✅ **Low manipulation risk.** Financial ratios and fraud indicators appear within normal ranges.")

st.divider()

# ── Gauge + Beneish bar ───────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Ensemble Fraud Risk Gauge**")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=ensemble_score*100,
        title={"text": "Risk Score (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "#e74c3c" if ensemble_score > 0.5 else "#2ecc71"},
            "steps": [{"range":[0,33],"color":"#d5f5e3"},
                      {"range":[33,66],"color":"#fef9e7"},
                      {"range":[66,100],"color":"#fadbd8"}],
            "threshold": {"line":{"color":"red","width":4},"value":50}
        }
    ))
    fig_gauge.update_layout(height=300, margin=dict(t=30,b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col2:
    st.markdown("**Beneish M-Score Components**")
    beneish_df = pd.DataFrame({
        "Component": list(beneish_components.keys()),
        "Value": [round(v,3) if not np.isnan(v) else 0 for v in beneish_components.values()],
    })
    colors_b = ["#e74c3c" if v > 1 else "#2ecc71" for v in beneish_df["Value"]]
    fig_b = go.Figure(go.Bar(
        x=beneish_df["Component"], y=beneish_df["Value"],
        marker_color=colors_b, text=beneish_df["Value"].round(3),
        textposition="outside"
    ))
    fig_b.add_hline(y=1, line_dash="dash", line_color="red",
                    annotation_text="Benchmark = 1")
    fig_b.update_layout(height=300, margin=dict(t=30,b=10),
                        yaxis_title="Index Value")
    st.plotly_chart(fig_b, use_container_width=True)

st.divider()

# ── Fraud Risk Heatmap ────────────────────────────────────────────────────────
st.subheader("🌡️ Fraud Risk Heatmap")
st.caption("Red = potential red flag · Green = within normal range")

heatmap_data = []
for feat in FEATURES:
    val = float(feat_latest[feat]) if not pd.isna(feat_latest[feat]) else 0
    is_flag = feat in RED_FLAGS and not pd.isna(feat_latest[feat]) and RED_FLAGS[feat](val)
    heatmap_data.append({
        "Feature": FEATURE_LABELS[feat],
        "Value": round(val, 4),
        "Risk": 1 if is_flag else 0,
        "Status": "🚩 Flag" if is_flag else "✅ Normal"
    })

hm_df = pd.DataFrame(heatmap_data)
n_flags = hm_df["Risk"].sum()

# Visual heatmap grid
hm_values = hm_df["Risk"].values.reshape(3, 5)
hm_labels = [[FEATURE_LABELS[FEATURES[i*5+j]] for j in range(5)] for i in range(3)]

fig_hm = go.Figure(go.Heatmap(
    z=hm_values,
    text=hm_labels,
    texttemplate="%{text}",
    colorscale=[[0,"#d5f5e3"],[1,"#e74c3c"]],
    showscale=False,
    xgap=3, ygap=3,
))
fig_hm.update_layout(
    height=220,
    margin=dict(t=10,b=10,l=10,r=10),
    xaxis=dict(showticklabels=False),
    yaxis=dict(showticklabels=False),
)
st.plotly_chart(fig_hm, use_container_width=True)
st.markdown(f"**{n_flags} out of {len(FEATURES)} indicators flagged as potential red flags.**")

st.divider()

# ── Peer comparison ───────────────────────────────────────────────────────────
st.subheader("🏢 Peer Comparison vs Industry Average")
peer_features = ['roa','profit_margin','current_ratio','debt_ratio',
                 'asset_turnover','accrual_ratio']

comp_df = pd.DataFrame({
    "Feature": [FEATURE_LABELS[f] for f in peer_features],
    f"{ticker_input}": [round(float(feat_latest[f]),4) if not pd.isna(feat_latest[f]) else 0
                        for f in peer_features],
    "Industry Avg": [round(industry_avg[f],4) for f in peer_features],
})

fig_peer = go.Figure()
fig_peer.add_trace(go.Bar(name=ticker_input, x=comp_df["Feature"],
                          y=comp_df[ticker_input], marker_color="#3498db"))
fig_peer.add_trace(go.Bar(name="Industry Avg", x=comp_df["Feature"],
                          y=comp_df["Industry Avg"], marker_color="#95a5a6"))
fig_peer.update_layout(barmode="group", height=360,
                       margin=dict(t=20,b=10), yaxis_title="Ratio Value")
st.plotly_chart(fig_peer, use_container_width=True)

st.divider()

# ── Feature importance ────────────────────────────────────────────────────────
st.subheader("📊 Feature Importance (Random Forest)")
importance = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values()
importance.index = [FEATURE_LABELS[f] for f in importance.index]

fig_imp = px.bar(importance, orientation='h',
                 color=importance, color_continuous_scale=["#2ecc71","#e74c3c"],
                 labels={"value":"Importance","index":"Feature"})
fig_imp.update_layout(height=420, coloraxis_showscale=False, margin=dict(t=10,b=10))
st.plotly_chart(fig_imp, use_container_width=True)

st.divider()

# ── Ratio detail table ────────────────────────────────────────────────────────
st.subheader("📋 Full Financial Ratio Breakdown")
table_rows = []
for feat in FEATURES:
    val = feat_latest[feat]
    flag = feat in RED_FLAGS and not pd.isna(val) and RED_FLAGS[feat](float(val))
    table_rows.append({
        "Indicator": FEATURE_LABELS[feat],
        "Value": round(float(val),4) if not pd.isna(val) else "N/A",
        "Status": "🚩 Flag" if flag else "✅ Normal",
    })
st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

st.caption("Data: Yahoo Finance · For educational purposes only.")
