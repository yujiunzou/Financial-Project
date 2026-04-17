import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import FEATURES, FEATURE_LABELS, INDUSTRY_PM, RED_FLAGS, compute_features_all_years

st.set_page_config(page_title="Financial Trends", page_icon="📈", layout="wide")
st.title("📈 Financial Trends Over Time")
st.markdown("Visualize multi-year financial metrics to spot unusual jumps or deterioration.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Input")
    ticker_input = st.text_input("Stock Ticker", value="AAPL").upper().strip()
    industry_sel = st.selectbox("Industry", list(INDUSTRY_PM.keys()))
    selected_features = st.multiselect(
        "Select metrics to display",
        options=list(FEATURE_LABELS.values()),
        default=["Revenue Growth", "Accrual Ratio", "Profit Margin",
                 "CFO / Net Income", "Debt Ratio"]
    )
    run_btn = st.button("🚀 Load Trends", use_container_width=True)

if not run_btn:
    st.info("👈 Enter a ticker and click **Load Trends**.")
    st.stop()

industry_pm = INDUSTRY_PM[industry_sel]
inv_labels = {v: k for k, v in FEATURE_LABELS.items()}
selected_keys = [inv_labels[l] for l in selected_features if l in inv_labels]

with st.spinner(f"Fetching multi-year data for **{ticker_input}**…"):
    try:
        feat_all = compute_features_all_years(ticker_input, industry_pm)
    except Exception as e:
        st.error(str(e)); st.stop()

feat_all.index = pd.to_datetime(feat_all.index)
feat_all.index = feat_all.index.strftime("%Y")

st.subheader(f"📅 {ticker_input} — {feat_all.index[0]} to {feat_all.index[-1]}")

# ── Line charts for each selected metric ─────────────────────────────────────
if not selected_keys:
    st.warning("Please select at least one metric in the sidebar.")
    st.stop()

# Show 2 charts per row
pairs = [selected_keys[i:i+2] for i in range(0, len(selected_keys), 2)]

for pair in pairs:
    cols = st.columns(len(pair))
    for col, feat in zip(cols, pair):
        with col:
            series = feat_all[feat].dropna()
            if series.empty:
                st.warning(f"No data for {FEATURE_LABELS[feat]}")
                continue

            # Flag points that exceed red-flag threshold
            flag_mask = []
            for val in series:
                is_flag = feat in RED_FLAGS and not np.isnan(val) and RED_FLAGS[feat](val)
                flag_mask.append(is_flag)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=series.index, y=series.values,
                mode="lines+markers",
                line=dict(color="#3498db", width=2),
                marker=dict(
                    size=8,
                    color=["#e74c3c" if f else "#3498db" for f in flag_mask],
                    line=dict(width=1, color="white")
                ),
                name=FEATURE_LABELS[feat]
            ))

            # Shade red-flag zone if threshold exists
            if feat == 'accrual_ratio':
                fig.add_hline(y=0.05, line_dash="dot", line_color="red",
                              annotation_text="Red Flag Threshold")
            elif feat == 'debt_ratio':
                fig.add_hline(y=0.80, line_dash="dot", line_color="red",
                              annotation_text="Red Flag Threshold")
            elif feat == 'cfo_to_income':
                fig.add_hline(y=0.50, line_dash="dot", line_color="orange",
                              annotation_text="Warning Threshold")

            fig.update_layout(
                title=FEATURE_LABELS[feat],
                height=280,
                margin=dict(t=40,b=20,l=20,r=20),
                xaxis_title="Fiscal Year",
                yaxis_title="Value",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Combined trend chart ──────────────────────────────────────────────────────
st.subheader("📊 All Selected Metrics — Combined View")

fig_combined = go.Figure()
colors = ["#3498db","#e74c3c","#2ecc71","#f39c12","#9b59b6","#1abc9c"]
for i, feat in enumerate(selected_keys):
    series = feat_all[feat].dropna()
    if series.empty: continue
    # Normalize to [0,1] for comparable scale
    mn, mx = series.min(), series.max()
    norm = (series - mn) / (mx - mn) if mx != mn else series * 0
    fig_combined.add_trace(go.Scatter(
        x=series.index, y=norm.values,
        mode="lines+markers", name=FEATURE_LABELS[feat],
        line=dict(color=colors[i % len(colors)], width=2)
    ))
fig_combined.update_layout(
    height=380,
    margin=dict(t=20,b=20),
    xaxis_title="Fiscal Year",
    yaxis_title="Normalized Value (0–1)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02)
)
st.plotly_chart(fig_combined, use_container_width=True)
st.caption("Values normalized to [0–1] within each metric for comparable scale.")

st.divider()

# ── Year-over-year change table ───────────────────────────────────────────────
st.subheader("📋 Year-over-Year Change")

yoy_df = feat_all[selected_keys].rename(columns=FEATURE_LABELS)
yoy_pct = yoy_df.pct_change().round(4) * 100
yoy_pct.index.name = "Fiscal Year"

st.dataframe(
    yoy_pct.style.format("{:.1f}%", na_rep="N/A")
    .background_gradient(cmap="RdYlGn", axis=None),
    use_container_width=True
)
st.caption("Green = improvement · Red = deterioration")

st.caption("Data: Yahoo Finance · For educational purposes only.")
