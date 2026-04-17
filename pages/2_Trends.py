import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import FEATURES, FEATURE_LABELS, INDUSTRY_PM, RED_FLAGS, compute_features_all_years

st.set_page_config(page_title="Financial Trends", page_icon="📈", layout="wide")
st.title("📈 Financial Trends Over Time")
st.markdown("Visualize multi-year financial metrics to spot unusual jumps or deterioration.")

SECTOR_MAP = {
    "Technology": "Technology", "Healthcare": "Healthcare",
    "Financial Services": "Finance", "Consumer Defensive": "Consumer Goods",
    "Consumer Cyclical": "Consumer Goods", "Energy": "Energy",
    "Industrials": "Industrials", "Utilities": "Utilities",
    "Real Estate": "Real Estate",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Input")
    ticker_input = st.text_input("Stock Ticker", value="AAPL").upper().strip()
    selected_features = st.multiselect(
        "Select metrics to display",
        options=list(FEATURE_LABELS.values()),
        default=["Revenue Growth", "Accrual Ratio", "Profit Margin",
                 "CFO / Net Income", "Debt Ratio"]
    )
    run_btn = st.button("🚀 Load Trends", use_container_width=True)

if not run_btn:
    st.info("👈 Enter a ticker and click **Load Trends**."); st.stop()

# ── Auto-detect industry ──────────────────────────────────────────────────────
with st.spinner(f"Fetching data for **{ticker_input}**…"):
    try:
        info         = yf.Ticker(ticker_input).info
        sector       = info.get("sector", "Other")
        industry_sel = SECTOR_MAP.get(sector, "Other")
        industry_pm  = INDUSTRY_PM[industry_sel]
        feat_all     = compute_features_all_years(ticker_input, industry_pm)
    except Exception as e:
        st.error(str(e)); st.stop()

st.caption(f"🏭 Auto-detected industry: **{industry_sel}** (sector: {sector})")

feat_all.index = pd.to_datetime(feat_all.index).strftime("%Y")
inv_labels     = {v: k for k, v in FEATURE_LABELS.items()}
selected_keys  = [inv_labels[l] for l in selected_features if l in inv_labels]

st.subheader(f"📅 {ticker_input} — {feat_all.index[0]} to {feat_all.index[-1]}")

if not selected_keys:
    st.warning("Please select at least one metric in the sidebar."); st.stop()

# ── Line charts ───────────────────────────────────────────────────────────────
pairs = [selected_keys[i:i+2] for i in range(0, len(selected_keys), 2)]
for pair in pairs:
    cols = st.columns(len(pair))
    for col, feat in zip(cols, pair):
        with col:
            series = feat_all[feat].dropna()
            if series.empty:
                st.warning(f"No data for {FEATURE_LABELS[feat]}"); continue

            flag_mask = [feat in RED_FLAGS and not np.isnan(v) and RED_FLAGS[feat](v)
                         for v in series]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=series.index, y=series.values,
                mode="lines+markers",
                line=dict(color="#3498db", width=2),
                marker=dict(size=8,
                            color=["#e74c3c" if f else "#3498db" for f in flag_mask],
                            line=dict(width=1, color="white")),
                name=FEATURE_LABELS[feat]
            ))
            # threshold reference lines
            thresholds = {'accrual_ratio': (0.05, "red", "Red Flag"),
                          'debt_ratio':    (0.80, "red", "Red Flag"),
                          'cfo_to_income': (0.50, "orange", "Warning")}
            if feat in thresholds:
                y_val, color, label = thresholds[feat]
                fig.add_hline(y=y_val, line_dash="dot", line_color=color,
                              annotation_text=label)

            fig.update_layout(title=FEATURE_LABELS[feat], height=280,
                              margin=dict(t=40,b=20,l=20,r=20),
                              xaxis_title="Fiscal Year", yaxis_title="Value",
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Combined normalized chart ─────────────────────────────────────────────────
st.subheader("📊 All Selected Metrics — Combined View (Normalized)")
colors = ["#3498db","#e74c3c","#2ecc71","#f39c12","#9b59b6","#1abc9c"]
fig_combined = go.Figure()
for i, feat in enumerate(selected_keys):
    series = feat_all[feat].dropna()
    if series.empty: continue
    mn, mx = series.min(), series.max()
    norm   = (series - mn) / (mx - mn) if mx != mn else series * 0
    fig_combined.add_trace(go.Scatter(
        x=series.index, y=norm.values, mode="lines+markers",
        name=FEATURE_LABELS[feat],
        line=dict(color=colors[i % len(colors)], width=2)
    ))
fig_combined.update_layout(height=380, margin=dict(t=20,b=20),
                            xaxis_title="Fiscal Year",
                            yaxis_title="Normalized Value (0–1)",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig_combined, use_container_width=True)
st.caption("Values normalized to [0–1] within each metric for comparable scale.")

st.divider()

# ── YoY change table ──────────────────────────────────────────────────────────
st.subheader("📋 Year-over-Year Change")
yoy_df  = feat_all[selected_keys].rename(columns=FEATURE_LABELS)
yoy_pct = yoy_df.pct_change().round(4) * 100
yoy_pct.index.name = "Fiscal Year"
st.dataframe(
    yoy_pct.style.format("{:.1f}%", na_rep="N/A")
    .background_gradient(cmap="RdYlGn", axis=None),
    use_container_width=True
)
st.caption("Green = improvement · Red = deterioration · Data: Yahoo Finance")
