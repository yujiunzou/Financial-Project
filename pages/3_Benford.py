import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import INDUSTRY_PM, benford_analysis

st.set_page_config(page_title="Benford's Law", page_icon="🔢", layout="wide")
st.title("🔢 Benford's Law Analysis")
st.markdown("Detect abnormal digit patterns in financial statements that may signal manipulation.")

# ── What is Benford's Law ─────────────────────────────────────────────────────
with st.expander("📖 What is Benford's Law?", expanded=False):
    st.markdown("""
**Benford's Law** states that in naturally occurring numerical data, the leading digit is more
likely to be small. Specifically, the probability that the first digit is **d** is:

$$P(d) = \\log_{10}\\left(1 + \\frac{1}{d}\\right)$$

| Digit | Expected Frequency |
|-------|-------------------|
| 1 | 30.1% |
| 2 | 17.6% |
| 3 | 12.5% |
| 4 | 9.7% |
| 5 | 7.9% |
| 6 | 6.7% |
| 7 | 5.8% |
| 8 | 5.1% |
| 9 | 4.6% |

**In forensic accounting**, significant deviation from this pattern can signal that numbers
were fabricated or manipulated — because humans tend to choose numbers that *feel* random
but are actually too uniform.

**MAD (Mean Absolute Deviation)** measures how far the observed distribution deviates:
- MAD < 0.006 → Excellent conformity
- 0.006–0.012 → Acceptable conformity
- 0.012–0.015 → Marginal conformity
- MAD > 0.015 → **Non-conformity → Potential manipulation signal**
    """)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Input")
    ticker_input = st.text_input("Stock Ticker", value="AAPL").upper().strip()
    run_btn = st.button("🚀 Run Analysis", use_container_width=True)
    st.divider()
    st.caption("Try comparing a clean company vs a known fraud case.")

if not run_btn:
    st.info("👈 Enter a ticker and click **Run Analysis**.")
    st.stop()

with st.spinner(f"Running Benford's Law analysis for **{ticker_input}**…"):
    try:
        benford_df, mad = benford_analysis(ticker_input)
    except Exception as e:
        st.error(str(e)); st.stop()

# ── MAD result ────────────────────────────────────────────────────────────────
st.subheader(f"Results for: {ticker_input}")
c1, c2, c3 = st.columns(3)
c1.metric("Mean Absolute Deviation (MAD)", f"{mad:.4f}")
if mad < 0.006:
    c2.metric("Conformity Level", "Excellent ✅")
    c3.metric("Benford Signal", "No anomaly detected ✅")
elif mad < 0.012:
    c2.metric("Conformity Level", "Acceptable ✅")
    c3.metric("Benford Signal", "Minor deviation ⚠️")
elif mad < 0.015:
    c2.metric("Conformity Level", "Marginal ⚠️")
    c3.metric("Benford Signal", "Moderate deviation ⚠️")
else:
    c2.metric("Conformity Level", "Non-conforming 🚩")
    c3.metric("Benford Signal", "Potential manipulation signal 🚩")

if mad > 0.015:
    st.error("🚩 **Benford's Law flags this company.** The leading digit distribution deviates significantly from the expected pattern.")
elif mad > 0.012:
    st.warning("⚠️ **Marginal conformity.** The distribution shows some deviation worth monitoring.")
else:
    st.success("✅ **Good Benford conformity.** The digit distribution is consistent with natural financial data.")

st.divider()

# ── Main Benford chart ────────────────────────────────────────────────────────
st.subheader("📊 Observed vs Expected Leading Digit Distribution")

fig = go.Figure()
fig.add_trace(go.Bar(
    x=benford_df["Digit"],
    y=benford_df["Observed (%)"],
    name="Observed",
    marker_color="#3498db",
    opacity=0.8,
))
fig.add_trace(go.Scatter(
    x=benford_df["Digit"],
    y=benford_df["Expected (%)"],
    name="Benford Expected",
    mode="lines+markers",
    line=dict(color="#e74c3c", width=2.5, dash="dash"),
    marker=dict(size=7),
))
fig.update_layout(
    xaxis=dict(title="Leading Digit", tickmode="linear", tick0=1, dtick=1),
    yaxis=dict(title="Frequency (%)"),
    height=400,
    margin=dict(t=20,b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    bargap=0.2,
)
st.plotly_chart(fig, use_container_width=True)

# ── Deviation bar chart ───────────────────────────────────────────────────────
st.subheader("📉 Deviation from Expected (Observed − Expected)")

benford_df["Deviation (%)"] = (benford_df["Observed (%)"] - benford_df["Expected (%)"]).round(2)
colors = ["#e74c3c" if abs(d) > 2 else "#f39c12" if abs(d) > 1 else "#2ecc71"
          for d in benford_df["Deviation (%)"]]

fig_dev = go.Figure(go.Bar(
    x=benford_df["Digit"],
    y=benford_df["Deviation (%)"],
    marker_color=colors,
    text=benford_df["Deviation (%)"].apply(lambda x: f"{x:+.2f}%"),
    textposition="outside",
))
fig_dev.add_hline(y=0, line_color="black", line_width=1)
fig_dev.update_layout(
    xaxis=dict(title="Leading Digit", tickmode="linear", tick0=1, dtick=1),
    yaxis=dict(title="Deviation (%)"),
    height=320,
    margin=dict(t=20,b=20),
)
st.plotly_chart(fig_dev, use_container_width=True)
st.caption("Red = deviation > 2% · Orange = deviation > 1% · Green = within 1%")

st.divider()

# ── Detail table ──────────────────────────────────────────────────────────────
st.subheader("📋 Digit Distribution Table")
st.dataframe(benford_df, use_container_width=True, hide_index=True)

st.caption("Data: Yahoo Finance · For educational purposes only.")
