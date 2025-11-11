# app.py
# Inventory Health & Planning — Streamlit app (tolerance, financials, and visuals updated)
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import re
from math import sqrt

st.set_page_config(page_title="InventoryApp", layout="wide")

# -------------------- Styling --------------------
CUSTOM_CSS = """
<style>
[data-testid="stAppViewContainer"] > .main {
    max-width: 1200px;
    margin: 0 auto;
    padding-top: 12px;
}
[data-testid="stSidebar"] { min-width: 240px; max-width: 280px; }

.kpi-card {
  border-radius: 10px;
  padding: 16px;
  color: #111827;
  display:flex;
  flex-direction:column;
  align-items:flex-start;
  justify-content:center;
  min-height:84px;
  box-shadow: 0 1px 2px rgba(16,24,40,0.05);
  border: 1px solid rgba(16,24,40,0.06);
  margin: 6px 0;
  background:#fff;
}
.kpi-title { font-size:13px; color:#6b7280; margin-bottom:6px; }
.kpi-value { font-size:20px; font-weight:800; margin-top:2px; }
.card { background:#fff; border-radius:8px; padding:12px; box-shadow:0 1px 2px rgba(0,0,0,0.04); border:1px solid rgba(0,0,0,0.04); }
.section-title { font-size:18px; font-weight:700; margin-bottom:6px; }

.hline { display:flex; gap:24px; flex-wrap:wrap; align-items:center; margin-bottom:6px; }
.hitem { min-width:220px; }
.hitem dt { font-weight:700; color:#374151; font-size:13px; margin:0; }
.hitem dd { margin:0; color:#111827; font-size:14px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------------- Constants --------------------
DATA_DIR = "data"
YEAR_DEFAULT = 2025
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
DEFAULT_MONTH = "Oct"
OVER_TOLERANCE = 0.05  # 5% tolerance for over-forecast only; 0% for under

# -------------------- Utilities --------------------
def parse_currency_to_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).replace(",", "")
    s = re.sub(r"[^\d\.\-]", "", s)
    try:
        return float(s) if s != "" else np.nan
    except:
        return np.nan

def indian_abbrev(n):
    try:
        x = float(n)
    except:
        return n
    absx = abs(x)
    if absx >= 1e7:
        return f"{x/1e7:.2f}Cr"
    if absx >= 1e5:
        return f"{x/1e5:.2f}L"
    if absx >= 1e3:
        return f"{x/1e3:.2f}K"
    return f"{int(round(x)):,}"

def month_order(m):
    return MONTHS.index(m) if m in MONTHS else 12

# -------------------- Data Loading --------------------
@st.cache_data
def load_data():
    def pth(fname):
        p1 = os.path.join(DATA_DIR, fname)
        if os.path.exists(p1): return p1
        if os.path.exists(fname): return fname
        return p1

    def try_read_csv(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    forecast = try_read_csv(pth("Forecast.csv"))
    sales = try_read_csv(pth("Sales.csv"))
    inventory = try_read_csv(pth("Inventory.csv"))
    master = try_read_csv(pth("Product_Master.csv"))

    # normalize numeric fields
    if not master.empty:
        for c in master.columns:
            lc = c.lower()
            if any(k in lc for k in ["holding", "unit cost", "selling", "price", "cost"]):
                master[c] = master[c].apply(parse_currency_to_float)
        for c in ["Z", "Lead Time (days)", "Lead Time", "lead_time", "LeadTime"]:
            if c in master.columns:
                master[c] = pd.to_numeric(master[c], errors="coerce")

    for df, col in [(forecast, "Forecast"), (sales, "Sales")]:
        if not df.empty and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if not inventory.empty:
        for c in ["On-Hand Stock", "In-Transit Stock", "Safety Stock", "Total Stock"]:
            if c in inventory.columns:
                inventory[c] = pd.to_numeric(inventory[c], errors="coerce")

    return forecast, sales, inventory, master

forecast_df, sales_df, inventory_df, master_df = load_data()

# SKU -> Product mapping
if not master_df.empty and "SKU" in master_df.columns:
    _mi = master_df.set_index("SKU")
    sku_to_product = {sku: (_mi.at[sku, "Product"] if "Product" in _mi.columns and pd.notna(_mi.at[sku, "Product"]) else sku) for sku in _mi.index}
else:
    sku_to_product = {}

def product_for_sku(sku):
    return sku_to_product.get(sku, sku)

# -------------------- Aggregates & helpers --------------------
def classify_status(inv_qty: float, fc_qty: float, over_tol=OVER_TOLERANCE):
    """Excess if inventory > (1+tol)*forecast, Stockout if inventory < forecast (no tolerance), else Optimal"""
    inv = float(inv_qty or 0.0)
    fc = float(fc_qty or 0.0)
    if inv > (1.0 + over_tol) * fc:
        return "excess"
    if inv < fc:
        return "stockout"
    return "optimal"

def get_col_case_insensitive(df, keys):
    """Return the first matching column name (case-insensitive contains) or None"""
    if df is None or df.empty: return None
    for c in df.columns:
        lc = c.lower()
        for k in keys:
            if k in lc:
                return c
    return None

def build_month_df(year: int, month: str) -> pd.DataFrame:
    """Build month-sliced per-SKU table with financials and status."""
    f_month = forecast_df[(forecast_df.get("Year")==year) & (forecast_df.get("Month")==month)].copy() if not forecast_df.empty else pd.DataFrame()
    s_month = sales_df[(sales_df.get("Year")==year) & (sales_df.get("Month")==month)].copy() if not sales_df.empty else pd.DataFrame()
    inv_month = inventory_df[(inventory_df.get("Year")==year) & (inventory_df.get("Month")==month)].copy() if not inventory_df.empty else pd.DataFrame()

    f_agg = f_month.groupby("SKU")["Forecast"].sum() if (not f_month.empty and "SKU" in f_month.columns and "Forecast" in f_month.columns) else pd.Series(dtype=float)
    s_agg = s_month.groupby("SKU")["Sales"].sum() if (not s_month.empty and "SKU" in s_month.columns and "Sales" in s_month.columns) else pd.Series(dtype=float)
    inv_agg = inv_month.groupby("SKU")["Total Stock"].sum() if (not inv_month.empty and "SKU" in inv_month.columns and "Total Stock" in inv_month.columns) else pd.Series(dtype=float)

    # Union of SKU indexes
    skus = sorted(set(list(getattr(f_agg, "index", []))) | set(list(getattr(s_agg, "index", []))) | set(list(getattr(inv_agg, "index", []))))

    mi = master_df.set_index("SKU") if (not master_df.empty and "SKU" in master_df.columns) else pd.DataFrame()
    holding_col = get_col_case_insensitive(master_df, ["holding"])
    unit_cost_col = get_col_case_insensitive(master_df, ["unit cost"])
    selling_price_col = get_col_case_insensitive(master_df, ["selling", "price"])

    rows = []
    for sku in skus:
        f_val = float(f_agg.get(sku, 0.0)) if hasattr(f_agg, "get") else 0.0
        s_val = float(s_agg.get(sku, 0.0)) if hasattr(s_agg, "get") else 0.0
        inv_val = float(inv_agg.get(sku, 0.0)) if hasattr(inv_agg, "get") else 0.0

        hold = float(mi.at[sku, holding_col]) if (not mi.empty and sku in mi.index and holding_col in mi.columns and pd.notna(mi.at[sku, holding_col])) else np.nan
        unit_cost = float(mi.at[sku, unit_cost_col]) if (not mi.empty and sku in mi.index and unit_cost_col in mi.columns and pd.notna(mi.at[sku, unit_cost_col])) else np.nan
        sell_price = float(mi.at[sku, selling_price_col]) if (not mi.empty and sku in mi.index and selling_price_col in mi.columns and pd.notna(mi.at[sku, selling_price_col])) else np.nan

        rows.append({
            "SKU": sku,
            "Product": product_for_sku(sku),
            "Forecast_Month": f_val,
            "Sales_Month": s_val,
            "Inventory_Month": inv_val,
            "Holding_Cost": hold,
            "Unit_Cost": unit_cost,
            "Selling_Price": sell_price
        })

    dfm = pd.DataFrame(rows).set_index("SKU") if rows else pd.DataFrame()

    if not dfm.empty:
        dfm["Status"] = dfm.apply(lambda r: classify_status(r["Inventory_Month"], r["Forecast_Month"]), axis=1)
        dfm["Excess_Qty"] = (dfm["Inventory_Month"] - dfm["Forecast_Month"]).clip(lower=0)
        dfm["Stockout_Qty"] = (dfm["Forecast_Month"] - dfm["Inventory_Month"]).clip(lower=0)

        # Financials
        dfm["Excess_Holding_Cost"] = dfm["Excess_Qty"] * dfm["Holding_Cost"].fillna(0.0)
        dfm["Unit_Margin"] = (dfm["Selling_Price"].fillna(0.0) - dfm["Unit_Cost"].fillna(0.0))
        dfm["Est_Loss_of_Sale_Value"] = dfm["Stockout_Qty"] * dfm["Unit_Margin"]
        dfm["Inventory_Value"] = dfm["Inventory_Month"] * dfm["Unit_Cost"].fillna(0.0)
    else:
        for c in ["Status","Excess_Qty","Stockout_Qty","Excess_Holding_Cost","Est_Loss_of_Sale_Value","Inventory_Value","Unit_Margin"]:
            dfm[c] = pd.Series(dtype=float)
    return dfm

# -------------------- Plot helpers --------------------
def deviation_waterfall(df, top_n=20):
    """Deviation bar chart: Inventory - Forecast, colored by status."""
    if df.empty:
        return go.Figure()

    d = df.copy()
    d["Deviation"] = d["Inventory_Month"] - d["Forecast_Month"]

    # Sort by absolute deviation
    d = d.reindex(d["Deviation"].abs().sort_values(ascending=False).head(top_n).index)

    # Ensure label column
    if "Product" in d.columns:
        labels = d["Product"].astype(str)
    else:
        labels = d.index.astype(str)

    # Colors
    color_map = {"excess": "#16a34a", "stockout": "#e11d48", "optimal": "#9ca3af"}
    colors = d["Status"].map(lambda s: color_map.get(s, "#9ca3af"))

    fig = go.Figure(go.Bar(
        x=labels,
        y=d["Deviation"],
        marker_color=colors
    ))

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=30, b=80),
        xaxis_title="Product",
        yaxis_title="Inventory - Forecast (Units)",
        title="Inventory vs Forecast — Deviation (Inventory - Forecast)"
    )
    return fig

def pie_status_counts(df):
    if df.empty:
        return go.Figure()
    counts = df["Status"].value_counts().reindex(["optimal","excess","stockout"]).fillna(0).astype(int)
    d = pd.DataFrame({"Status": counts.index, "Count": counts.values})
    fig = px.pie(d, names="Status", values="Count", hole=0.35, title="Inventory Status (SKU count)")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def bar_top_excess_costs(df, top_n=10):
    if df.empty: return go.Figure()
    d = df[df["Status"] != "optimal"].copy()
    d = d.sort_values("Excess_Holding_Cost", ascending=True).tail(top_n)
    if "Product" not in d.columns:
        d["Product"] = d.index.astype(str)
    fig = go.Figure(go.Bar(
        x=d["Excess_Holding_Cost"],
        y=d["Product"].astype(str),
        orientation='h',
        text=d["Excess_Holding_Cost"].apply(indian_abbrev),
        textposition='auto'
    ))
    fig.update_layout(margin=dict(l=80, r=10, t=10, b=10), height=320,
                      xaxis_title="Excess Holding Cost (INR)", yaxis_title="Product", showlegend=False)
    return fig

def bar_top_loss_of_sale(df, top_n=10):
    if df.empty: return go.Figure()
    d = df[df["Status"] != "optimal"].copy()
    d = d.sort_values("Est_Loss_of_Sale_Value", ascending=True).tail(top_n)
    if "Product" not in d.columns:
        d["Product"] = d.index.astype(str)
    fig = go.Figure(go.Bar(
        x=d["Est_Loss_of_Sale_Value"],
        y=d["Product"].astype(str),
        orientation='h',
        text=d["Est_Loss_of_Sale_Value"].apply(indian_abbrev),
        textposition='auto'
    ))
    fig.update_layout(margin=dict(l=80, r=10, t=10, b=10), height=320,
                      xaxis_title="Estimated Loss of Sale (INR)", yaxis_title="Product", showlegend=False)
    return fig

# ------------- Classification helpers for Forecast Accuracy page -------------
def classify_vs_baseline(forecast, baseline, over_tol=OVER_TOLERANCE):
    """Compare forecast against a baseline with 5% over-tolerance, no tolerance for under."""
    f = float(forecast or 0.0)
    b = float(baseline or 0.0)
    if b == 0 and f == 0:
        return "optimal"
    if f > (1.0 + over_tol) * b:
        return "over"
    if f < b:
        return "under"
    return "optimal"

def counts_by_category(series):
    order = ["optimal","over","under"]
    vc = series.value_counts().reindex(order).fillna(0).astype(int)
    return pd.DataFrame({"Category": vc.index, "Count": vc.values})

def bar_counts(df_counts, title):
    fig = px.bar(df_counts, x="Category", y="Count", title=title)
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), yaxis_title="# of SKUs")
    return fig

# -------------------- App UI --------------------
st.title("InventoryApp — Inventory Health & Planning")
st.caption("Overview and tools to monitor inventory health, forecast accuracy and stock planning")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ("Inventory Coverage", "Product View", "Inventory Calculator", "Forecast Accuracy"))

# ----------------- Inventory Coverage -----------------
if page == "Inventory Coverage":
    col_title, col_filter = st.columns([7, 3])
    with col_title:
        st.subheader("Inventory Coverage Dashboard")
        st.write("Month-level inventory health, risk, and financial impact")
    with col_filter:
        selected_month = st.selectbox("Month", options=MONTHS, index=MONTHS.index(DEFAULT_MONTH) if DEFAULT_MONTH in MONTHS else 0)
        selected_year = YEAR_DEFAULT
        st.markdown(f"**Year:** {selected_year}")

    month_df = build_month_df(selected_year, selected_month)

    left, right = st.columns([7, 3])
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.plotly_chart(pie_status_counts(month_df), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        total_inv_value = month_df["Inventory_Value"].sum(skipna=True) if "Inventory_Value" in month_df.columns else 0.0
        total_excess_cost = month_df["Excess_Holding_Cost"].sum(skipna=True) if "Excess_Holding_Cost" in month_df.columns else 0.0
        total_loss_est = month_df["Est_Loss_of_Sale_Value"].sum(skipna=True) if "Est_Loss_of_Sale_Value" in month_df.columns else 0.0
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Total Inventory Value (INR)</div><div class="kpi-value">{indian_abbrev(total_inv_value)}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Loss of Sale (Est, INR)</div><div class="kpi-value">{indian_abbrev(total_loss_est)}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Excess Holding Cost (INR)</div><div class="kpi-value">{indian_abbrev(total_excess_cost)}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Deviation Waterfall / Deviation Bar (AFTER pie + KPI)
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Inventory vs Forecast — Deviation by Product</div>', unsafe_allow_html=True)
    st.plotly_chart(deviation_waterfall(month_df, top_n=20), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Top 10 charts (exclude optimal; label by Product)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Top 10 by Excess Holding Cost</div>', unsafe_allow_html=True)
        st.plotly_chart(bar_top_excess_costs(month_df, top_n=10), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Top 10 by Loss of Sale</div>', unsafe_allow_html=True)
        st.plotly_chart(bar_top_loss_of_sale(month_df, top_n=10), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------- Product View -----------------
elif page == "Product View":
    st.subheader("Product Master View")
    col_l, col_r = st.columns([7, 3])
    with col_r:
        selected_month = st.selectbox("Month", options=MONTHS, index=MONTHS.index(DEFAULT_MONTH) if DEFAULT_MONTH in MONTHS else 0)
        selected_year = YEAR_DEFAULT
        st.markdown(f"**Year:** {selected_year}")
    with col_l:
        product_options = sorted(set(master_df["Product"].dropna().unique().tolist())) if (not master_df.empty and "Product" in master_df.columns) else []
        selected_product = st.selectbox("Select Product / SKU", options=product_options) if product_options else None

    if selected_product:
        # Resolve SKU (first match)
        sku = None
        if not master_df.empty and "SKU" in master_df.columns and "Product" in master_df.columns:
            sku_list = master_df[master_df["Product"] == selected_product]["SKU"].dropna().astype(str).tolist()
            sku = sku_list[0] if sku_list else None

        # Horizontal details: Line 1 and Line 2
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Product Details</div>', unsafe_allow_html=True)

        def safe_val(col):
            try:
                return master_df.set_index("SKU").at[sku, col] if (col in master_df.columns) else "—"
            except Exception:
                return "—"

        line1 = [("SKU ID", sku if sku else "—"),
                 ("Product Name", selected_product),
                 ("Category", safe_val("Category"))]
        line2 = [("Lead Time (days)", safe_val("Lead Time (days)") if "Lead Time (days)" in master_df.columns else safe_val("Lead Time")),
                 ("Service Level Target (Z)", safe_val("Z")),
                 ("Supplier Name", safe_val("Supplier"))]

        def render_line(items):
            cols = st.columns(len(items))
            for i, (k, v) in enumerate(items):
                with cols[i]:
                    st.markdown(f"<div class='hline'><div class='hitem'><dt>{k}</dt><dd>{v}</dd></div></div>", unsafe_allow_html=True)

        render_line(line1)
        render_line(line2)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # Month-on-Month Forecast vs Sales (2025) for selected product
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Monthly Forecast vs Sales (2025)</div>', unsafe_allow_html=True)
        months_list = MONTHS
        fdf = forecast_df[(forecast_df.get("Product") == selected_product) & (forecast_df.get("Year") == YEAR_DEFAULT)].copy() if not forecast_df.empty else pd.DataFrame()
        sdf = sales_df[(sales_df.get("Product") == selected_product) & (sales_df.get("Year") == YEAR_DEFAULT)].copy() if not sales_df.empty else pd.DataFrame()
        f_series = fdf.set_index("Month")["Forecast"] if not fdf.empty else pd.Series(dtype=float)
        s_series = sdf.set_index("Month")["Sales"] if not sdf.empty else pd.Series(dtype=float)
        dmo = pd.DataFrame({"Month": months_list})
        dmo["Forecast"] = dmo["Month"].map(f_series).fillna(0)
        dmo["Sales"] = dmo["Month"].map(s_series).fillna(0)
        fig_pv = go.Figure()
        fig_pv.add_trace(go.Bar(x=dmo["Month"], y=dmo["Forecast"], name="Forecast"))
        fig_pv.add_trace(go.Bar(x=dmo["Month"], y=dmo["Sales"], name="Sales"))
        fig_pv.update_layout(barmode='group', height=340, margin=dict(l=10, r=10, t=10, b=10),
                             yaxis_title="Units", xaxis_title="Month")
        st.plotly_chart(fig_pv, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Inventory Health for selected month
        month_df = build_month_df(selected_year, selected_month)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">Inventory Health — {selected_month} {selected_year}</div>', unsafe_allow_html=True)

        f_val = float(forecast_df[(forecast_df.get("Product") == selected_product) &
                                  (forecast_df.get("Year") == selected_year) &
                                  (forecast_df.get("Month") == selected_month)]["Forecast"].sum()) if not forecast_df.empty else 0.0
        s_val = float(sales_df[(sales_df.get("Product") == selected_product) &
                               (sales_df.get("Year") == selected_year) &
                               (sales_df.get("Month") == selected_month)]["Sales"].sum()) if not sales_df.empty else 0.0
        inv_row = inventory_df[(inventory_df.get("SKU") == sku) &
                               (inventory_df.get("Year") == selected_year) &
                               (inventory_df.get("Month") == selected_month)] if (not inventory_df.empty and sku) else pd.DataFrame()
        on_hand = int(inv_row.iloc[0]["On-Hand Stock"]) if (not inv_row.empty and "On-Hand Stock" in inv_row.columns) else 0
        in_transit = int(inv_row.iloc[0]["In-Transit Stock"]) if (not inv_row.empty and "In-Transit Stock" in inv_row.columns) else 0
        safety_stock = int(inv_row.iloc[0]["Safety Stock"]) if (not inv_row.empty and "Safety Stock" in inv_row.columns) else 0

        avg_monthly_sales = float(sales_df[(sales_df.get("Product") == selected_product) &
                                           (sales_df.get("Year") == selected_year)]["Sales"].mean()) if not sales_df.empty else 0.0
        avg_daily = avg_monthly_sales / 30.0 if avg_monthly_sales > 0 else 0.0
        stock_coverage_days = int((on_hand / avg_daily)) if avg_daily > 0 else "N/A"

        status = "Optimal"
        if on_hand > (1.0 + OVER_TOLERANCE) * f_val:
            status = "Excess"
        elif on_hand < f_val:
            status = "Stockout"

        cL, cR = st.columns([2, 1])
        with cL:
            st.markdown(f"**Status:** {status}")
            st.markdown(f"**Forecast ({selected_month}):** {int(f_val):,} units")
            st.markdown(f"**Actual ({selected_month}):** {int(s_val):,} units")
            st.markdown(f"**On-hand:** {on_hand:,} units")
            st.markdown(f"**In-transit:** {in_transit:,} units")
            st.markdown(f"**Safety Stock:** {safety_stock:,} units")
            st.markdown(f"**Stock Coverage (days):** {stock_coverage_days}")
        with cR:
            ex_cost = month_df.loc[month_df["Product"] == selected_product, "Excess_Holding_Cost"].sum() if not month_df.empty else 0.0
            lossv = month_df.loc[month_df["Product"] == selected_product, "Est_Loss_of_Sale_Value"].sum() if not month_df.empty else 0.0
            st.markdown(f'<div class="kpi-card"><div class="kpi-title">Excess Holding Cost (This Month)</div><div class="kpi-value">{indian_abbrev(ex_cost)}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-card"><div class="kpi-title">Loss of Sale (This Month)</div><div class="kpi-value">{indian_abbrev(lossv)}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- Inventory Calculator ----------
elif page == "Inventory Calculator":
    st.subheader("Inventory Calculator")
    # product selector
    product_options = sorted(set(master_df["Product"].dropna().unique().tolist())) if (not master_df.empty and "Product" in master_df.columns) else []
    sel_product = st.selectbox("Select Product", options=product_options) if product_options else None

    sku = None
    if sel_product and (not master_df.empty) and "SKU" in master_df.columns:
        sku_list = master_df[master_df["Product"] == sel_product]["SKU"].dropna().astype(str).tolist()
        sku = sku_list[0] if sku_list else None

    # Z & Lead Time from master
    try:
        mi = master_df.set_index("SKU")
    except Exception:
        mi = pd.DataFrame()

    Z = 1.65
    if (not mi.empty) and (sku in mi.index) and ("Z" in mi.columns) and pd.notna(mi.at[sku, "Z"]):
        try:
            Z = float(mi.at[sku, "Z"])
        except Exception:
            Z = 1.65

    lead_time_days = 14.0
    lt_col = "Lead Time (days)" if "Lead Time (days)" in master_df.columns else ("Lead Time" if "Lead Time" in master_df.columns else None)
    if lt_col and (not mi.empty) and (sku in mi.index) and pd.notna(mi.at[sku, lt_col]):
        try:
            lead_time_days = float(mi.at[sku, lt_col])
        except Exception:
            pass

    # Build 2025 monthly sales series for SKU
    months_list = MONTHS
    s_series = None
    if sku and not sales_df.empty:
        try:
            s_series = sales_df[(sales_df["SKU"] == sku) & (sales_df["Year"] == YEAR_DEFAULT)].set_index("Month").reindex(months_list)["Sales"].fillna(0).astype(float)
        except Exception:
            s_series = None

    if s_series is not None and len(s_series) > 0:
        last3 = s_series.values[-3:] if len(s_series) >= 3 else s_series.values
        sigma_monthly = float(np.std(last3, ddof=1)) if len(last3) > 1 else float(np.std(last3))
        avg_monthly = float(np.mean(s_series.values))
    else:
        sigma_monthly = 0.0
        avg_monthly = 0.0

    avg_daily = avg_monthly / 30.0 if avg_monthly > 0 else 0.0
    lead_time_months = (lead_time_days / 30.0) if lead_time_days > 0 else 0.0

    st.markdown("**Safety Stock formula**: `Z × σ_d × √L` where σ_d is std dev of monthly demand (last 3 months), L in months.")
    safety_stock_units = (Z * sigma_monthly * sqrt(lead_time_months)) if lead_time_months > 0 else 0.0
    safety_days = (safety_stock_units / avg_daily) if avg_daily > 0 else 0.0

    demand_during_lead = avg_daily * lead_time_days
    rop_units = demand_during_lead + safety_stock_units
    rop_days = (rop_units / avg_daily) if avg_daily > 0 else 0.0

    stock_norm_units = avg_daily * (lead_time_days + safety_days)
    stock_norm_days = (lead_time_days + safety_days)

    # Formulas & step-by-step calculations
    st.markdown("**Reorder Point (ROP)** = Average Daily Demand × Lead Time (days) + Safety Stock")
    st.markdown(f"Calculation: ROP = {avg_daily:.3f} × {lead_time_days:.1f} + {safety_stock_units:.2f} = **{rop_units:.2f} units**")
    st.markdown("**Stock Norm** = Average Daily Demand × (Lead Time (days) + Safety Days)")
    st.markdown(f"Calculation: Stock Norm = {avg_daily:.3f} × ({lead_time_days:.1f} + {safety_days:.1f}) = **{stock_norm_units:.2f} units**")

    # Final line (3 columns x 2 rows text; no KPI cards)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Safety Stock**")
        st.write(f"{safety_stock_units:.0f} units")
        st.write(f"{safety_days:.1f} days")
    with c2:
        st.markdown("**Reorder Point**")
        st.write(f"{rop_units:.0f} units")
        st.write(f"{rop_days:.1f} days")
    with c3:
        st.markdown("**Stock Norm**")
        st.write(f"{stock_norm_units:.0f} units")
        st.write(f"{stock_norm_days:.1f} days")

# ------------- Forecast Accuracy -------------
elif page == "Forecast Accuracy":
    st.subheader("Forecast Accuracy Dashboard")
    col_title, col_filter = st.columns([7, 3])
    with col_title:
        st.write("Analyze forecast vs demand using multiple baselines")
    with col_filter:
        selected_month = st.selectbox("Month", options=MONTHS, index=MONTHS.index(DEFAULT_MONTH) if DEFAULT_MONTH in MONTHS else 0)
        selected_year = YEAR_DEFAULT
        st.markdown(f"**Year:** {selected_year}")

    # First visual: Forecast vs Sales — Month on Month (plot only months that have Sales values)
    f_all = forecast_df[forecast_df.get("Year") == selected_year].copy() if not forecast_df.empty else pd.DataFrame()
    s_all = sales_df[sales_df.get("Year") == selected_year].copy() if not sales_df.empty else pd.DataFrame()
    merged = pd.merge(f_all, s_all, on=["SKU", "Product", "Year", "Month"], how="outer", suffixes=("_f", "_s")) if (not f_all.empty or not s_all.empty) else pd.DataFrame()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">Forecast vs Sales — Month on Month ({selected_year})</div>', unsafe_allow_html=True)
    if not merged.empty:
        month_agg = merged.groupby("Month").agg({"Forecast": "sum", "Sales": "sum"})
        # keep only months with Sales > 0
        month_agg = month_agg.loc[month_agg["Sales"] > 0]
        month_agg = month_agg.reindex(MONTHS).dropna(how="any")
        if not month_agg.empty:
            fig_fva = go.Figure()
            fig_fva.add_trace(go.Bar(x=month_agg.index, y=month_agg["Forecast"], name="Forecast"))
            fig_fva.add_trace(go.Bar(x=month_agg.index, y=month_agg["Sales"], name="Sales"))
            fig_fva.update_layout(barmode="group", height=360, margin=dict(l=10, r=10, t=10, b=10),
                                  yaxis_title="Units", xaxis_title="Month")
            st.plotly_chart(fig_fva, use_container_width=True)
        else:
            st.info("No months with Sales available to plot yet.")
    else:
        st.info("No data available to plot.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Then: 12M, 3M (stability baseline), LYSM (seasonality)
    if not sales_df.empty:
        sales_12m = sales_df.groupby("SKU")["Sales"].mean()

        s_y = sales_df[sales_df["Year"] == selected_year].copy()
        def avg_last_n(group, n):
            g = group.sort_values("Month", key=lambda s: s.map(lambda m: month_order(m)))
            vals = g["Sales"].tail(n).values
            return float(np.mean(vals)) if len(vals) > 0 else 0.0
        sales_3m = s_y.groupby("SKU").apply(lambda g: avg_last_n(g, 3))

        s_ly = sales_df[(sales_df["Year"] == selected_year - 1) & (sales_df["Month"] == selected_month)]
        ly_same = s_ly.groupby("SKU")["Sales"].sum()
    else:
        sales_12m = pd.Series(dtype=float)
        sales_3m = pd.Series(dtype=float)
        ly_same = pd.Series(dtype=float)

    f_sel = forecast_df[(forecast_df.get("Year") == selected_year) & (forecast_df.get("Month") == selected_month)]
    f_by_sku = f_sel.groupby("SKU")["Forecast"].sum() if not f_sel.empty else pd.Series(dtype=float)

    all_skus = sorted(set(list(f_by_sku.index)) |
                      set(list(sales_12m.index)) |
                      set(list(sales_3m.index)) |
                      set(list(ly_same.index)))
    def aligned(series):
        return pd.Series({sku: float(series.get(sku, 0.0)) for sku in all_skus})

    fA = aligned(f_by_sku)
    b12 = aligned(sales_12m)
    b3 = aligned(sales_3m)
    bly = aligned(ly_same)

    cat_12 = fA.combine(b12, lambda f, b: classify_vs_baseline(f, b))
    cat_3 = fA.combine(b3, lambda f, b: classify_vs_baseline(f, b))
    cat_ly = fA.combine(bly, lambda f, b: classify_vs_baseline(f, b))

    c12 = counts_by_category(cat_12)
    c3 = counts_by_category(cat_3)
    cly = counts_by_category(cat_ly)

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.plotly_chart(
            bar_counts(c12, f"Forecast vs 12M Avg Sales — {selected_month} {selected_year} <BR>(12M Avg — Long-term Demand Baseline)"),
            use_container_width=True
        )
    with cc2:
        st.plotly_chart(
            bar_counts(c3, f"Forecast vs 3M Avg Sales — {selected_month} {selected_year} <BR>(3M Avg — Short-term Trend Sensitivity)"),
            use_container_width=True
        )
    with cc3:
        st.plotly_chart(
            bar_counts(cly, f"Forecast vs LY Same Month — {selected_month} {selected_year} <BR>(LYSM — Seasonality Comparison)"),
            use_container_width=True
        )

# End
