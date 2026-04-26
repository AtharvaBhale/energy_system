import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from io import BytesIO

from src.ai_agent import answer_analyst_query
from src.data_pipeline import fetch_and_clean_eia_data
from src.forecasting import generate_forecast

st.set_page_config(page_title="CDF Energy AI Command Center", page_icon="⚡", layout="wide")

HACKATHON_GOLD_SCALE = [
    [0.0, "#2f3d2f"],
    [0.25, "#6f7d65"],
    [0.5, "#b8a070"],
    [0.75, "#d9c49a"],
    [1.0, "#f0e8da"],
]

HACKATHON_GOLD_SCALE_DARK = [
    [0.0, "#1a241a"],
    [0.35, "#3d4f38"],
    [0.55, "#6b7a5c"],
    [0.75, "#a89a6e"],
    [1.0, "#e8dcc4"],
]

REGION_DEFAULTS = {
    "Texas": {"ip_rate": 900, "decline_rate": 22.0, "dnc_cost_mm": 8.5, "opex_per_bbl": 13.0, "price": 70.0},
    "North Dakota": {"ip_rate": 700, "decline_rate": 25.0, "dnc_cost_mm": 9.0, "opex_per_bbl": 16.0, "price": 68.0},
    "New Mexico": {"ip_rate": 850, "decline_rate": 21.0, "dnc_cost_mm": 8.2, "opex_per_bbl": 12.0, "price": 70.0},
    "Oklahoma": {"ip_rate": 500, "decline_rate": 18.0, "dnc_cost_mm": 6.5, "opex_per_bbl": 14.0, "price": 67.0},
}


def init_session_state():
    if "ui_theme" not in st.session_state:
        st.session_state["ui_theme"] = "light"
    if "selected_region" not in st.session_state:
        st.session_state["selected_region"] = None


def apply_hackathon_theme(mode: str):
    if mode == "dark":
        pio.templates.default = "plotly_dark"
        st.markdown(
            """
            <style>
            .stApp {
                background: linear-gradient(180deg, #0f1218 0%, #1a1f28 100%);
                color: #e8eaed;
            }
            [data-testid="stSidebar"] {
                background: #141820;
                border-right: 1px solid #2a3140;
            }
            [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
                color: #c4c9d4 !important;
            }
            .main .block-container { color: #e8eaed; }
            .stTabs [data-baseweb="tab-list"] { background-color: #1a1f28; gap: 4px; }
            div[data-testid="stMetricValue"] { color: #f0e8da; }
            div[data-testid="stMetricLabel"] { color: #9aa3b2; }
            .hero {
                background: linear-gradient(90deg, rgba(26,36,26,0.98), rgba(45,58,42,0.95));
                color: #f6ecde;
                border-radius: 16px;
                padding: 18px 22px;
                margin-bottom: 14px;
                border: 1px solid #5c6b52;
            }
            .hero h1 { margin: 0; font-size: 1.8rem; }
            .hero p { margin: 6px 0 0 0; opacity: 0.92; }
            [data-testid="stDataFrame"] { border-radius: 8px; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        pio.templates.default = "plotly"
        st.markdown(
            """
            <style>
            .stApp {
                background: linear-gradient(180deg, #f7f3eb 0%, #ece4d6 100%);
                color: #1f2a1f;
            }
            [data-testid="stSidebar"] {
                background: #f2ebdd;
                border-right: 1px solid #d5c7ad;
            }
            .hero {
                background: linear-gradient(90deg, rgba(35,49,35,0.96), rgba(70,82,57,0.92));
                color: #f6ecde;
                border-radius: 16px;
                padding: 18px 22px;
                margin-bottom: 14px;
                border: 1px solid #b9a37e;
            }
            .hero h1 { margin: 0; font-size: 1.8rem; }
            .hero p { margin: 6px 0 0 0; opacity: 0.95; }
            .kpi-label { font-size: 0.82rem; color: #5d6a58; text-transform: uppercase; letter-spacing: 0.08em; }
            </style>
            """,
            unsafe_allow_html=True,
        )


def calculate_well_economics(ip_rate, decline_rate, dnc_cost_mm, opex_per_bbl, price):
    months = pd.Series(range(1, 121), name="Month")
    year_bucket = ((months - 1) // 12) + 1
    monthly_decline = (1 - (decline_rate / 100)) ** (1 / 12)
    production = ip_rate * (monthly_decline ** (months - 1)) * 30
    revenue = production * price
    opex = production * opex_per_bbl
    net_cash_flow = revenue - opex
    net_cash_flow.iloc[0] -= dnc_cost_mm * 1_000_000
    discount_rate_m = (1 + 0.10) ** (1 / 12) - 1
    discounted = net_cash_flow / ((1 + discount_rate_m) ** months)
    cumulative = net_cash_flow.cumsum()

    annual = pd.DataFrame({"Year": year_bucket, "Net_Cash_Flow": net_cash_flow}).groupby(
        "Year", as_index=False
    )["Net_Cash_Flow"].sum()
    annual.columns = ["Year", "Annual_Cash_Flow"]

    payback_candidates = months[cumulative > 0]
    payback_month = int(payback_candidates.iloc[0]) if not payback_candidates.empty else None

    monthly_df = pd.DataFrame(
        {
            "Month": months,
            "Production_bbl": production,
            "Net_Cash_Flow": net_cash_flow,
            "Cumulative_Cash_Flow": cumulative,
        }
    )
    return {
        "monthly_df": monthly_df,
        "annual_df": annual,
        "eur": production.sum(),
        "npv10": discounted.sum(),
        "irr_proxy": ((net_cash_flow.sum() / (dnc_cost_mm * 1_000_000)) / 10) * 100,
        "payback_month": payback_month,
    }


def compute_custom_kpis(forecast_df, latest_year_df, selected_year):
    current_total = latest_year_df["Production_Volume"].sum()
    projected_total = forecast_df["Projected_Production"].sum()
    years_delta = max(1, selected_year - int(latest_year_df["Year"].max()))
    cagr = ((projected_total / current_total) ** (1 / years_delta) - 1) * 100 if current_total > 0 else 0

    top5 = forecast_df.nlargest(5, "Projected_Production")
    concentration = (top5["Projected_Production"].sum() / projected_total * 100) if projected_total > 0 else 0
    volatility = forecast_df["Projected_Production"].std() / forecast_df["Projected_Production"].mean() * 100
    return {
        "portfolio_cagr_pct": cagr,
        "top5_concentration_pct": concentration,
        "cross_region_volatility_pct": volatility,
    }


def build_sensitivity_matrix(forecast_df, base_price=70.0):
    decline_grid = [5, 10, 15, 20, 25, 30]
    price_grid = [50, 60, 70, 80, 90]
    base_total = forecast_df["Projected_Production"].sum()
    rows = []
    for d in decline_grid:
        row = {"Decline_%": d}
        for p in price_grid:
            stressed_total = forecast_df["Projected_Production"].sum() * (1 - d / 100)
            revenue_index = (stressed_total * p) / (base_total * base_price) if base_total > 0 else 0
            row[f"${p}"] = revenue_index
        rows.append(row)
    sens_df = pd.DataFrame(rows)
    return sens_df, price_grid


def build_excel_workbook(forecast_df, selected_year):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        export_df = forecast_df.copy().sort_values("Projected_Production", ascending=False).reset_index(drop=True)
        export_df.to_excel(writer, index=False, sheet_name="Forecast")
        workbook = writer.book
        sheet = writer.sheets["Forecast"]
        money_fmt = workbook.add_format({"num_format": "$#,##0"})
        pct_fmt = workbook.add_format({"num_format": "0.00%"})
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#ECE4D6", "border": 1})
        for col_idx, col_name in enumerate(export_df.columns):
            sheet.write(0, col_idx, col_name, header_fmt)
        start_col = len(export_df.columns)
        sheet.write(0, start_col, "Assumed_Price", header_fmt)
        sheet.write(0, start_col + 1, "Revenue_Potential", header_fmt)
        sheet.write(0, start_col + 2, "Share_of_Portfolio", header_fmt)
        for i in range(1, len(export_df) + 1):
            sheet.write_number(i, start_col, 70)
            sheet.write_formula(i, start_col + 1, f"=C{i+1}*{chr(65+start_col)}{i+1}", money_fmt)
            sheet.write_formula(i, start_col + 2, f"=C{i+1}/SUM($C$2:$C${len(export_df)+1})", pct_fmt)
        sheet.set_column(0, start_col + 2, 20)
        notes = pd.DataFrame(
            {
                "Metadata": [
                    f"Selected Year: {selected_year}",
                    "Revenue_Potential = Projected_Production * Assumed_Price",
                    "Share_of_Portfolio = Region Projected_Production / Total Projected_Production",
                ]
            }
        )
        notes.to_excel(writer, index=False, sheet_name="ReadMe")
    output.seek(0)
    return output.getvalue()


def build_map(map_df, overlay_col, is_dark: bool):
    scale = HACKATHON_GOLD_SCALE_DARK if is_dark else HACKATHON_GOLD_SCALE
    line_color = "#2a3140" if is_dark else "#f6f0e4"
    font_color = "#e8eaed" if is_dark else "#1f2a1f"
    geo_bg = "#1a1f28" if is_dark else "rgba(0,0,0,0)"
    map_fig = px.choropleth(
        map_df,
        locations="State_Code",
        locationmode="USA-states",
        color=overlay_col,
        scope="usa",
        hover_name="Region",
        hover_data={
            "Production_Volume": ":,.0f",
            "Projected_Production": ":,.0f",
            "Stressed_Production": ":,.0f",
            "YoY_Growth_Pct": ":.2f",
            "Relative_Performance": ":.2f",
            "State_Code": False,
        },
        color_continuous_scale=scale,
    )
    map_fig.update_traces(marker_line_color=line_color, marker_line_width=1.2)
    map_fig.update_layout(
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        geo_bgcolor=geo_bg,
        font={"color": font_color},
        template="plotly_dark" if is_dark else "plotly",
        geo=dict(
            lakecolor=geo_bg,
            bgcolor=geo_bg,
            landcolor="#2a3140" if is_dark else "#f5f0e6",
            showlakes=True,
            showland=True,
        ),
    )
    return map_fig


def get_api_key():
    secret_key = st.secrets.get("EIA_API_KEY", "")
    with st.sidebar:
        st.title("Mission Settings")
        dark = st.checkbox(
            "Dark mode",
            value=st.session_state.get("ui_theme") == "dark",
            help="Switches light/dark styling for the whole app (Streamlit reruns the page).",
        )
        st.session_state["ui_theme"] = "dark" if dark else "light"
        st.divider()
        if secret_key:
            st.success("EIA key loaded from secrets.")
            with st.expander("Override API key"):
                manual_key = st.text_input("Temporary API key", type="password")
        else:
            st.warning("No EIA key in secrets. Add one or paste below.")
            manual_key = st.text_input("EIA API Key", type="password")
        api_key = manual_key.strip() or secret_key
        if st.button("Refresh Live Data"):
            fetch_and_clean_eia_data.clear()
            st.success("Live data cache refreshed.")
        st.divider()
        selected_year = st.slider("Forecast year", 2020, 2035, 2027)
        decline_rate = st.slider("Downside stress (%)", 0.0, 30.0, 0.0, 0.5)
        overlay_labels = {
            "Projected production": "Projected_Production",
            "Stressed production": "Stressed_Production",
            "Year-over-year growth %": "YoY_Growth_Pct",
            "Relative performance index": "Relative_Performance",
        }
        overlay_label = st.selectbox("Map view", list(overlay_labels.keys()))
        overlay = overlay_labels[overlay_label]
        st.caption("Hackathon Demo UI")
        st.divider()
        st.caption("Region focus")
        if st.session_state.get("selected_region"):
            st.write(f"**{st.session_state['selected_region']}**")
            if st.button("Clear region selection", use_container_width=True):
                st.session_state["selected_region"] = None
                st.rerun()
        else:
            st.caption("No state selected — map shows all regions.")
    return api_key, selected_year, decline_rate, overlay


init_session_state()
api_key, selected_year, decline_rate, overlay_choice = get_api_key()
is_dark = st.session_state.get("ui_theme") == "dark"
apply_hackathon_theme(st.session_state.get("ui_theme", "light"))

st.markdown(
    """
    <div class="hero">
      <h1>CDF Energy AI Command Center</h1>
      <p>Oil & Gas Investment Intelligence Platform • Forecasting • Interactive Geospatial Analysis • AI Analyst</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if api_key:
    try:
        df, metadata = fetch_and_clean_eia_data(api_key)
    except Exception as e:
        st.error(f"Data load error: {e}")
        df, metadata = pd.DataFrame(), {}

    if not df.empty:
        forecast_df = generate_forecast(df, selected_year, decline_adjustment_pct=decline_rate)
        latest_year_df = df[df["Year"] == df["Year"].max()].copy()
        prior_year_df = df[df["Year"] == df["Year"].max() - 1][["Region", "Production_Volume"]].copy()
        prior_year_df.columns = ["Region", "Prior_Production"]

        map_df = (
            latest_year_df[["Region", "State_Code", "Production_Volume"]]
            .groupby(["Region", "State_Code"], as_index=False)
            .sum()
            .merge(prior_year_df, on="Region", how="left")
            .merge(forecast_df[["Region", "Projected_Production", "Stressed_Production"]], on="Region", how="left")
        )
        map_df["Prior_Production"] = map_df["Prior_Production"].fillna(map_df["Production_Volume"])
        map_df["YoY_Growth_Pct"] = (
            (map_df["Production_Volume"] - map_df["Prior_Production"]) / map_df["Prior_Production"].replace(0, pd.NA)
        ).fillna(0) * 100
        map_df["Relative_Performance"] = map_df["Projected_Production"] / map_df["Projected_Production"].mean()

        top_region = forecast_df.loc[forecast_df["Projected_Production"].idxmax()]
        active_projection_col = "Stressed_Production" if decline_rate > 0 else "Projected_Production"
        active_avg = forecast_df[active_projection_col].mean()
        kpis = compute_custom_kpis(forecast_df, latest_year_df, selected_year)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Lead Region", top_region["Region"])
        k2.metric("Avg Projection", f"{int(active_avg):,} bbl")
        k3.metric("Forecast Year", selected_year)
        k4.metric("Stress Scenario", f"{decline_rate:.1f}%")
        st.caption(
            f"Source: {metadata.get('source', 'Unknown')} | Last refresh (UTC): {metadata.get('last_updated_utc', 'N/A')}"
        )

        st.markdown("### Tier 2 KPI Pack")
        t1, t2, t3 = st.columns(3)
        t1.metric("Portfolio CAGR", f"{kpis['portfolio_cagr_pct']:.2f}%")
        t2.metric("Top-5 Concentration", f"{kpis['top5_concentration_pct']:.1f}%")
        t3.metric("Cross-Region Volatility", f"{kpis['cross_region_volatility_pct']:.2f}%")

        tabs = st.tabs(["Live Map Studio", "Forecast & KPI Grid", "AI Analyst", "Well Economics", "Sensitivity Lab"])

        with tabs[0]:
            st.subheader("🗺️ Interactive Geographic Intelligence")
            map_fig = build_map(map_df, overlay_choice, is_dark)
            try:
                map_event = st.plotly_chart(map_fig, use_container_width=True, on_select="rerun")
            except TypeError:
                map_event = st.plotly_chart(map_fig, use_container_width=True)
            if map_event and map_event.get("selection", {}).get("points"):
                clicked_code = map_event["selection"]["points"][0].get("location")
                selected_match = map_df[map_df["State_Code"] == clicked_code]
                if not selected_match.empty:
                    st.session_state["selected_region"] = selected_match.iloc[0]["Region"]
            row_map = st.columns([3, 1])
            with row_map[0]:
                if st.session_state["selected_region"]:
                    st.success(
                        f"Selected region: **{st.session_state['selected_region']}** — filters forecast, charts, and well defaults."
                    )
                else:
                    st.info("Click a state on the map to focus one region, or choose **All regions** below / **Clear** in the sidebar.")
            with row_map[1]:
                if st.button("Clear selection", key="clear_map_tab", use_container_width=True):
                    st.session_state["selected_region"] = None
                    st.rerun()

            region_options = ["— All regions —"] + sorted(map_df["Region"].unique().tolist())
            sel = st.session_state.get("selected_region")
            region_index = region_options.index(sel) if sel in region_options else 0
            region_pick = st.selectbox(
                "Focus region (optional)",
                options=region_options,
                index=region_index,
            )
            if region_pick == "— All regions —":
                st.session_state["selected_region"] = None
            else:
                st.session_state["selected_region"] = region_pick

            top5 = map_df.sort_values(overlay_choice, ascending=False).head(5)[["Region", overlay_choice]]
            st.dataframe(top5, hide_index=True, use_container_width=True)

        with tabs[1]:
            selected_region = st.session_state.get("selected_region")
            c1, c2 = st.columns([2, 1])
            with c1:
                display_df = forecast_df.copy()
                if selected_region:
                    display_df = display_df[display_df["Region"] == selected_region]
                st.dataframe(display_df.sort_values(active_projection_col, ascending=False), hide_index=True, use_container_width=True)
            with c2:
                bar_df = forecast_df.copy()
                if selected_region:
                    bar_df = bar_df[bar_df["Region"] == selected_region]
                st.bar_chart(bar_df.set_index("Region")[active_projection_col])
            xlsx_bytes = build_excel_workbook(forecast_df, selected_year)
            st.download_button(
                "Download Analyst Excel Pack",
                data=xlsx_bytes,
                file_name=f"energy_intelligence_forecast_{selected_year}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        with tabs[2]:
            st.subheader("🤖 Conversational Analyst Agent")
            analyst_prompt = st.text_input(
                "Ask your analyst",
                placeholder="Summarize the opportunity in Texas and compare downside under 15% steeper decline.",
            )
            if st.button("Run Analysis", type="primary"):
                agent_output = answer_analyst_query(analyst_prompt, forecast_df, selected_year)
                for k, v in agent_output.get("actions", {}).items():
                    if k == "selected_region":
                        st.session_state["selected_region"] = v
                st.markdown(agent_output["response"])
            st.caption("AI output is split between verified data-backed claims and model inference.")

        with tabs[3]:
            st.subheader("🧮 Well Economics Calculator (Tier 2 Stretch)")
            region_for_defaults = st.session_state.get("selected_region") or top_region["Region"]
            defaults = REGION_DEFAULTS.get(
                region_for_defaults,
                {"ip_rate": 600, "decline_rate": 20.0, "dnc_cost_mm": 7.5, "opex_per_bbl": 14.0, "price": 68.0},
            )
            w1, w2, w3, w4, w5 = st.columns(5)
            ip_rate = w1.number_input("Initial Daily Rate (bbl/d)", value=float(defaults["ip_rate"]), min_value=100.0, step=25.0)
            well_decline = w2.number_input("Annual Decline (%)", value=float(defaults["decline_rate"]), min_value=1.0, max_value=60.0, step=0.5)
            dnc_cost = w3.number_input("Drill+Complete Cost ($MM)", value=float(defaults["dnc_cost_mm"]), min_value=1.0, step=0.1)
            opex = w4.number_input("LOE ($/bbl)", value=float(defaults["opex_per_bbl"]), min_value=1.0, step=0.5)
            price = w5.number_input("Commodity Price ($/bbl)", value=float(defaults["price"]), min_value=20.0, step=1.0)

            well = calculate_well_economics(ip_rate, well_decline, dnc_cost, opex, price)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("EUR (10y)", f"{int(well['eur']):,} bbl")
            m2.metric("NPV10", f"${int(well['npv10']):,}")
            m3.metric("IRR Proxy", f"{well['irr_proxy']:.1f}%")
            m4.metric("Payback", f"{well['payback_month']} mo" if well["payback_month"] else "No payback")

            d_line, d_fill = ("#8fbc8f", "rgba(143,188,143,0.2)") if is_dark else ("#3f5a3f", "rgba(184,160,112,0.25)")
            c_line, c_fill = ("#c4d4c4", "rgba(196,212,196,0.18)") if is_dark else ("#1f2a1f", "rgba(111,125,101,0.22)")

            viz1, viz2 = st.columns(2)
            with viz1:
                decline_fig = go.Figure()
                decline_fig.add_trace(
                    go.Scatter(
                        x=well["monthly_df"]["Month"],
                        y=well["monthly_df"]["Production_bbl"],
                        mode="lines",
                        line={"color": d_line, "width": 3},
                        fill="tozeroy",
                        fillcolor=d_fill,
                        name="Decline Curve",
                    )
                )
                decline_fig.update_layout(
                    title="Production Decline Curve",
                    xaxis_title="Month",
                    yaxis_title="bbl/month",
                    height=340,
                    template="plotly_dark" if is_dark else "plotly",
                )
                st.plotly_chart(decline_fig, use_container_width=True)
            with viz2:
                cash_fig = go.Figure()
                cash_fig.add_trace(
                    go.Scatter(
                        x=well["monthly_df"]["Month"],
                        y=well["monthly_df"]["Cumulative_Cash_Flow"],
                        mode="lines",
                        line={"color": c_line, "width": 3},
                        fill="tozeroy",
                        fillcolor=c_fill,
                        name="Cumulative Cash Flow",
                    )
                )
                cash_fig.update_layout(
                    title="Cumulative Cash Flow",
                    xaxis_title="Month",
                    yaxis_title="$",
                    height=340,
                    template="plotly_dark" if is_dark else "plotly",
                )
                st.plotly_chart(cash_fig, use_container_width=True)

        with tabs[4]:
            st.subheader("📈 Sensitivity Analysis Matrix")
            st.caption("Revenue opportunity index across decline and price assumptions, tied to current forecast year.")
            sens_df, price_grid = build_sensitivity_matrix(forecast_df)
            heat_df = sens_df.set_index("Decline_%")
            heat_fig = px.imshow(
                heat_df,
                aspect="auto",
                color_continuous_scale="RdYlGn",
                labels={"color": "Revenue Index"},
                text_auto=".2f",
            )
            heat_fig.update_layout(
                height=420,
                xaxis_title="Price Assumption",
                yaxis_title="Decline Rate (%)",
                template="plotly_dark" if is_dark else "plotly",
            )
            st.plotly_chart(heat_fig, use_container_width=True)
            st.dataframe(sens_df, use_container_width=True, hide_index=True)
    else:
        st.error("Invalid API Key or no data found.")
else:
    st.info("No API key found. Add `EIA_API_KEY` in `.streamlit/secrets.toml` or enter manually in sidebar.")