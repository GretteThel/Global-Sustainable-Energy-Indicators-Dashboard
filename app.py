from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Sustainable Energy Indicators Dashboard",
    page_icon="🌍",
    layout="wide",
)

# ============================================================
# CONSTANTS
# ============================================================
DATA_FILE = Path(__file__).parent / "global-data-on-sustainable-energy.csv"

OKABE_ITO = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
]

GENERATION_COLOURS = {
    "Fossil fuels": "#7A7A7A",
    "Nuclear": "#E69F00",
    "Renewables": "#0072B2",
}

COLUMN_RENAME = {
    "Entity": "country",
    "Year": "year",
    "Access to electricity (% of population)": "electricity_access_pct",
    "Access to clean fuels for cooking": "clean_fuels_pct",
    "Renewable-electricity-generating-capacity-per-capita": "renewable_capacity_pc",
    "Financial flows to developing countries (US $)": "financial_flows_usd",
    "Renewable energy share in the total final energy consumption (%)": "renewable_energy_share_pct",
    "Electricity from fossil fuels (TWh)": "fossil_twh",
    "Electricity from nuclear (TWh)": "nuclear_twh",
    "Electricity from renewables (TWh)": "renewables_twh",
    "Low-carbon electricity (% electricity)": "low_carbon_electricity_pct",
    "Primary energy consumption per capita (kWh/person)": "energy_consumption_pc_kwh",
    "Energy intensity level of primary energy (MJ/$2017 PPP GDP)": "energy_intensity_mj_per_gdp",
    "Value_co2_emissions_kt_by_country": "co2_emissions_kt",
    "Renewables (% equivalent primary energy)": "renewables_equiv_primary_energy_pct",
    "gdp_growth": "gdp_growth_pct",
    "gdp_per_capita": "gdp_per_capita",
    "Density\\n(P/Km2)": "population_density_per_km2",
    "Land Area(Km2)": "land_area_km2",
    "Latitude": "latitude",
    "Longitude": "longitude",
}

METRIC_OPTIONS = {
    "Access to electricity (% of population)": "electricity_access_pct",
    "Access to clean fuels for cooking (%)": "clean_fuels_pct",
    "Low-carbon electricity (% of electricity)": "low_carbon_electricity_pct",
    "Renewable energy share (% of final energy)": "renewable_energy_share_pct",
    "Renewable electricity capacity per capita": "renewable_capacity_pc",
    "Energy intensity (MJ per $2017 PPP GDP)": "energy_intensity_mj_per_gdp",
    "Primary energy use per person (kWh/person)": "energy_consumption_pc_kwh",
    "Estimated CO₂ emissions per person (t/person)": "co2_tonnes_per_person_est",
    "Total CO₂ emissions (kt)": "co2_emissions_kt",
    "GDP per capita": "gdp_per_capita",
}

PERCENT_COLUMNS = {
    "electricity_access_pct",
    "clean_fuels_pct",
    "low_carbon_electricity_pct",
    "renewable_energy_share_pct",
    "renewables_equiv_primary_energy_pct",
    "gdp_growth_pct",
}

# ============================================================
# STYLING
# ============================================================
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.3rem;
        padding-bottom: 2rem;
    }
    .dashboard-title {
        font-size: 2.15rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
        color: #111827;
    }
    .dashboard-subtitle {
        font-size: 1.02rem;
        color: #4B5563;
        margin-bottom: 1.1rem;
        max-width: 1100px;
        line-height: 1.45;
    }
    .kpi-card {
        background: #F8FAFC;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        min-height: 112px;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 650;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #111827;
    }
    .kpi-note {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 0.2rem;
    }
    .section-note {
        color: #4B5563;
        font-size: 0.9rem;
        margin-top: -0.45rem;
        margin-bottom: 0.6rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA LOADING AND PREPARATION
# ============================================================
@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}. Put global-data-on-sustainable-energy.csv "
            "in the same folder as streamlit_app.py."
        )

    df = pd.read_csv(DATA_FILE)
    missing_cols = [col for col in COLUMN_RENAME if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in dataset: {missing_cols}")

    df = df.rename(columns=COLUMN_RENAME)

    numeric_cols = [col for col in df.columns if col not in ["country"]]
    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
            .replace({"nan": np.nan, "None": np.nan, "": np.nan})
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["year"] = df["year"].astype("Int64")

    # Estimated population is used only for bubble size and estimated CO₂ per person.
    # The dataset provides density and land area, not a direct annual population variable.
    df["population_est"] = df["population_density_per_km2"] * df["land_area_km2"]
    df.loc[df["population_est"] <= 0, "population_est"] = np.nan

    df["co2_tonnes_per_person_est"] = (
        df["co2_emissions_kt"] * 1000 / df["population_est"]
    )

    df["total_electricity_twh"] = (
        df[["fossil_twh", "nuclear_twh", "renewables_twh"]]
        .fillna(0)
        .sum(axis=1)
    )
    df.loc[df["total_electricity_twh"] <= 0, "total_electricity_twh"] = np.nan

    df["renewables_generation_share_pct"] = (
        df["renewables_twh"] / df["total_electricity_twh"] * 100
    )
    df["fossil_generation_share_pct"] = (
        df["fossil_twh"] / df["total_electricity_twh"] * 100
    )
    df["nuclear_generation_share_pct"] = (
        df["nuclear_twh"] / df["total_electricity_twh"] * 100
    )

    # Simple GDP per capita grouping for exploration. Labels are descriptive, not official income classes.
    gdp_bins = [-np.inf, 2_000, 10_000, 25_000, 50_000, np.inf]
    gdp_labels = [
        "Very low GDP/capita",
        "Low GDP/capita",
        "Middle GDP/capita",
        "High GDP/capita",
        "Very high GDP/capita",
    ]
    df["gdp_per_capita_group"] = pd.cut(
        df["gdp_per_capita"],
        bins=gdp_bins,
        labels=gdp_labels,
    )

    return df


def format_value(value, metric_col=None):
    if pd.isna(value):
        return "No data"
    if metric_col in PERCENT_COLUMNS:
        return f"{value:,.1f}%"
    if metric_col == "financial_flows_usd":
        return f"${value/1_000_000:,.1f}M"
    if metric_col == "co2_emissions_kt":
        return f"{value/1_000:,.1f}M kt"
    if metric_col == "population_est":
        return f"{value/1_000_000:,.1f}M"
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:,.1f}M"
    if abs(value) >= 10_000:
        return f"{value:,.0f}"
    return f"{value:,.1f}"


def clean_for_metric(data: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    return data.dropna(subset=[metric_col]).copy()


def apply_common_layout(fig, height=430, show_legend=True):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=70, b=35),
        font=dict(family="Arial", size=13, color="#111827"),
        title=dict(font=dict(size=17, color="#111827"), x=0.02, xanchor="left"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            title="",
        ),
        showlegend=show_legend,
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    return fig


try:
    data = load_data()
except Exception as exc:
    st.error(str(exc))
    st.stop()

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.title("Dashboard controls")
st.sidebar.caption(
    "Use the filters to explore geography, time, country comparisons, and the energy-development relationship."
)

year_min = int(data["year"].min())
year_max = int(data["year"].max())
selected_year = st.sidebar.slider(
    "Snapshot year",
    min_value=year_min,
    max_value=year_max,
    value=year_max,
    step=1,
    help="Updates the map, KPI cards, ranking, scatter plot, and distribution view.",
)

all_countries = sorted(data["country"].dropna().unique().tolist())
default_countries = [
    country for country in ["Norway", "United States", "China", "India", "Germany", "Lebanon"]
    if country in all_countries
]
selected_countries = st.sidebar.multiselect(
    "Countries for time-series comparison",
    options=all_countries,
    default=default_countries,
    help="These countries update the trend line and heatmap.",
)
if not selected_countries:
    selected_countries = default_countries[:3]

focus_country = st.sidebar.selectbox(
    "Focus country for drill-down",
    options=all_countries,
    index=all_countries.index("Norway") if "Norway" in all_countries else 0,
    help="Highlights one country in the scatter plot and drives the electricity generation mix chart.",
)

map_metric_label = st.sidebar.selectbox(
    "Map metric",
    options=list(METRIC_OPTIONS.keys()),
    index=list(METRIC_OPTIONS.keys()).index("Low-carbon electricity (% of electricity)"),
)
map_metric_col = METRIC_OPTIONS[map_metric_label]

trend_metric_label = st.sidebar.selectbox(
    "Time-series metric",
    options=[
        "Access to electricity (% of population)",
        "Access to clean fuels for cooking (%)",
        "Low-carbon electricity (% of electricity)",
        "Renewable energy share (% of final energy)",
        "Estimated CO₂ emissions per person (t/person)",
    ],
    index=2,
)
trend_metric_col = METRIC_OPTIONS[trend_metric_label]

ranking_metric_label = st.sidebar.selectbox(
    "Ranking metric",
    options=list(METRIC_OPTIONS.keys()),
    index=list(METRIC_OPTIONS.keys()).index("Renewable energy share (% of final energy)"),
)
ranking_metric_col = METRIC_OPTIONS[ranking_metric_label]

top_n = st.sidebar.slider("Number of countries in ranking", 5, 20, 10)

available_groups = [
    str(group) for group in data["gdp_per_capita_group"].dropna().unique().tolist()
]
selected_gdp_groups = st.sidebar.multiselect(
    "GDP per capita groups",
    options=available_groups,
    default=available_groups,
    help="Optional filter for the snapshot-year views.",
)

# ============================================================
# FILTERED SNAPSHOT DATA
# ============================================================
year_data = data[data["year"] == selected_year].copy()
if selected_gdp_groups:
    year_data = year_data[
        year_data["gdp_per_capita_group"].astype(str).isin(selected_gdp_groups)
        | year_data["gdp_per_capita_group"].isna()
    ]

# ============================================================
# TITLE AND INTRO
# ============================================================
st.markdown(
    "<div class='dashboard-title'>Global Sustainable Energy Indicators Dashboard</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='dashboard-subtitle'>This interactive dashboard explores how electricity access, "
    "clean fuels, low-carbon electricity, renewable energy, emissions, GDP, and energy intensity vary "
    "across countries from 2000 to 2020. The dashboard is designed for exploratory analysis: filters "
    "change multiple views at once, while the focus country provides a drill-down view of generation mix.</div>",
    unsafe_allow_html=True,
)

# ============================================================
# KPI CARDS
# ============================================================
st.subheader(f"Snapshot summary for {selected_year}")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

avg_electricity = year_data["electricity_access_pct"].mean()
avg_clean = year_data["clean_fuels_pct"].mean()
avg_low_carbon = year_data["low_carbon_electricity_pct"].mean()
median_co2_pc = year_data["co2_tonnes_per_person_est"].median()
valid_countries = year_data["country"].nunique()

with kpi1:
    st.markdown(
        f"<div class='kpi-card'><div class='kpi-label'>Countries with data</div>"
        f"<div class='kpi-value'>{valid_countries}</div>"
        f"<div class='kpi-note'>after current filters</div></div>",
        unsafe_allow_html=True,
    )
with kpi2:
    st.markdown(
        f"<div class='kpi-card'><div class='kpi-label'>Average electricity access</div>"
        f"<div class='kpi-value'>{format_value(avg_electricity, 'electricity_access_pct')}</div>"
        f"<div class='kpi-note'>mean across countries</div></div>",
        unsafe_allow_html=True,
    )
with kpi3:
    st.markdown(
        f"<div class='kpi-card'><div class='kpi-label'>Average low-carbon electricity</div>"
        f"<div class='kpi-value'>{format_value(avg_low_carbon, 'low_carbon_electricity_pct')}</div>"
        f"<div class='kpi-note'>renewables + nuclear share</div></div>",
        unsafe_allow_html=True,
    )
with kpi4:
    st.markdown(
        f"<div class='kpi-card'><div class='kpi-label'>Median estimated CO₂/person</div>"
        f"<div class='kpi-value'>{format_value(median_co2_pc, 'co2_tonnes_per_person_est')}</div>"
        f"<div class='kpi-note'>tonnes per person, estimated</div></div>",
        unsafe_allow_html=True,
    )

st.divider()

# ============================================================
# VISUAL 1 + VISUAL 2
# ============================================================
left_col, right_col = st.columns([1.15, 1])

with left_col:
    map_df = clean_for_metric(year_data, map_metric_col)
    if map_df.empty:
        st.warning("No data available for the selected map metric and filters.")
    else:
        range_color = (0, 100) if map_metric_col in PERCENT_COLUMNS else None
        fig_map = px.choropleth(
            map_df,
            locations="country",
            locationmode="country names",
            color=map_metric_col,
            hover_name="country",
            hover_data={
                "country": False,
                map_metric_col: ":.2f",
                "electricity_access_pct": ":.1f",
                "clean_fuels_pct": ":.1f",
                "low_carbon_electricity_pct": ":.1f",
                "gdp_per_capita": ":,.0f",
            },
            color_continuous_scale="Viridis",
            range_color=range_color,
            projection="natural earth",
            title=f"1. Global geographic pattern: {map_metric_label} ({selected_year})",
        )
        fig_map.update_geos(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#CBD5E1",
            landcolor="#F8FAFC",
        )
        fig_map.update_coloraxes(colorbar_title=map_metric_label)
        fig_map = apply_common_layout(fig_map, height=520, show_legend=False)
        st.plotly_chart(fig_map, use_container_width=True)
        st.caption(
            "Question answered: Which countries stand out geographically for the selected sustainable-energy indicator?"
        )

with right_col:
    ranking_df = clean_for_metric(year_data, ranking_metric_col)
    if ranking_df.empty:
        st.warning("No data available for the selected ranking metric and filters.")
    else:
        top_df = (
            ranking_df[["country", ranking_metric_col]]
            .sort_values(ranking_metric_col, ascending=False)
            .head(top_n)
            .sort_values(ranking_metric_col, ascending=True)
        )
        fig_rank = px.bar(
            top_df,
            x=ranking_metric_col,
            y="country",
            orientation="h",
            color=ranking_metric_col,
            color_continuous_scale="Blues",
            text=ranking_metric_col,
            title=f"2. Top {top_n} countries: {ranking_metric_label} ({selected_year})",
        )
        fig_rank.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
        fig_rank.update_layout(coloraxis_showscale=False)
        fig_rank.update_xaxes(title=ranking_metric_label, showgrid=True, gridcolor="#E5E7EB")
        fig_rank.update_yaxes(title="")
        fig_rank = apply_common_layout(fig_rank, height=520, show_legend=False)
        st.plotly_chart(fig_rank, use_container_width=True)
        st.caption(
            "Question answered: Which countries rank highest on the selected indicator in the chosen year?"
        )

st.divider()

# ============================================================
# VISUAL 3 + VISUAL 4
# ============================================================
left_col, right_col = st.columns(2)

with left_col:
    scatter_df = year_data.dropna(
        subset=["gdp_per_capita", "co2_tonnes_per_person_est", "population_est", "low_carbon_electricity_pct"]
    ).copy()
    if scatter_df.empty:
        st.warning("No data available for the GDP, CO₂, population, and low-carbon scatter plot.")
    else:
        # Limit extreme bubble sizes by keeping size positive and relying on Plotly sizemax.
        scatter_df["population_est"] = scatter_df["population_est"].clip(lower=1)
        fig_scatter = px.scatter(
            scatter_df,
            x="gdp_per_capita",
            y="co2_tonnes_per_person_est",
            size="population_est",
            color="low_carbon_electricity_pct",
            hover_name="country",
            hover_data={
                "gdp_per_capita": ":,.0f",
                "co2_tonnes_per_person_est": ":.2f",
                "population_est": ":,.0f",
                "low_carbon_electricity_pct": ":.1f",
            },
            size_max=48,
            color_continuous_scale="Viridis",
            log_x=True,
            title=f"3. Development, emissions, population, and low-carbon electricity ({selected_year})",
        )
        focus_row = scatter_df[scatter_df["country"] == focus_country]
        if not focus_row.empty:
            fig_scatter.add_trace(
                go.Scatter(
                    x=focus_row["gdp_per_capita"],
                    y=focus_row["co2_tonnes_per_person_est"],
                    mode="markers+text",
                    marker=dict(
                        size=18,
                        symbol="star",
                        color="#111827",
                        line=dict(color="white", width=1.2),
                    ),
                    text=focus_row["country"],
                    textposition="top center",
                    name=f"Focus: {focus_country}",
                    hoverinfo="skip",
                )
            )
        fig_scatter.update_xaxes(title="GDP per capita (log scale)", showgrid=True, gridcolor="#E5E7EB")
        fig_scatter.update_yaxes(title="Estimated CO₂ emissions (t/person)", showgrid=True, gridcolor="#E5E7EB")
        fig_scatter.update_coloraxes(colorbar_title="Low-carbon<br>electricity (%)")
        fig_scatter = apply_common_layout(fig_scatter, height=510, show_legend=True)
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption(
            "Question answered: How do national income, estimated emissions per person, population size, and low-carbon electricity relate?"
        )

with right_col:
    trend_df = data[data["country"].isin(selected_countries)].dropna(subset=[trend_metric_col]).copy()
    if trend_df.empty:
        st.warning("No data available for the selected countries and time-series metric.")
    else:
        fig_line = px.line(
            trend_df,
            x="year",
            y=trend_metric_col,
            color="country",
            markers=True,
            color_discrete_sequence=OKABE_ITO,
            hover_data={trend_metric_col: ":.2f", "year": False},
            title=f"4. Country trends over time: {trend_metric_label}",
        )
        fig_line.update_xaxes(title="Year", dtick=2, showgrid=True, gridcolor="#E5E7EB")
        fig_line.update_yaxes(title=trend_metric_label, showgrid=True, gridcolor="#E5E7EB")
        if trend_metric_col in PERCENT_COLUMNS:
            fig_line.update_yaxes(range=[0, 105])
        fig_line = apply_common_layout(fig_line, height=510, show_legend=True)
        st.plotly_chart(fig_line, use_container_width=True)
        st.caption(
            "Question answered: How has the selected indicator changed for the comparison countries between 2000 and 2020?"
        )

st.divider()

# ============================================================
# VISUAL 5 + VISUAL 6
# ============================================================
left_col, right_col = st.columns([1, 1.1])

with left_col:
    mix_df = data[data["country"] == focus_country][
        ["year", "fossil_twh", "nuclear_twh", "renewables_twh"]
    ].copy()
    mix_long = mix_df.melt(
        id_vars="year",
        value_vars=["fossil_twh", "nuclear_twh", "renewables_twh"],
        var_name="source",
        value_name="electricity_twh",
    )
    mix_long["source"] = mix_long["source"].map(
        {
            "fossil_twh": "Fossil fuels",
            "nuclear_twh": "Nuclear",
            "renewables_twh": "Renewables",
        }
    )
    mix_long = mix_long.dropna(subset=["electricity_twh"])
    if mix_long.empty:
        st.warning(f"No electricity generation mix data available for {focus_country}.")
    else:
        fig_area = px.area(
            mix_long,
            x="year",
            y="electricity_twh",
            color="source",
            color_discrete_map=GENERATION_COLOURS,
            title=f"5. Electricity generation mix over time: {focus_country}",
        )
        fig_area.update_xaxes(title="Year", dtick=2, showgrid=True, gridcolor="#E5E7EB")
        fig_area.update_yaxes(title="Electricity generation (TWh)", showgrid=True, gridcolor="#E5E7EB")
        fig_area = apply_common_layout(fig_area, height=500, show_legend=True)
        st.plotly_chart(fig_area, use_container_width=True)
        st.caption(
            "Question answered: Is the focus country relying more on fossil, nuclear, or renewable electricity over time?"
        )

with right_col:
    heat_df = data[data["country"].isin(selected_countries)].copy()
    heat_pivot = heat_df.pivot_table(
        index="country",
        columns="year",
        values="low_carbon_electricity_pct",
        aggfunc="mean",
    )
    if heat_pivot.empty:
        st.warning("No low-carbon electricity data available for the selected countries.")
    else:
        heat_pivot = heat_pivot.reindex(selected_countries)
        fig_heat = go.Figure(
            data=go.Heatmap(
                z=heat_pivot.values,
                x=heat_pivot.columns.astype(str),
                y=heat_pivot.index,
                colorscale="Viridis",
                zmin=0,
                zmax=100,
                colorbar=dict(title="Low-carbon<br>electricity (%)"),
                hovertemplate="Country: %{y}<br>Year: %{x}<br>Low-carbon electricity: %{z:.1f}%<extra></extra>",
            )
        )
        fig_heat.update_layout(
            title="6. Low-carbon electricity transition heatmap for selected countries",
            xaxis_title="Year",
            yaxis_title="Country",
        )
        fig_heat = apply_common_layout(fig_heat, height=500, show_legend=False)
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption(
            "Question answered: Which selected countries move toward higher low-carbon electricity shares over time?"
        )

st.divider()

# ============================================================
# VISUAL 7
# ============================================================
st.subheader("7. Energy efficiency context by GDP per capita group")
st.markdown(
    "<div class='section-note'>Lower energy intensity means less primary energy is used per unit of GDP. "
    "This view helps compare efficiency patterns across broad GDP-per-capita groups for the selected year.</div>",
    unsafe_allow_html=True,
)
box_df = year_data.dropna(subset=["gdp_per_capita_group", "energy_intensity_mj_per_gdp"]).copy()
if box_df.empty:
    st.warning("No energy intensity data available for the selected year and GDP filters.")
else:
    fig_box = px.box(
        box_df,
        x="gdp_per_capita_group",
        y="energy_intensity_mj_per_gdp",
        points="outliers",
        category_orders={
            "gdp_per_capita_group": [
                "Very low GDP/capita",
                "Low GDP/capita",
                "Middle GDP/capita",
                "High GDP/capita",
                "Very high GDP/capita",
            ]
        },
        title=f"7. Distribution of energy intensity by GDP-per-capita group ({selected_year})",
    )
    fig_box.update_traces(marker=dict(color="#0072B2", opacity=0.65), line=dict(color="#111827"))
    fig_box.update_xaxes(title="GDP per capita group")
    fig_box.update_yaxes(title="Energy intensity (MJ per $2017 PPP GDP)", showgrid=True, gridcolor="#E5E7EB")
    fig_box = apply_common_layout(fig_box, height=470, show_legend=False)
    st.plotly_chart(fig_box, use_container_width=True)
    st.caption(
        "Question answered: How does energy intensity vary across broad economic groups in the selected year?"
    )

# ============================================================
# DATA AND LIMITATIONS NOTE
# ============================================================
with st.expander("Data notes and limitations"):
    st.markdown(
        """
        - The dataset covers global sustainable energy indicators from 2000 to 2020.
        - Some variables contain missing values, especially financial flows and renewable equivalent primary energy.
        - CO₂ emissions are provided in kilotonnes by country. Estimated CO₂ per person is calculated as: emissions in tonnes divided by estimated population.
        - Estimated population is calculated from population density multiplied by land area because the dataset does not provide a direct annual population column.
        - GDP-per-capita groups are simple dashboard groupings for exploration and are not official World Bank income categories.
        - The dashboard shows associations and patterns; it does not prove causal relationships.
        """
    )

st.caption("Source: global-data-on-sustainable-energy.csv")
