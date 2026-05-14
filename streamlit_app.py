import re
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
    page_title="Global Sustainable Energy Indicators Dashboard",
    page_icon="🌍",
    layout="wide",
)

DATA_FILE = "global-data-on-sustainable-energy.csv"


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }

        h1, h2, h3 {
            letter-spacing: -0.02em;
        }

        .small-note {
            color: #64748b;
            font-size: 0.88rem;
            line-height: 1.35;
        }

        .kpi-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 18px 18px;
            min-height: 118px;
        }

        .kpi-label {
            color: #64748b;
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }

        .kpi-value {
            color: #0f172a;
            font-size: 1.65rem;
            font-weight: 800;
            line-height: 1.1;
        }

        .kpi-help {
            color: #64748b;
            font-size: 0.78rem;
            margin-top: 0.45rem;
        }

        div[data-testid="stSidebar"] {
            background-color: #f1f5f9;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

OKABE_BLUE = "#0072B2"
OKABE_ORANGE = "#E69F00"
OKABE_GREEN = "#009E73"
OKABE_PURPLE = "#CC79A7"
OKABE_GREY = "#999999"
DARK_TEXT = "#0f172a"
MUTED_TEXT = "#64748b"

PLOT_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
}

AGGREGATE_ENTITIES = {
    "World",
    "Africa",
    "Asia",
    "Europe",
    "European Union",
    "North America",
    "South America",
    "Oceania",
    "High income",
    "Low income",
    "Lower middle income",
    "Upper middle income",
    "Middle income",
    "Low and middle income",
    "Least developed countries",
    "Small island developing states",
}

COLUMN_CANDIDATES = {
    "entity": [
        "Entity",
        "Country",
        "Country Name",
        "country",
    ],
    "year": [
        "Year",
        "year",
    ],
    "electricity_access_pct": [
        "Access to electricity (% of population)",
        "Access to electricity (%)",
        "Access to electricity",
    ],
    "clean_fuels_access_pct": [
        "Access to clean fuels for cooking (% of population)",
        "Access to clean fuels (% of population)",
        "Access to clean fuels (%)",
        "Access to clean fuels",
    ],
    "renewable_capacity_per_capita": [
        "Renewable-electricity-generating-capacity-per-capita",
        "Renewable electricity capacity per capita",
        "Renewable electricity generating capacity per capita",
    ],
    "financial_flows_usd": [
        "Financial flows to developing countries (US $)",
        "Financial flows (USD)",
        "Financial flows",
    ],
    "renewable_energy_share_pct": [
        "Renewable energy share in the total final energy consumption (%)",
        "Renewable energy share (% of final energy)",
        "Renewable energy share (%)",
        "Renewable energy share",
    ],
    "electricity_fossil_twh": [
        "Electricity from fossil fuels (TWh)",
        "Electricity from fossil fuels",
    ],
    "electricity_nuclear_twh": [
        "Electricity from nuclear (TWh)",
        "Electricity from nuclear",
    ],
    "electricity_renewables_twh": [
        "Electricity from renewables (TWh)",
        "Electricity from renewables",
    ],
    "low_carbon_electricity_pct": [
        "Low-carbon electricity (% electricity)",
        "Low carbon electricity (% electricity)",
        "Low-carbon electricity (%)",
        "Low carbon electricity (%)",
    ],
    "primary_energy_per_capita_kwh": [
        "Primary energy consumption per capita (kWh/person)",
        "Primary energy consumption per capita",
    ],
    "energy_intensity": [
        "Energy intensity level of primary energy (MJ/$2017 PPP GDP)",
        "Energy intensity level of primary energy (MJ/$2011 PPP GDP)",
        "Energy intensity level of primary energy",
        "Energy intensity",
    ],
    "co2_total_kt": [
        "Value_co2_emissions_kt_by_country",
        "Value co2 emissions kt by country",
        "CO2 emissions kt",
        "CO₂ emissions kt",
    ],
    "co2_direct_tonnes_per_capita": [
        "Value co2 emissions (metric tons per capita)",
        "CO2 emissions (metric tons per capita)",
        "CO₂ emissions (metric tons per capita)",
    ],
    "renewables_equivalent_primary_energy": [
        "Renewables (% equivalent primary energy)",
        "Renewables",
    ],
    "gdp_growth_pct": [
        "gdp_growth",
        "GDP growth (%)",
        "GDP growth",
    ],
    "gdp_per_capita": [
        "gdp_per_capita",
        "GDP per capita",
    ],
    "population_density_per_km2": [
        "Density\\n(P/Km2)",
        "Density\n(P/Km2)",
        "Density (P/Km2)",
        "Density(P/Km2)",
        "Population density (P/Km2)",
        "Population density (P/Km²)",
        "Population density",
    ],
    "land_area_km2": [
        "Land Area(Km2)",
        "Land Area (Km2)",
        "Land area (Km2)",
        "Land area (Km²)",
        "Land area",
    ],
    "latitude": [
        "Latitude",
        "latitude",
    ],
    "longitude": [
        "Longitude",
        "longitude",
    ],
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalise_name(name: str) -> str:
    """Normalise column names so that spaces, newlines, brackets, and symbols do not break matching."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_column(df: pd.DataFrame, possible_names: list[str]) -> str | None:
    """Find a dataset column using exact and normalised matching."""
    columns = list(df.columns)

    for name in possible_names:
        if name in columns:
            return name

    normalised_lookup = {normalise_name(col): col for col in columns}

    for name in possible_names:
        key = normalise_name(name)
        if key in normalised_lookup:
            return normalised_lookup[key]

    for col in columns:
        col_key = normalise_name(col)
        for name in possible_names:
            name_key = normalise_name(name)
            if name_key and (name_key in col_key or col_key in name_key):
                return col

    return None


def clean_numeric(series: pd.Series) -> pd.Series:
    """Convert columns with commas, symbols, or text formatting into numeric values."""
    cleaned = (
        series.astype(str)
        .str.replace("\u00a0", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
        .str.replace(r"[^0-9eE+\-.]", "", regex=True)
    )

    cleaned = cleaned.replace(
        {
            "": np.nan,
            "nan": np.nan,
            "None": np.nan,
            "null": np.nan,
        }
    )

    return pd.to_numeric(cleaned, errors="coerce")


def format_value(value, suffix="", decimals=1):
    if pd.isna(value):
        return "No data"
    return f"{value:,.{decimals}f}{suffix}"


def make_kpi(label, value, help_text):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plotly_selectable_chart(fig, key: str):
    """
    Uses Streamlit's built-in Plotly selection when available.
    If the installed Streamlit version is old, the chart still displays,
    but linked clicking will not work until Streamlit is upgraded.
    """
    try:
        return st.plotly_chart(
            fig,
            use_container_width=True,
            key=key,
            on_select="rerun",
            selection_mode="points",
            config=PLOT_CONFIG,
        )
    except TypeError:
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=key,
            config=PLOT_CONFIG,
        )
        return None


def extract_country_from_selection(event, valid_countries: set[str]) -> str | None:
    """Extract selected country name from Plotly selection event."""
    if event is None:
        return None

    try:
        selection = event.selection
    except Exception:
        selection = None

    if selection is None and isinstance(event, dict):
        selection = event.get("selection", None)

    if selection is None:
        return None

    if isinstance(selection, dict):
        points = selection.get("points", [])
    else:
        points = getattr(selection, "points", [])

    if not points:
        return None

    point = points[0]

    if not isinstance(point, dict):
        try:
            point = dict(point)
        except Exception:
            return None

    candidates = []

    customdata = point.get("customdata", None)
    if isinstance(customdata, (list, tuple)) and len(customdata) > 0:
        candidates.append(customdata[0])
    elif isinstance(customdata, str):
        candidates.append(customdata)

    for key in ["location", "hovertext", "text", "label", "y", "x"]:
        if key in point:
            candidates.append(point.get(key))

    for item in candidates:
        if isinstance(item, str) and item in valid_countries:
            return item

    return None


def update_focus_country(event, source_name: str, valid_countries: set[str]):
    """
    Updates the internal focus country when a country is clicked in a visual.

    Important:
    We update st.session_state["focus_country"], NOT the dropdown widget key.
    The dropdown widget key is "focus_country_dropdown".
    """
    clicked_country = extract_country_from_selection(event, valid_countries)

    if clicked_country and clicked_country in valid_countries:
        current_country = st.session_state.get("focus_country")

        if clicked_country != current_country:
            st.session_state["focus_country"] = clicked_country
            st.toast(f"Focus country changed to {clicked_country} from {source_name}.")
            st.rerun()


def no_data_message(message: str):
    st.info(
        f"Not enough complete data is available for this visual. {message} "
        "Try changing the year, metric, focus country, or filters."
    )


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    path = Path(DATA_FILE)

    if not path.exists():
        st.error(
            f"Could not find `{DATA_FILE}`. Put the CSV file in the same folder as `streamlit_app.py`."
        )
        st.stop()

    raw = pd.read_csv(path)
    raw.columns = [str(col).strip() for col in raw.columns]

    rename_map = {}

    for new_name, candidates in COLUMN_CANDIDATES.items():
        found = find_column(raw, candidates)
        if found is not None:
            rename_map[found] = new_name

    df = raw.rename(columns=rename_map).copy()

    required = ["entity", "year"]
    missing_required = [col for col in required if col not in df.columns]

    if missing_required:
        st.error(
            "Missing required columns after column matching: "
            f"{missing_required}. Available columns are: {list(raw.columns)}"
        )
        st.stop()

    df["entity"] = df["entity"].astype(str).str.strip()
    df["year"] = clean_numeric(df["year"])
    df = df.dropna(subset=["entity", "year"]).copy()
    df["year"] = df["year"].astype(int)

    numeric_cols = [
        col for col in df.columns
        if col not in ["entity"]
    ]

    for col in numeric_cols:
        if col != "year":
            df[col] = clean_numeric(df[col])

    # Derived population estimate from population density and land area
    if {"population_density_per_km2", "land_area_km2"}.issubset(df.columns):
        df["estimated_population"] = (
            df["population_density_per_km2"] * df["land_area_km2"]
        )
        df["estimated_population_millions"] = df["estimated_population"] / 1_000_000
    else:
        df["estimated_population"] = np.nan
        df["estimated_population_millions"] = np.nan

    # CO2 per person
    if "co2_direct_tonnes_per_capita" in df.columns:
        df["co2_person_tonnes"] = df["co2_direct_tonnes_per_capita"]
        co2_source = "dataset CO₂/person column"
    elif "co2_total_kt" in df.columns:
        df["co2_person_tonnes"] = np.where(
            (df["estimated_population"] > 0) & df["co2_total_kt"].notna(),
            (df["co2_total_kt"] * 1000) / df["estimated_population"],
            np.nan,
        )
        co2_source = "estimated from total CO₂ and population"
    else:
        df["co2_person_tonnes"] = np.nan
        co2_source = "not available"

    # Total electricity generation
    generation_cols = [
        "electricity_fossil_twh",
        "electricity_nuclear_twh",
        "electricity_renewables_twh",
    ]

    for col in generation_cols:
        if col not in df.columns:
            df[col] = np.nan

    df["electricity_total_twh"] = df[generation_cols].sum(axis=1, min_count=1)

    # GDP groups for filtering
    if "gdp_per_capita" in df.columns:
        conditions = [
            df["gdp_per_capita"] < 2_000,
            (df["gdp_per_capita"] >= 2_000) & (df["gdp_per_capita"] < 10_000),
            (df["gdp_per_capita"] >= 10_000) & (df["gdp_per_capita"] < 25_000),
            df["gdp_per_capita"] >= 25_000,
        ]
        labels = [
            "Very low GDP/capita",
            "Low GDP/capita",
            "Middle GDP/capita",
            "High GDP/capita",
        ]
        df["gdp_group"] = np.select(conditions, labels, default="Missing GDP/capita")
    else:
        df["gdp_group"] = "Missing GDP/capita"

    df["is_aggregate_entity"] = df["entity"].isin(AGGREGATE_ENTITIES)

    return df, list(raw.columns), co2_source


df, raw_columns, co2_source = load_data()


# ============================================================
# METRIC REGISTRY
# ============================================================

all_metric_options = {
    "Electricity access (% population)": "electricity_access_pct",
    "Clean fuels access (% population)": "clean_fuels_access_pct",
    "Renewable energy share (% final energy)": "renewable_energy_share_pct",
    "Low-carbon electricity (% electricity)": "low_carbon_electricity_pct",
    "CO₂/person (tonnes)": "co2_person_tonnes",
    "GDP per capita": "gdp_per_capita",
    "Energy intensity (MJ/$ PPP GDP)": "energy_intensity",
    "Primary energy use/person (kWh/person)": "primary_energy_per_capita_kwh",
    "Renewable capacity/person": "renewable_capacity_per_capita",
    "Financial flows for clean energy (USD)": "financial_flows_usd",
}

available_metric_options = {
    label: col
    for label, col in all_metric_options.items()
    if col in df.columns and df[col].notna().any()
}

if not available_metric_options:
    st.error("No usable numeric indicator columns were found.")
    st.stop()


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.title("Dashboard controls")

st.sidebar.markdown(
    """
    <div class="small-note">
    Use the filters to explore geography, time, country comparisons, and the relationship between
    energy access, low-carbon electricity, emissions, GDP, and energy intensity.
    </div>
    """,
    unsafe_allow_html=True,
)

exclude_aggregates = st.sidebar.checkbox(
    "Exclude regional aggregates",
    value=True,
    help="Keeps the dashboard focused on country-level comparison.",
)

working_df = df[~df["is_aggregate_entity"]].copy() if exclude_aggregates else df.copy()

all_countries = sorted(working_df["entity"].dropna().unique().tolist())
valid_country_set = set(all_countries)

min_year = int(working_df["year"].min())
max_year = int(working_df["year"].max())
default_year = 2019 if 2019 in working_df["year"].unique() else max_year

snapshot_year = st.sidebar.slider(
    "Snapshot year",
    min_value=min_year,
    max_value=max_year,
    value=default_year,
    help="This year controls the map, ranking chart, scatter plot, and KPI cards.",
)

default_focus_candidates = ["Norway", "Lebanon", "United States", "China", "India", "Germany"]
default_focus = next(
    (country for country in default_focus_candidates if country in all_countries),
    all_countries[0],
)

# -------------------------------
# Focus-country state setup
# -------------------------------

if "focus_country" not in st.session_state or st.session_state["focus_country"] not in all_countries:
    st.session_state["focus_country"] = default_focus

# The dropdown must use a different key from the internal linked-selection variable.
# This avoids the Streamlit error:
# "st.session_state.focus_country cannot be modified after the widget with key focus_country is instantiated."
if "focus_country_dropdown" not in st.session_state or st.session_state["focus_country_dropdown"] not in all_countries:
    st.session_state["focus_country_dropdown"] = st.session_state["focus_country"]

# Keep the dropdown visually synced with the current focus country.
# This is safe because it happens BEFORE the dropdown widget is created.
if st.session_state["focus_country_dropdown"] != st.session_state["focus_country"]:
    st.session_state["focus_country_dropdown"] = st.session_state["focus_country"]


def sync_focus_country_from_dropdown():
    st.session_state["focus_country"] = st.session_state["focus_country_dropdown"]


st.sidebar.selectbox(
    "Focus country for drill-down",
    options=all_countries,
    key="focus_country_dropdown",
    on_change=sync_focus_country_from_dropdown,
    help="This country controls the generation-mix and access-over-time visuals. Clicking a country in a chart also updates this.",
)

focus_country = st.session_state["focus_country"]

default_compare = [
    country for country in ["Norway", "United States", "China", "India", "Germany", focus_country]
    if country in all_countries
]

countries_for_trend = st.sidebar.multiselect(
    "Countries for time-series comparison",
    options=all_countries,
    default=list(dict.fromkeys(default_compare)),
    help="These countries appear in the trend chart.",
)

if focus_country not in countries_for_trend:
    countries_for_trend = [focus_country] + countries_for_trend

country_scope = st.sidebar.multiselect(
    "Optional country filter for map, ranking, and scatter",
    options=all_countries,
    default=[],
    help="Leave empty to show all countries. Select countries here if you want a narrower comparison.",
)

gdp_group_order = [
    "Very low GDP/capita",
    "Low GDP/capita",
    "Middle GDP/capita",
    "High GDP/capita",
    "Missing GDP/capita",
]

available_gdp_groups = [
    group for group in gdp_group_order
    if group in working_df["gdp_group"].dropna().unique()
]

selected_gdp_groups = st.sidebar.multiselect(
    "GDP per capita groups",
    options=available_gdp_groups,
    default=available_gdp_groups,
    help="Applies to the snapshot visuals: map, ranking, and scatter.",
)

def metric_index(label):
    labels = list(available_metric_options.keys())
    return labels.index(label) if label in labels else 0

map_metric_label = st.sidebar.selectbox(
    "Map metric",
    options=list(available_metric_options.keys()),
    index=metric_index("Low-carbon electricity (% electricity)"),
)

time_metric_label = st.sidebar.selectbox(
    "Time-series metric",
    options=list(available_metric_options.keys()),
    index=metric_index("Low-carbon electricity (% electricity)"),
)

ranking_metric_label = st.sidebar.selectbox(
    "Ranking metric",
    options=list(available_metric_options.keys()),
    index=metric_index("Renewable energy share (% final energy)"),
)

top_n = st.sidebar.slider(
    "Number of countries in ranking",
    min_value=5,
    max_value=20,
    value=10,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div class="small-note">
    <b>Linked interaction:</b><br>
    Click a country in the map, ranking chart, scatter plot, or trend chart.
    The focus country updates automatically, and the country-level visuals below change with it.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FILTER DATA
# ============================================================

snapshot_df = working_df[working_df["year"] == snapshot_year].copy()

if selected_gdp_groups:
    snapshot_df = snapshot_df[snapshot_df["gdp_group"].isin(selected_gdp_groups)]

if country_scope:
    snapshot_df = snapshot_df[snapshot_df["entity"].isin(country_scope)]

map_metric_col = available_metric_options[map_metric_label]
time_metric_col = available_metric_options[time_metric_label]
ranking_metric_col = available_metric_options[ranking_metric_label]


# ============================================================
# HEADER
# ============================================================

st.title("Global Sustainable Energy Indicators Dashboard")

st.markdown(
    f"""
    This interactive dashboard explores sustainable-energy indicators from **{min_year} to {max_year}**.
    It supports country-level exploration across electricity access, clean fuels, renewable energy,
    low-carbon electricity, emissions, GDP, and energy intensity.
    """
)

st.caption(
    "Design note: the dashboard uses filters, linked country selection, shared visual encodings, and focus-country drill-down views to support exploratory analysis."
)


# ============================================================
# KPI CARDS
# ============================================================

st.subheader(f"Snapshot summary for {snapshot_year}")

countries_with_data = snapshot_df["entity"].nunique()
avg_electricity_access = snapshot_df["electricity_access_pct"].mean() if "electricity_access_pct" in snapshot_df.columns else np.nan
avg_low_carbon = snapshot_df["low_carbon_electricity_pct"].mean() if "low_carbon_electricity_pct" in snapshot_df.columns else np.nan
median_co2_person = snapshot_df["co2_person_tonnes"].median() if "co2_person_tonnes" in snapshot_df.columns else np.nan

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    make_kpi(
        "Countries with data",
        f"{countries_with_data:,}",
        "after current filters",
    )

with kpi2:
    make_kpi(
        "Average electricity access",
        format_value(avg_electricity_access, "%", 1),
        "mean across countries",
    )

with kpi3:
    make_kpi(
        "Average low-carbon electricity",
        format_value(avg_low_carbon, "%", 1),
        "renewables + nuclear share",
    )

with kpi4:
    make_kpi(
        "Median CO₂/person",
        format_value(median_co2_person, "", 2),
        f"tonnes/person; {co2_source}",
    )

st.divider()


# ============================================================
# VISUAL 1: CHOROPLETH MAP
# ============================================================

left_col, right_col = st.columns([1.1, 1])

with left_col:
    st.markdown(
        f"### 1. Geographic pattern: {map_metric_label} ({snapshot_year})"
    )

    map_df = snapshot_df.dropna(subset=[map_metric_col]).copy()

    if map_df.empty:
        no_data_message("The selected map metric has no valid values for the current year and filters.")
    else:
        fig_map = px.choropleth(
            map_df,
            locations="entity",
            locationmode="country names",
            color=map_metric_col,
            hover_name="entity",
            custom_data=["entity", map_metric_col, "gdp_group"],
            color_continuous_scale="Blues",
            labels={map_metric_col: map_metric_label},
        )

        fig_map.update_layout(
            height=470,
            margin=dict(l=0, r=0, t=20, b=0),
            coloraxis_colorbar=dict(
                title=map_metric_label,
                thickness=14,
            ),
            geo=dict(
                showframe=False,
                showcoastlines=False,
                projection_type="natural earth",
            ),
        )

        fig_map.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"{map_metric_label}: " + "%{customdata[1]:,.2f}<br>"
                "GDP group: %{customdata[2]}<extra></extra>"
            )
        )

        map_event = plotly_selectable_chart(fig_map, key="map_chart")
        update_focus_country(map_event, "the map", valid_country_set)

    st.caption(
        "Question answered: Which countries stand out geographically for the selected sustainable-energy indicator?"
    )


# ============================================================
# VISUAL 2: RANKING BAR CHART
# ============================================================

with right_col:
    st.markdown(
        f"### 2. Top {top_n} countries: {ranking_metric_label} ({snapshot_year})"
    )

    ranking_df = (
        snapshot_df.dropna(subset=[ranking_metric_col])
        .sort_values(ranking_metric_col, ascending=False)
        .head(top_n)
        .sort_values(ranking_metric_col, ascending=True)
        .copy()
    )

    if ranking_df.empty:
        no_data_message("The ranking metric has no valid values for the current year and filters.")
    else:
        fig_rank = px.bar(
            ranking_df,
            x=ranking_metric_col,
            y="entity",
            orientation="h",
            custom_data=["entity", ranking_metric_col],
            labels={
                ranking_metric_col: ranking_metric_label,
                "entity": "",
            },
        )

        fig_rank.update_traces(
            marker_color=OKABE_BLUE,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"{ranking_metric_label}: " + "%{customdata[1]:,.2f}<extra></extra>"
            ),
        )

        fig_rank.update_layout(
            height=470,
            margin=dict(l=10, r=20, t=20, b=40),
            xaxis_title=ranking_metric_label,
            yaxis_title="",
            plot_bgcolor="white",
        )

        rank_event = plotly_selectable_chart(fig_rank, key="ranking_chart")
        update_focus_country(rank_event, "the ranking chart", valid_country_set)

    st.caption(
        "Question answered: Which countries rank highest on the selected indicator in the chosen year?"
    )

st.divider()


# ============================================================
# VISUAL 3: SCATTER PLOT
# ============================================================

scatter_left, scatter_right = st.columns([1.1, 1])

with scatter_left:
    st.markdown(
        f"### 3. Development, emissions, and low-carbon electricity ({snapshot_year})"
    )

    scatter_required = [
        "gdp_per_capita",
        "co2_person_tonnes",
        "low_carbon_electricity_pct",
    ]

    scatter_df = snapshot_df.dropna(subset=scatter_required).copy()

    if "estimated_population_millions" not in scatter_df.columns:
        scatter_df["estimated_population_millions"] = np.nan

    if scatter_df["estimated_population_millions"].notna().sum() == 0:
        scatter_df["bubble_size"] = 8
    else:
        median_pop = scatter_df["estimated_population_millions"].median()
        scatter_df["bubble_size"] = scatter_df["estimated_population_millions"].fillna(median_pop)

    scatter_df = scatter_df[scatter_df["gdp_per_capita"] > 0]

    if len(scatter_df) < 5:
        no_data_message(
            "The scatter plot needs GDP per capita, CO₂/person, and low-carbon electricity values."
        )
    else:
        fig_scatter = px.scatter(
            scatter_df,
            x="gdp_per_capita",
            y="co2_person_tonnes",
            color="low_carbon_electricity_pct",
            size="bubble_size",
            hover_name="entity",
            custom_data=[
                "entity",
                "gdp_per_capita",
                "co2_person_tonnes",
                "low_carbon_electricity_pct",
            ],
            color_continuous_scale="Blues",
            size_max=24,
            log_x=True,
            labels={
                "gdp_per_capita": "GDP per capita",
                "co2_person_tonnes": "CO₂/person (tonnes)",
                "low_carbon_electricity_pct": "Low-carbon electricity (%)",
            },
        )

        fig_scatter.update_layout(
            height=500,
            margin=dict(l=10, r=20, t=20, b=45),
            plot_bgcolor="white",
            coloraxis_colorbar=dict(
                title="Low-carbon<br>electricity (%)",
                thickness=14,
            ),
        )

        fig_scatter.update_traces(
            opacity=0.78,
            marker=dict(line=dict(width=0.6, color="white")),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "GDP/capita: %{customdata[1]:,.0f}<br>"
                "CO₂/person: %{customdata[2]:,.2f} tonnes<br>"
                "Low-carbon electricity: %{customdata[3]:,.1f}%<extra></extra>"
            ),
        )

        scatter_event = plotly_selectable_chart(fig_scatter, key="scatter_chart")
        update_focus_country(scatter_event, "the scatter plot", valid_country_set)

    st.caption(
        "Question answered: How do income level, CO₂/person, population size, and low-carbon electricity relate to each other?"
    )


# ============================================================
# VISUAL 4: TIME-SERIES COMPARISON
# ============================================================

with scatter_right:
    st.markdown(
        f"### 4. Country trends over time: {time_metric_label}"
    )

    trend_df = working_df[
        working_df["entity"].isin(countries_for_trend)
    ].dropna(subset=[time_metric_col]).copy()

    if trend_df.empty:
        no_data_message("The selected countries do not have enough values for the chosen time-series metric.")
    else:
        fig_trend = px.line(
            trend_df,
            x="year",
            y=time_metric_col,
            color="entity",
            markers=True,
            custom_data=["entity", time_metric_col],
            labels={
                "year": "Year",
                time_metric_col: time_metric_label,
                "entity": "Country",
            },
            color_discrete_sequence=[
                OKABE_BLUE,
                OKABE_ORANGE,
                OKABE_GREEN,
                OKABE_PURPLE,
                OKABE_GREY,
                "#56B4E9",
                "#D55E00",
            ],
        )

        fig_trend.add_vline(
            x=snapshot_year,
            line_dash="dash",
            line_color="#94a3b8",
            annotation_text=f"{snapshot_year}",
            annotation_position="top",
        )

        for trace in fig_trend.data:
            if trace.name == focus_country:
                trace.line.width = 4
                trace.marker.size = 8
            else:
                trace.line.width = 2
                trace.marker.size = 6
                trace.opacity = 0.72

        fig_trend.update_layout(
            height=500,
            margin=dict(l=10, r=20, t=20, b=45),
            plot_bgcolor="white",
            legend_title_text="Country",
        )

        fig_trend.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Year: %{x}<br>"
                f"{time_metric_label}: " + "%{customdata[1]:,.2f}<extra></extra>"
            )
        )

        trend_event = plotly_selectable_chart(fig_trend, key="trend_chart")
        update_focus_country(trend_event, "the trend chart", valid_country_set)

    st.caption(
        "Question answered: How has the selected indicator changed over time for the comparison countries?"
    )

st.divider()


# ============================================================
# FOCUS COUNTRY SECTION
# ============================================================

st.subheader(f"Focus country drill-down: {focus_country}")

focus_df = working_df[working_df["entity"] == focus_country].copy()

focus_snapshot = focus_df[focus_df["year"] == snapshot_year].copy()

focus_cols = st.columns(4)

if focus_snapshot.empty:
    with focus_cols[0]:
        make_kpi("Selected year", str(snapshot_year), "no focus-country row for this year")
    with focus_cols[1]:
        make_kpi("Electricity access", "No data", "selected year")
    with focus_cols[2]:
        make_kpi("Low-carbon electricity", "No data", "selected year")
    with focus_cols[3]:
        make_kpi("CO₂/person", "No data", "selected year")
else:
    row = focus_snapshot.iloc[0]

    with focus_cols[0]:
        make_kpi(
            "Selected year",
            str(snapshot_year),
            "focus-country snapshot",
        )

    with focus_cols[1]:
        make_kpi(
            "Electricity access",
            format_value(row.get("electricity_access_pct", np.nan), "%", 1),
            "share of population",
        )

    with focus_cols[2]:
        make_kpi(
            "Low-carbon electricity",
            format_value(row.get("low_carbon_electricity_pct", np.nan), "%", 1),
            "renewables + nuclear share",
        )

    with focus_cols[3]:
        make_kpi(
            "CO₂/person",
            format_value(row.get("co2_person_tonnes", np.nan), "", 2),
            "tonnes per person",
        )


# ============================================================
# VISUAL 5: FOCUS COUNTRY ELECTRICITY MIX
# ============================================================

mix_col, access_col = st.columns([1.1, 1])

with mix_col:
    st.markdown(
        f"### 5. Electricity generation mix over time: {focus_country}"
    )

    generation_cols = [
        "electricity_fossil_twh",
        "electricity_nuclear_twh",
        "electricity_renewables_twh",
    ]

    mix_df = focus_df[["year"] + generation_cols].copy()
    mix_df = mix_df.dropna(subset=generation_cols, how="all")

    if mix_df.empty:
        no_data_message(f"{focus_country} has no complete generation-mix values.")
    else:
        mix_long = mix_df.melt(
            id_vars="year",
            value_vars=generation_cols,
            var_name="source",
            value_name="electricity_twh",
        ).dropna(subset=["electricity_twh"])

        source_labels = {
            "electricity_fossil_twh": "Fossil fuels",
            "electricity_nuclear_twh": "Nuclear",
            "electricity_renewables_twh": "Renewables",
        }

        mix_long["source"] = mix_long["source"].map(source_labels)

        color_map = {
            "Fossil fuels": OKABE_GREY,
            "Nuclear": OKABE_PURPLE,
            "Renewables": OKABE_GREEN,
        }

        fig_mix = px.area(
            mix_long,
            x="year",
            y="electricity_twh",
            color="source",
            labels={
                "year": "Year",
                "electricity_twh": "Electricity generation (TWh)",
                "source": "Source",
            },
            color_discrete_map=color_map,
        )

        fig_mix.add_vline(
            x=snapshot_year,
            line_dash="dash",
            line_color="#94a3b8",
        )

        fig_mix.update_layout(
            height=470,
            margin=dict(l=10, r=20, t=20, b=45),
            plot_bgcolor="white",
            legend_title_text="Electricity source",
        )

        fig_mix.update_traces(
            hovertemplate=(
                "Year: %{x}<br>"
                "Electricity: %{y:,.2f} TWh<extra></extra>"
            )
        )

        st.plotly_chart(fig_mix, use_container_width=True, config=PLOT_CONFIG)

    st.caption(
        "Question answered: How has the selected country generated electricity from fossil fuels, nuclear, and renewables over time?"
    )


# ============================================================
# VISUAL 6: FOCUS COUNTRY ACCESS TREND
# ============================================================

with access_col:
    st.markdown(
        f"### 6. Energy access over time: {focus_country}"
    )

    access_vars = [
        "electricity_access_pct",
        "clean_fuels_access_pct",
    ]

    access_available = [
        col for col in access_vars
        if col in focus_df.columns and focus_df[col].notna().any()
    ]

    if not access_available:
        no_data_message(f"{focus_country} has no electricity-access or clean-fuels access values.")
    else:
        access_long = focus_df[["year"] + access_available].melt(
            id_vars="year",
            value_vars=access_available,
            var_name="indicator",
            value_name="value",
        ).dropna(subset=["value"])

        access_labels = {
            "electricity_access_pct": "Electricity access",
            "clean_fuels_access_pct": "Clean fuels access",
        }

        access_long["indicator"] = access_long["indicator"].map(access_labels)

        fig_access = px.line(
            access_long,
            x="year",
            y="value",
            color="indicator",
            markers=True,
            labels={
                "year": "Year",
                "value": "Population access (%)",
                "indicator": "Indicator",
            },
            color_discrete_map={
                "Electricity access": OKABE_BLUE,
                "Clean fuels access": OKABE_ORANGE,
            },
        )

        fig_access.add_vline(
            x=snapshot_year,
            line_dash="dash",
            line_color="#94a3b8",
        )

        fig_access.update_yaxes(range=[0, 105])

        fig_access.update_layout(
            height=470,
            margin=dict(l=10, r=20, t=20, b=45),
            plot_bgcolor="white",
            legend_title_text="Access indicator",
        )

        fig_access.update_traces(
            hovertemplate=(
                "Year: %{x}<br>"
                "Access: %{y:,.1f}%<extra></extra>"
            )
        )

        st.plotly_chart(fig_access, use_container_width=True, config=PLOT_CONFIG)

    st.caption(
        "Question answered: Has the selected country improved access to electricity and clean fuels over time?"
    )

st.divider()


# ============================================================
# DATA COVERAGE SECTION
# ============================================================

st.subheader("Data coverage and interpretation notes")

coverage_col1, coverage_col2 = st.columns([1, 1])

with coverage_col1:
    st.markdown(
        f"""
        - Current snapshot year: **{snapshot_year}**
        - Current focus country: **{focus_country}**
        - Countries in snapshot after filters: **{countries_with_data:,}**
        - CO₂/person source: **{co2_source}**
        """
    )

with coverage_col2:
    st.markdown(
        """
        - Blank visuals are avoided by showing an explanatory message when data is missing.
        - The map, ranking, scatter, and trend chart support linked country selection.
        - The dashboard should be interpreted as exploratory, not causal.
        - Missing values and unequal country coverage may affect comparisons.
        """
    )

with st.expander("Show original dataset columns"):
    st.write(raw_columns)

st.caption(
    "Source: global-data-on-sustainable-energy.csv. Dashboard created for UC3DVS10 Task 2."
)
