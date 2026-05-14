# Task 2: Sustainable Energy Indicators Dashboard

## Files
- `streamlit_app.py` — main Streamlit dashboard script
- `global-data-on-sustainable-energy.csv` — dataset
- `requirements.txt` — Python packages needed to run the dashboard

## How to run locally
1. Put all files in the same folder.
2. Open a terminal in that folder.
3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Run the dashboard:

```bash
streamlit run streamlit_app.py
```

## Dashboard theme
The dashboard explores global sustainable energy indicators from 2000 to 2020. It supports filtering by year, country, focus country, ranking metric, map metric, trend metric, and GDP-per-capita group.

## Visualisations included
1. Choropleth map for global geographic patterns
2. Ranked bar chart for top countries by selected indicator
3. Bubble scatter plot linking GDP, CO₂, population, and low-carbon electricity
4. Time-series line chart for selected countries
5. Electricity generation mix area chart for a focus country
6. Low-carbon electricity transition heatmap
7. Energy intensity distribution by GDP-per-capita group

## Data preparation notes
- Column names were simplified in the script.
- Numeric fields were converted to numeric values.
- Population was estimated from density multiplied by land area.
- Estimated CO₂ per person was calculated from total CO₂ emissions and estimated population.
- GDP-per-capita groups were created for exploratory comparison.
