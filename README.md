# Enterprise AI Decision Support System

An AI-powered decision support app with an Apple-style interface. You upload your own business data, it is stored in a
local SQLite database on your computer, and the app gives you dashboards, predictions, forecasts and recommendations.
Everything can be downloaded.

## 1. Set up (Windows, macOS, Linux)

Python 3.10 or newer is required.

```bash
cd edss
python -m venv .venv
# Windows:   .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

Optional PyTorch neural network: `pip install torch` (it then appears in the Models page).

## 2. How to use it

1. **Data > Upload**: drop a CSV or Excel (.xlsx) file, tick the cleaning options, press *Save to local database*.
2. **Data > Set up**: choose the numeric *target* (for example revenue), the *date column*, and segment columns
   (region, product). Press *Save settings*.
3. **Overview**: KPIs, trend, segments, correlations and the AI briefing (trends, risks, opportunities).
4. **Forecast**: pick a model and how many periods ahead. Error is measured on recent periods the model did not see.
5. **Models**: train several algorithms, compare MAE, RMSE and R2 on held-out data, see the drivers, save a model,
   predict one case with a form, or predict a whole file.
6. **Recommendations**: actions based on your data. Tick the ones you want in your action plan and download it.
7. **Export**: dataset (CSV, Excel), the whole database file, data dictionary, report (HTML, Markdown), saved models.

## 3. Where your data lives

| What | Where |
|---|---|
| Datasets, settings, forecasts | `data/decision_support.db` (SQLite, tables `ds_<id>`) |
| Trained models | `models/*.joblib` |

Nothing is sent to the internet. Open the database in any SQLite tool, or use the SQL console in
Data > Library and SQL (read-only SELECT queries).

## 4. Technology used

| Requirement | Used in |
|---|---|
| Python, Pandas, NumPy | loading, cleaning, aggregation, statistics (`app.py`, `ml.py`, `insights.py`) |
| Scikit-learn | pipelines, Linear/Ridge/Random Forest/Gradient Boosting/MLP, metrics (`ml.py`) |
| PyTorch (optional) | `TorchMLPRegressor` in `ml.py` |
| SQL | SQLite storage and the SQL console (`db.py`) |
| Streamlit | the interface and dashboards (`app.py`) |
| Git/GitHub | `.gitignore` keeps your data and models out of the repository |

Power BI or Tableau can connect to the same data: export the CSV or point them at `decision_support.db`.

## 5. Mapping to the project deliverables

| Deliverable | Where to get it |
|---|---|
| Source code | this folder |
| Dataset and data dictionary | Export page |
| AI/ML models | Models page, saved to `models/` |
| Working decision-support app and dashboard | `streamlit run app.py` |
| Model evaluation report | Export page (report) and the Models page table |
| Technical documentation | this README |
| Project report, presentation, demo video | you create these; record the demo by walking through the pages in order |

## 6. Good to know

- Forecasting needs at least 12 time periods. If you have fewer, choose a finer time step (weekly or daily).
- An incomplete first or last period (for example a half-finished month) is left out of totals and trends.
- Insights are rule-based on your data and models. They support a decision; they do not replace judgement.
- Tested with Python 3.12, Streamlit 1.65, pandas 3.0 and scikit-learn 1.8.

## 7. Files

```
app.py          interface and pages
db.py           local SQLite storage, SQL console, model files
ml.py           preprocessing, models, evaluation, forecasting
insights.py     trends, risks, opportunities, recommendations, reports
requirements.txt
.streamlit/config.toml   theme and upload limit
```
