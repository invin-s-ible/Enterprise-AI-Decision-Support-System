"""Enterprise AI Decision Support System: Streamlit app with an Apple-style interface.

Run with:  streamlit run app.py
"""
import csv
import hmac
import html
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import db
import insights as ins
import ml

st.set_page_config(page_title="Insight · Decision Support", page_icon="📈", layout="wide")
db.init_db()

PAGES = ["Overview", "Data", "Forecast", "Models", "Recommendations", "Export"]
BLUE, INK, GREEN, ORANGE = "#F5424F", "#1C1B3A", "#FFA33A", "#FF7A45"  # BLUE is the red accent
FONT = "Montserrat, -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif"
KIND_STYLE = {
    "Risk": ("#FFE1E3", "#B3202C"),
    "Opportunity": ("#DDF3E4", "#16622E"),
    "Trend": ("#FFE9D1", "#9A4400"),
    "Driver": ("#E4E3F7", "#2D2B6B"),
    "Info": ("#ECECEF", "#3A3A4A"),
}

# ----------------------------------------------------------------- Apple look
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap');
:root {
  --bg: #ECECEF;
  --ink: #1C1B3A;
  --muted: #6B6B78;
  --dark: #24262B;
  --grad: linear-gradient(100deg, #FFA33A 0%, #FA4E49 62%, #F8384A 100%);
  --grad-red: linear-gradient(100deg, #FF5C5C 0%, #F42B2B 100%);
  --sh-d: rgba(163,163,180,0.55);
  --sh-l: rgba(255,255,255,0.95);
  --raised: 8px 8px 18px var(--sh-d), -8px -8px 18px var(--sh-l);
  --raised-sm: 5px 5px 12px var(--sh-d), -5px -5px 12px var(--sh-l);
  --inset: inset 4px 4px 9px var(--sh-d), inset -4px -4px 9px var(--sh-l);
}
html, body, .stApp, [class*="css"] {
  font-family: "Montserrat", -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", system-ui, sans-serif;
  color: var(--ink);
}
.stApp { background: var(--bg); }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2.4rem; max-width: 1280px; }
h1 { font-weight: 700 !important; letter-spacing: -0.02em !important; font-size: 2.4rem !important; color: var(--ink); }
h2, h3, h4 { font-weight: 600 !important; letter-spacing: -0.01em !important; color: var(--ink); }

/* Sidebar: raised soft panel */
[data-testid="stSidebar"] {
  background: var(--bg);
  border-right: 0;
  box-shadow: 10px 0 26px rgba(163,163,180,0.35);
}
[data-testid="stSidebar"] [role="radiogroup"] { gap: 10px; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  border-radius: 999px; padding: 10px 20px; min-height: 46px; width: 100%; align-items: center;
  background: var(--bg); box-shadow: var(--raised-sm); cursor: pointer;
}
[data-testid="stSidebar"] [role="radiogroup"] label p { color: var(--muted); font-weight: 600; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: var(--grad); box-shadow: 0 8px 18px rgba(248,56,74,0.35);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p { color: #FFFFFF; }
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display: none; }

/* Hide the round radio dots in the sidebar menu */
[data-testid="stSidebar"] [role="radiogroup"] label > span:not(:has([data-testid="stMarkdownContainer"])),
[data-testid="stSidebar"] [role="radiogroup"] label > div:not([data-testid="stMarkdownContainer"]):not(:has([data-testid="stMarkdownContainer"])) {
  display: none !important;
}
/* Captions must stay readable, including in the sidebar */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: var(--muted) !important; }
/* Keep the kit font on text, but never on icons */
.stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp label, .stApp li, .stApp button,
.stApp input, .stApp textarea, .stApp td, .stApp th,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
  font-family: "Montserrat", -apple-system, "Segoe UI", system-ui, sans-serif !important;
}

/* Metric tiles: raised */
[data-testid="stMetric"] {
  background: var(--bg); border-radius: 24px; padding: 20px 24px; box-shadow: var(--raised);
}
[data-testid="stMetricLabel"] p { color: var(--muted); font-weight: 500; }
[data-testid="stMetricValue"] { font-weight: 700; letter-spacing: -0.02em; color: var(--ink); }

/* Buttons: light embossed pills, gradient for primary */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button,
[data-testid="stBaseButton-secondary"] {
  border-radius: 999px; min-height: 46px; padding: 0 26px; font-weight: 600; border: 0;
  background: linear-gradient(145deg, #F3F3F6, #E6E6EA); color: var(--muted);
  box-shadow: var(--raised-sm);
}
.stButton > button:hover, .stDownloadButton > button:hover { color: var(--ink); box-shadow: var(--raised); }
.stButton > button:active, .stDownloadButton > button:active { box-shadow: var(--inset); }
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"],
[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"] {
  background: var(--grad); color: #FFFFFF; box-shadow: 0 10px 20px rgba(248,56,74,0.32);
}
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover { color: #FFFFFF; filter: brightness(1.05); }
.stButton > button[kind="primary"] p, [data-testid="stBaseButton-primary"] p { color: #FFFFFF; }

/* Tabs: pressed track, gradient selected pill */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg); border-radius: 999px; padding: 6px; gap: 4px; width: fit-content; box-shadow: var(--inset);
}
.stTabs [data-baseweb="tab"] { border-radius: 999px; height: 42px; padding: 0 22px; font-weight: 600; color: var(--muted); }
.stTabs [aria-selected="true"] { background: var(--grad); color: #FFFFFF; box-shadow: 0 6px 14px rgba(248,56,74,0.3); }
.stTabs [aria-selected="true"] p { color: #FFFFFF; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* Inputs: white soft pills */
div[data-baseweb="input"], div[data-baseweb="base-input"], div[data-baseweb="textarea"] {
  background: #FFFFFF !important; border: 0 !important; border-radius: 16px !important; box-shadow: var(--raised-sm);
}
div[data-baseweb="input"] input, div[data-baseweb="base-input"] input, div[data-baseweb="textarea"] textarea {
  background: transparent !important; color: var(--ink) !important;
}
div[data-baseweb="input"]:focus-within, div[data-baseweb="textarea"]:focus-within { outline: 2px solid rgba(250,78,73,0.55); }
[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
  background: #FFFFFF; border: 0; border-radius: 16px; box-shadow: var(--raised-sm);
}
[data-baseweb="tag"] { background: var(--grad) !important; border-radius: 999px !important; color: #FFFFFF !important; }
[data-baseweb="tag"] * { color: #FFFFFF !important; }

/* Single select: orange to red pill, like the Select button in the kit */
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
  background: var(--grad); border: 0; border-radius: 999px; box-shadow: 0 10px 20px rgba(248,56,74,0.28); min-height: 46px;
}
[data-testid="stSelectbox"] [data-baseweb="select"] * { color: #FFFFFF !important; fill: #FFFFFF !important; }

/* Dropdown list: dark navy panel with a red highlighted option */
[data-baseweb="popover"] > div { border-radius: 20px !important; overflow: hidden; box-shadow: 0 18px 40px rgba(28,27,58,0.35); }
[data-baseweb="popover"] ul[role="listbox"], [data-baseweb="popover"] [data-baseweb="menu"] { background: #1C1B3A !important; padding: 8px; }
[data-baseweb="popover"] li { background: transparent !important; color: #FFFFFF !important; border-radius: 12px; }
[data-baseweb="popover"] li * { color: #FFFFFF !important; }
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] li[aria-selected="true"] { background: #D9444C !important; }

/* Upload area, charts, tables, panels */
[data-testid="stFileUploader"] section {
  border-radius: 24px; border: 2px dashed rgba(245,66,79,0.55); background: var(--bg); box-shadow: var(--inset);
}
[data-testid="stPlotlyChart"] {
  background: linear-gradient(145deg, #F7F7F9, #E9E9EC); border-radius: 26px; padding: 14px; box-shadow: var(--raised);
}
[data-testid="stDataFrame"] { border-radius: 22px; overflow: hidden; box-shadow: var(--raised-sm); }
[data-testid="stExpander"] details { background: var(--bg); border: 0; border-radius: 22px; box-shadow: var(--raised-sm); }
[data-testid="stAlert"] { border-radius: 20px; box-shadow: var(--raised-sm); }
[data-testid="stForm"] { border: 0; border-radius: 26px; box-shadow: var(--raised); background: var(--bg); }
hr { border-color: rgba(163,163,180,0.35); }

/* Dark insight panel, like the dark list card in the kit */
.card {
  background: var(--dark); border-radius: 24px; padding: 22px 26px; margin-bottom: 16px;
  box-shadow: 0 16px 32px rgba(28,27,58,0.28); color: #FFFFFF;
}
.card h4 { margin: 10px 0 6px; font-size: 18px; color: #FFFFFF; }
.card .muted { color: #B8B8C4; }
.card .muted b { color: #FFFFFF; }
.chip { display: inline-block; font-size: 12px; font-weight: 700; padding: 4px 12px; border-radius: 999px; }
.muted { color: var(--muted); font-size: 14px; line-height: 1.55; }
.brand { display: flex; align-items: center; gap: 12px; padding: 4px 4px 22px; }
.brand-mark {
  width: 42px; height: 42px; border-radius: 50%; background: var(--grad); color: #fff; font-weight: 700;
  display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 16px rgba(248,56,74,0.35);
}
.brand-name { font-size: 20px; font-weight: 700; letter-spacing: -0.01em; color: var(--ink); }
.sub { color: var(--muted); font-size: 16px; margin-top: -0.4rem; margin-bottom: 1.4rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------- optional password gate
# Set the APP_PASSWORD environment variable (for example on Render) to protect a public link.
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
if APP_PASSWORD and not st.session_state.get("_authed"):
    st.title("Sign in")
    pw = st.text_input("Password", type="password")
    if st.button("Open", type="primary"):
        if hmac.compare_digest(pw, APP_PASSWORD):
            st.session_state["_authed"] = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()


# --------------------------------------------------------------------- helpers
def style_fig(fig, height=380):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK),
        margin=dict(l=10, r=10, t=30, b=10),
        height=height,
        colorway=[BLUE, INK, GREEN, ORANGE, "#6C63FF", "#8E8E99"],
        legend=dict(orientation="h", y=1.1, x=0),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#D9D9DF", zeroline=False)
    return fig


def header(title, subtitle):
    st.title(title)
    st.markdown(f'<div class="sub">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def card(chip_kind, headline, body, extra=""):
    bg, fg = KIND_STYLE.get(chip_kind, KIND_STYLE["Info"])
    st.markdown(
        f'<div class="card"><span class="chip" style="background:{bg};color:{fg}">{html.escape(chip_kind)}</span>'
        f"<h4>{html.escape(headline)}</h4><div class='muted'>{html.escape(body)}</div>{extra}</div>",
        unsafe_allow_html=True,
    )


def go_to(page):
    st.session_state["page"] = page


def empty_state(message, button_label="Open Data", target_page="Data"):
    st.markdown(f'<div class="card"><h4>Nothing here yet</h4><div class="muted">{html.escape(message)}</div></div>', unsafe_allow_html=True)
    st.button(button_label, type="primary", on_click=go_to, args=(target_page,), key=f"goto_{target_page}_{button_label}")


def get_cfg(meta):
    cfg = {"date_col": None, "target": None, "agg": "sum", "freq": "MS", "segments": [], "descriptions": {}}
    cfg.update(meta.get("config", {}))
    return cfg


@st.cache_data(show_spinner=False)
def load_df(ds_id, token):
    return db.load_dataset(ds_id)


def read_upload(up):
    name = up.name.lower()
    if name.endswith(".xlsx"):
        xl = pd.ExcelFile(up)
        sheet = xl.sheet_names[0]
        if len(xl.sheet_names) > 1:
            sheet = st.selectbox("Sheet", xl.sheet_names)
        return xl.parse(sheet)
    sample = up.read(8192)
    up.seek(0)
    try:
        sep = csv.Sniffer().sniff(sample.decode("utf-8", errors="ignore"), delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    try:
        return pd.read_csv(up, sep=sep, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        up.seek(0)
        return pd.read_csv(up, sep=sep, encoding="latin-1", low_memory=False)


def clean_df(raw, tidy, dedupe, date_cols, dayfirst):
    df = raw.copy()
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=dayfirst)
    names = []
    for i, c in enumerate(df.columns):
        n = str(c).strip() if tidy else str(c)
        if not n:
            n = f"column_{i + 1}"
        base, j = n, 2
        while n in names:
            n = f"{base}_{j}"
            j += 1
        names.append(n)
    rename = dict(zip(df.columns, names))
    df.columns = names
    if dedupe:
        df = df.drop_duplicates().reset_index(drop=True)
    return df, [rename[c] for c in date_cols]


def to_excel_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    return buf.getvalue()


def csv_bytes(df, index=False):
    return df.to_csv(index=index).encode("utf-8")


# --------------------------------------------------------------------- sidebar
if "_ds_pending" in st.session_state:
    st.session_state["ds_select"] = st.session_state.pop("_ds_pending")

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">I</div><div class="brand-name">Insight</div></div>', unsafe_allow_html=True)
    if "page" not in st.session_state:
        st.session_state["page"] = "Overview" if len(db.list_datasets()) else "Data"
    page = st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)
    datasets = db.list_datasets()
    ds_id = None
    if len(datasets):
        ids = [int(i) for i in datasets["id"].tolist()]
        names = {i: f"{n} (#{i})" for i, n in zip(ids, datasets["name"].tolist())}
        if st.session_state.get("ds_select") not in ids:
            st.session_state["ds_select"] = ids[0]
        ds_id = st.selectbox("Active dataset", ids, key="ds_select", format_func=lambda i: names[i])
    else:
        st.caption("No data yet. Open Data to upload your first file.")
    st.markdown(
        '<div class="muted" style="margin-top:24px">Your data stays on this computer, in a local SQLite database.</div>',
        unsafe_allow_html=True,
    )

meta = db.get_dataset(ds_id) if ds_id else None
df = load_df(ds_id, meta["created_at"]) if meta else None
cfg = get_cfg(meta) if meta else None


# ------------------------------------------------------------------- Data page
def page_data():
    header("Data", "Upload your own business data, keep it in a local database, and describe it.")
    tab_up, tab_lib, tab_setup, tab_dict = st.tabs(["Upload", "Library and SQL", "Set up", "Dictionary"])

    with tab_up:
        up = st.file_uploader("Drop a CSV or Excel (.xlsx) file", type=["csv", "xlsx"])
        if up is not None:
            try:
                raw = read_upload(up)
            except Exception as exc:
                st.error(f"Could not read this file: {exc}")
                raw = None
            if raw is not None:
                c1, c2, c3 = st.columns(3)
                c1.metric("Rows", f"{len(raw):,}")
                c2.metric("Columns", raw.shape[1])
                c3.metric("Missing values", f"{raw.isna().mean().mean() * 100:.1f}%")
                st.markdown("#### Prepare")
                a, b = st.columns(2)
                tidy = a.checkbox("Tidy column names (trim spaces, make unique)", True)
                dedupe = a.checkbox("Remove duplicate rows", True)
                dayfirst = b.checkbox("Dates are day first (31/12/2025)", False)
                date_cols = b.multiselect("Date columns", list(raw.columns), default=ml.detect_date_columns(raw))
                st.dataframe(raw.head(100))
                name = st.text_input("Dataset name", value=Path(up.name).stem)
                if st.button("Save to local database", type="primary"):
                    cleaned, new_dates = clean_df(raw, tidy, dedupe, date_cols, dayfirst)
                    new_id = db.save_dataset(cleaned, name or "dataset", up.name, new_dates)
                    load_df.clear()
                    st.session_state["_ds_pending"] = new_id
                    st.session_state["_flash"] = f"Saved {len(cleaned):,} rows to the local database. Next, open Set up to choose your target column."
                    st.rerun()
        if st.session_state.get("_flash"):
            st.success(st.session_state.pop("_flash"))

    with tab_lib:
        if datasets.empty:
            st.info("Saved datasets will appear here.")
        else:
            st.dataframe(datasets)
            if ds_id:
                with st.expander("Delete the active dataset"):
                    sure = st.checkbox("I understand this removes the dataset and its saved models from this computer.")
                    if st.button("Delete dataset", disabled=not sure):
                        db.delete_dataset(ds_id)
                        load_df.clear()
                        st.session_state.pop("ds_select", None)
                        st.rerun()
                st.markdown("#### SQL console")
                tables = ", ".join(f"ds_{i}" for i in datasets["id"].tolist())
                st.caption(f"Read-only SELECT queries. Tables: {tables}")
                q = st.text_area("Query", value=f"SELECT * FROM ds_{ds_id} LIMIT 100", key=f"sql_{ds_id}", height=110)
                if st.button("Run query"):
                    try:
                        res = db.run_query(q)
                        st.dataframe(res)
                        st.download_button("Download result (CSV)", csv_bytes(res), "query_result.csv", "text/csv")
                    except Exception as exc:
                        st.error(str(exc))

    with tab_setup:
        if df is None:
            st.info("Upload a dataset first.")
        else:
            numeric, dates, cats = ml.column_groups(df)
            if not numeric:
                st.warning("No numeric column found. A numeric column is needed as the target to analyse and predict.")
            c1, c2 = st.columns(2)
            target = c1.selectbox("Target to analyse and predict (numeric)", numeric, index=numeric.index(cfg["target"]) if cfg["target"] in numeric else 0) if numeric else None
            date_opts = ["(none)"] + dates
            date_col = c2.selectbox("Date column", date_opts, index=date_opts.index(cfg["date_col"]) if cfg["date_col"] in date_opts else 0)
            c3, c4 = st.columns(2)
            agg = c3.radio("Combine rows in a period by", ["sum", "mean"], index=["sum", "mean"].index(cfg["agg"]), horizontal=True)
            freq_labels = list(ml.FREQUENCIES)
            current = next((k for k, v in ml.FREQUENCIES.items() if v == cfg["freq"]), "Monthly")
            freq_label = c4.selectbox("Time step", freq_labels, index=freq_labels.index(current))
            seg_opts = ml.segment_candidates(df)
            segments = st.multiselect("Segment columns (region, product, customer type)", seg_opts, default=[s for s in cfg["segments"] if s in seg_opts] or seg_opts[:2])
            if date_col == "(none)":
                st.caption("Without a date column, trend and forecast pages are unavailable, but models and segments still work.")
            if st.button("Save settings", type="primary") and target:
                new_cfg = dict(cfg, target=target, date_col=None if date_col == "(none)" else date_col, agg=agg, freq=ml.FREQUENCIES[freq_label], segments=segments)
                db.update_config(ds_id, new_cfg)
                st.success("Settings saved. Open Overview to see your dashboard.")
                st.rerun()

    with tab_dict:
        if df is None:
            st.info("Upload a dataset first.")
        else:
            dic = ml.build_dictionary(df, cfg["descriptions"])
            st.caption("Generated from your data. Add your own descriptions and save them.")
            edited = st.data_editor(dic, disabled=[c for c in dic.columns if c != "Description"], hide_index=True, key=f"dict_{ds_id}")
            c1, c2 = st.columns([1, 1])
            if c1.button("Save descriptions"):
                desc = {r["Column"]: r["Description"] for r in edited.to_dict("records") if r["Description"]}
                db.update_config(ds_id, dict(cfg, descriptions=desc))
                st.success("Descriptions saved.")
            c2.download_button("Download data dictionary (CSV)", csv_bytes(edited), f"{meta['name']}_data_dictionary.csv", "text/csv")


# --------------------------------------------------------------- Overview page
def page_overview():
    header("Overview", "How the business is performing and what needs your attention.")
    if df is None:
        return empty_state("Upload a dataset to see your dashboard.")
    target, date_col = cfg["target"], cfg["date_col"]
    if not target:
        return empty_state("Choose a target column in Data > Set up.", "Open Data setup")
    agg_word = "total" if cfg["agg"] == "sum" else "average"
    ts = None
    if date_col:
        try:
            ts = ml.aggregate_series(df, date_col, target, cfg["freq"], cfg["agg"])
        except Exception as exc:
            st.warning(f"Could not build the time series: {exc}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Records", f"{len(df):,}")
    value = df[target].sum() if cfg["agg"] == "sum" else df[target].mean()
    k2.metric(f"{target} ({agg_word})", ins.fmt(value))
    if ts is not None and len(ts) >= 2 and ts.iloc[-2] != 0:
        pct = ts.iloc[-1] / ts.iloc[-2] - 1
        k3.metric("Latest period", ins.fmt(ts.iloc[-1]), f"{pct:+.1%} vs previous")
    else:
        k3.metric("Median value", ins.fmt(df[target].median()))
    k4.metric("Missing cells", f"{df.isna().mean().mean() * 100:.1f}%")

    if ts is not None and ts.attrs.get("dropped_partial"):
        st.caption("The first or last period is incomplete in your data, so it is left out of totals and trends.")
    t_trend, t_seg, t_corr = st.tabs(["Trend", "Segments", "Correlations"])
    with t_trend:
        if ts is not None and len(ts) >= 2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts.index, y=ts.values, name=target, mode="lines", line=dict(color=INK, width=3)))
            if len(ts) >= 3:
                slope, intercept = np.polyfit(np.arange(len(ts)), ts.values, 1)
                fig.add_trace(go.Scatter(x=ts.index, y=slope * np.arange(len(ts)) + intercept, name="Trend line", mode="lines", line=dict(color=BLUE, width=2, dash="dot")))
            st.plotly_chart(style_fig(fig))
        else:
            fig = px.histogram(df, x=target, nbins=40)
            st.plotly_chart(style_fig(fig))
            st.caption("Set a date column in Data > Set up to see the trend over time.")
    with t_seg:
        segs = [s for s in cfg["segments"] if s in df.columns] or ml.segment_candidates(df)
        if not segs:
            st.info("No segment columns found. Text columns with 2 to 50 groups work best.")
        else:
            seg = st.selectbox("Segment by", segs)
            grp = df.groupby(seg)[target].agg(cfg["agg"]).sort_values(ascending=False).head(15).reset_index()
            fig = px.bar(grp, x=target, y=seg, orientation="h")
            fig.update_traces(marker_color=BLUE)
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(style_fig(fig))
            if date_col:
                top = grp[seg].head(5).tolist()
                sub = df[df[seg].isin(top)]
                piv = sub.groupby([pd.Grouper(key=date_col, freq=cfg["freq"]), seg])[target].agg(cfg["agg"]).reset_index()
                st.plotly_chart(style_fig(px.line(piv, x=date_col, y=target, color=seg)))
    with t_corr:
        num = df.select_dtypes(include="number")
        if num.shape[1] < 2:
            st.info("At least two numeric columns are needed.")
        else:
            fig = px.imshow(num.corr(), color_continuous_scale=[INK, "#FFFFFF", BLUE], zmin=-1, zmax=1, aspect="auto")
            st.plotly_chart(style_fig(fig, 460))

    st.markdown("### AI briefing")
    trained = st.session_state.get(f"trained_{ds_id}")
    items = ins.build_insights(
        df, target, date_col, cfg["agg"], cfg["freq"], cfg["segments"],
        forecast=st.session_state.get(f"forecast_{ds_id}"),
        importance=trained["importance"] if trained else None,
    )
    cols = st.columns(min(3, len(items)))
    for i, it in enumerate(items[:6]):
        with cols[i % len(cols)]:
            card(it["kind"], it["headline"], it["detail"])
    st.caption("Insights are computed from your data. Train a model and run a forecast to add more.")


# --------------------------------------------------------------- Forecast page
def page_forecast():
    header("Forecast", "See where your numbers are heading and how sure the model is.")
    if df is None:
        return empty_state("Upload a dataset first.")
    if not cfg["target"] or not cfg["date_col"]:
        return empty_state("Forecasting needs a target and a date column. Set them in Data > Set up.", "Open Data setup")
    c1, c2, c3 = st.columns(3)
    algo = c1.selectbox("Model", ["Random Forest", "Gradient Boosting", "Linear Regression"])
    labels = list(ml.FREQUENCIES)
    cur = next((k for k, v in ml.FREQUENCIES.items() if v == cfg["freq"]), "Monthly")
    freq_label = c2.selectbox("Time step", labels, index=labels.index(cur))
    horizon = int(c3.number_input("Periods ahead", 1, 60, 6))
    key = f"forecast_{ds_id}"
    if st.button("Run forecast", type="primary"):
        try:
            with st.spinner("Training and forecasting..."):
                series = ml.aggregate_series(df, cfg["date_col"], cfg["target"], ml.FREQUENCIES[freq_label], cfg["agg"])
                st.session_state[key] = ml.forecast_series(series, ml.FREQUENCIES[freq_label], horizon, algo)
        except Exception as exc:
            st.error(str(exc))
    res = st.session_state.get(key)
    if res is None:
        st.markdown('<div class="muted">Choose a model and press Run forecast.</div>', unsafe_allow_html=True)
        return
    m = res["metrics"]
    k1, k2, k3 = st.columns(3)
    k1.metric("Backtest MAE", ins.fmt(m["MAE"]))
    k2.metric("Backtest RMSE", ins.fmt(m["RMSE"]))
    k3.metric("Backtest MAPE", "n/a" if not np.isfinite(m["MAPE"]) else f"{m['MAPE']:.1%}")
    hist, fc = res["history"], res["forecast"]
    fig = go.Figure()
    xs = list(fc.index) + list(fc.index[::-1])
    ys = list(fc["upper"]) + list(fc["lower"][::-1])
    fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself", fillcolor="rgba(245,66,79,0.14)", line=dict(width=0), name="80% range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Actual", line=dict(color=INK, width=3)))
    fig.add_trace(go.Scatter(x=[hist.index[-1]] + list(fc.index), y=[hist.iloc[-1]] + list(fc["forecast"]), name="Forecast", line=dict(color=BLUE, width=3)))
    st.plotly_chart(style_fig(fig, 430))
    st.caption(f"{res['algorithm']} using lagged values, trend and seasonality. Error measured on the most recent periods the model did not see.")
    out = fc.reset_index().rename(columns={"index": "date"})
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(out.round(2), hide_index=True)
    d1, d2 = st.columns(2)
    d1.download_button("Download forecast (CSV)", csv_bytes(out.round(4)), f"{meta['name']}_forecast.csv", "text/csv")
    if d2.button("Save forecast to local database"):
        table = db.save_table(f"forecast_{ds_id}", out)
        st.success(f"Saved as table {table}. Query it in Data > Library and SQL.")


# ----------------------------------------------------------------- Models page
def prediction_form(active):
    st.markdown("#### Predict a single case")
    feats = active["features"]
    missing = [f for f in feats if f not in df.columns]
    if missing:
        st.warning("This model was trained on columns that are not in the active dataset: " + ", ".join(missing))
        return
    with st.form("single_prediction"):
        values = {}
        cols = st.columns(3)
        for i, f in enumerate(feats):
            s, col = df[f], cols[i % 3]
            if pd.api.types.is_datetime64_any_dtype(s):
                default = s.dropna().max().date() if s.notna().any() else pd.Timestamp.today().date()
                values[f] = pd.Timestamp(col.date_input(f, value=default))
            elif pd.api.types.is_numeric_dtype(s):
                med = s.median()
                values[f] = col.number_input(f, value=0.0 if pd.isna(med) else float(med))
            else:
                options = sorted(s.dropna().astype(str).unique())[:200]
                values[f] = col.selectbox(f, options) if options else col.text_input(f)
        go_predict = st.form_submit_button("Predict", type="primary")
    if go_predict:
        pred = float(active["pipeline"].predict(pd.DataFrame([values]))[0])
        st.metric(f"Predicted {active['target']}", ins.fmt(pred))


def page_models():
    header("Models", "Train several models on your data, compare them honestly, and use the best one.")
    if df is None:
        return empty_state("Upload a dataset first.")
    if not cfg["target"]:
        return empty_state("Choose a target column in Data > Set up.", "Open Data setup")
    target = cfg["target"]
    algos = list(ml.available_models())
    c1, c2 = st.columns([2, 1])
    features = c1.multiselect("Input columns", [c for c in df.columns if c != target], default=ml.default_features(df, target))
    chosen = c2.multiselect("Algorithms", algos, default=[a for a in algos if a in ("Linear Regression", "Random Forest", "Gradient Boosting")])
    if "Neural Network (PyTorch)" not in algos:
        st.caption("Install PyTorch (pip install torch) to add a PyTorch neural network to the list.")
    key = f"trained_{ds_id}"
    if st.button("Train and compare", type="primary"):
        if not chosen:
            st.error("Choose at least one algorithm.")
        else:
            try:
                with st.spinner("Training models..."):
                    results = ml.train_models(df, target, features, chosen, cfg["date_col"])
                    best = results[0]
                    imp = ml.feature_importance(best["pipeline"], best["X_test"], best["y_test"])
                st.session_state[key] = {"results": results, "features": features, "importance": imp, "target": target}
                st.session_state[f"active_{ds_id}"] = {"pipeline": best["pipeline"], "features": features, "target": target, "algorithm": best["name"]}
            except Exception as exc:
                st.error(str(exc))
    trained = st.session_state.get(key)
    if trained:
        results = trained["results"]
        rows = pd.DataFrame(
            [{"Algorithm": r["name"], "MAE": r["metrics"]["MAE"], "RMSE": r["metrics"]["RMSE"], "R²": r["metrics"]["R2"], "MAPE": r["metrics"]["MAPE"]} for r in results]
        )
        st.session_state[f"model_rows_{ds_id}"] = rows
        st.markdown("#### Evaluation on held-out data")
        split = "the most recent 20% of rows" if cfg["date_col"] else "a random 20% of rows"
        st.caption(f"Scored on {split}, which the models never saw. Lower MAE and RMSE are better; R² closer to 1 is better. Best model: {results[0]['name']}.")
        st.dataframe(rows.round(4), hide_index=True)
        names = [r["name"] for r in results]
        pick = st.selectbox("Inspect model", names)
        r = next(x for x in results if x["name"] == pick)
        t1, t2 = st.tabs(["Actual vs predicted", "What drives it"])
        with t1:
            fig = px.scatter(x=r["y_test"], y=r["pred"], labels={"x": "Actual", "y": "Predicted"}, opacity=0.6)
            lo, hi = float(min(r["y_test"].min(), r["pred"].min())), float(max(r["y_test"].max(), r["pred"].max()))
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect", line=dict(color=INK, dash="dot")))
            fig.update_traces(marker_color=BLUE, selector=dict(mode="markers"))
            st.plotly_chart(style_fig(fig, 420))
        with t2:
            imp = trained["importance"].head(12)
            fig = px.bar(x=imp.values, y=imp.index, orientation="h", labels={"x": "Importance", "y": ""})
            fig.update_traces(marker_color=INK)
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(style_fig(fig, 400))
            st.caption("Permutation importance of the best model: how much accuracy drops when a column is shuffled.")
        s1, s2 = st.columns(2)
        if s1.button("Use this model for predictions"):
            st.session_state[f"active_{ds_id}"] = {"pipeline": r["pipeline"], "features": trained["features"], "target": target, "algorithm": r["name"]}
            st.success(f"{r['name']} is now the active model.")
        if s2.button("Save this model to my computer"):
            path = db.save_model(ds_id, r["name"], target, trained["features"], r["metrics"], r["pipeline"])
            st.success(f"Saved to {path}")

    saved = db.list_models(ds_id)
    if len(saved):
        with st.expander("Saved models"):
            st.dataframe(saved.drop(columns=["metrics"]), hide_index=True)
            pick_id = st.selectbox("Load a saved model", saved["id"].tolist(), format_func=lambda i: f"#{i} {saved.set_index('id').loc[i, 'algorithm']}")
            path = saved.set_index("id").loc[pick_id, "file_path"]
            l1, l2 = st.columns(2)
            if l1.button("Load saved model"):
                obj = db.load_model(path)
                st.session_state[f"active_{ds_id}"] = obj
                st.success(f"Loaded {obj['algorithm']}.")
            l2.download_button("Download model file", db.read_file_bytes(path), Path(path).name, "application/octet-stream")

    active = st.session_state.get(f"active_{ds_id}")
    if active:
        st.markdown("---")
        st.markdown(f"### Predictions with {active['algorithm']}")
        prediction_form(active)
        st.markdown("#### Predict a whole file")
        new_file = st.file_uploader("Upload rows with the same input columns (CSV or Excel)", type=["csv", "xlsx"], key="predict_file")
        if new_file is not None:
            try:
                new = read_upload(new_file)
                absent = [f for f in active["features"] if f not in new.columns]
                if absent:
                    st.error("Missing columns: " + ", ".join(absent))
                else:
                    for f in active["features"]:
                        if f in df.columns and pd.api.types.is_datetime64_any_dtype(df[f]):
                            new[f] = pd.to_datetime(new[f], errors="coerce")
                    new[f"predicted_{active['target']}"] = active["pipeline"].predict(new[active["features"]])
                    st.dataframe(new.head(200))
                    st.download_button("Download predictions (CSV)", csv_bytes(new), "predictions.csv", "text/csv")
            except Exception as exc:
                st.error(f"Could not predict: {exc}")


# ------------------------------------------------------------ Recommendations
def page_recommendations():
    header("Recommendations", "Actions suggested by your data, with the reasoning behind each.")
    if df is None:
        return empty_state("Upload a dataset first.")
    if not cfg["target"]:
        return empty_state("Choose a target column in Data > Set up.", "Open Data setup")
    trained = st.session_state.get(f"trained_{ds_id}")
    items = ins.build_insights(
        df, cfg["target"], cfg["date_col"], cfg["agg"], cfg["freq"], cfg["segments"],
        forecast=st.session_state.get(f"forecast_{ds_id}"),
        importance=trained["importance"] if trained else None,
    )
    if not trained or st.session_state.get(f"forecast_{ds_id}") is None:
        st.caption("Run a forecast and train a model to add forecast and driver recommendations.")
    plan = []
    for i, it in enumerate(items):
        extra = f"<div class='muted' style='margin-top:10px'><b>Recommended action:</b> {html.escape(it['action'])}</div>"
        if it["impact"]:
            extra += f"<div class='muted'><b>Impact:</b> {html.escape(it['impact'])}</div>"
        card(it["kind"], it["headline"], it["detail"], extra)
        if st.checkbox("Add to my action plan", key=f"plan_{ds_id}_{i}"):
            plan.append({"Type": it["kind"], "Finding": it["headline"], "Action": it["action"], "Impact": it["impact"]})
    if plan:
        st.markdown("### My action plan")
        plan_df = pd.DataFrame(plan)
        st.dataframe(plan_df, hide_index=True)
        p1, p2 = st.columns(2)
        p1.download_button("Download plan (CSV)", csv_bytes(plan_df), "action_plan.csv", "text/csv")
        md = "# Action plan\n\n" + "\n".join(f"- **{p['Type']}**: {p['Finding']}. {p['Action']}" for p in plan)
        p2.download_button("Download plan (Markdown)", md.encode("utf-8"), "action_plan.md", "text/markdown")


# ----------------------------------------------------------------- Export page
def page_export():
    header("Export", "Save your data, results and reports to your computer.")
    if df is None:
        return empty_state("Upload a dataset first.")
    name = meta["name"]
    st.markdown("#### Data")
    a, b, c = st.columns(3)
    a.download_button("Dataset (CSV)", csv_bytes(df), f"{name}.csv", "text/csv")
    b.download_button("Dataset (Excel)", to_excel_bytes(df), f"{name}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    c.download_button("Whole database (.db)", db.database_bytes(), "decision_support.db", "application/octet-stream")
    st.caption(f"The database file is also stored at {db.DB_PATH}. Open it with any SQLite tool.")
    st.markdown("#### Documentation")
    dic = ml.build_dictionary(df, cfg["descriptions"])
    st.download_button("Data dictionary (CSV)", csv_bytes(dic), f"{name}_data_dictionary.csv", "text/csv")
    st.markdown("#### Evaluation and report")
    if not cfg["target"]:
        st.info("Choose a target column in Data > Set up to build a report.")
        return
    trained = st.session_state.get(f"trained_{ds_id}")
    items = ins.build_insights(
        df, cfg["target"], cfg["date_col"], cfg["agg"], cfg["freq"], cfg["segments"],
        forecast=st.session_state.get(f"forecast_{ds_id}"),
        importance=trained["importance"] if trained else None,
    )
    rows = st.session_state.get(f"model_rows_{ds_id}")
    blocks = ins.build_report(meta, cfg, df, items, rows.round(4) if rows is not None else None, st.session_state.get(f"forecast_{ds_id}"))
    r1, r2 = st.columns(2)
    r1.download_button("Report (HTML)", ins.report_html(blocks).encode("utf-8"), f"{name}_report.html", "text/html")
    r2.download_button("Report (Markdown)", ins.report_markdown(blocks).encode("utf-8"), f"{name}_report.md", "text/markdown")
    if rows is None:
        st.caption("Train models on the Models page to include the evaluation table in the report.")
    saved = db.list_models(ds_id)
    if len(saved):
        st.markdown("#### Saved model files")
        for rec in saved.to_dict("records"):
            st.download_button(f"{rec['algorithm']} ({rec['created_at']})", db.read_file_bytes(rec["file_path"]), Path(rec["file_path"]).name, "application/octet-stream", key=f"dl_model_{rec['id']}")


{
    "Overview": page_overview,
    "Data": page_data,
    "Forecast": page_forecast,
    "Models": page_models,
    "Recommendations": page_recommendations,
    "Export": page_export,
}[page]()
