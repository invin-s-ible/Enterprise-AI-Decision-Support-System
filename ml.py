"""Machine learning for the Decision Support System.

Works on any business table you upload: it detects column types, builds a
preprocessing pipeline with pandas / NumPy / scikit-learn, trains several models,
evaluates them on held-out data and forecasts time series.
"""
import importlib.util
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FREQUENCIES = {"Daily": "D", "Weekly": "W", "Monthly": "MS", "Quarterly": "QS", "Yearly": "YS"}
SEASON = {"D": 7, "W": 52, "MS": 12, "QS": 4, "YS": 1}


# --------------------------------------------------------------- column types
def detect_date_columns(df):
    """Return columns that are, or look like, dates."""
    found = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            found.append(c)
        elif s.dtype == object or pd.api.types.is_string_dtype(s):
            sample = s.dropna().astype(str).head(200)
            if sample.empty or not sample.str.contains(r"[-/:\s]", regex=True).all():
                continue
            ratio = 0.0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for dayfirst in (False, True):
                    parsed = pd.to_datetime(sample, errors="coerce", dayfirst=dayfirst)
                    ratio = max(ratio, float(parsed.notna().mean()))
            if ratio > 0.9:
                found.append(c)
    return found


def column_groups(df):
    numeric = df.select_dtypes(include="number").columns.tolist()
    dates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    cats = [c for c in df.columns if c not in numeric and c not in dates]
    return numeric, dates, cats


def segment_candidates(df, max_unique=50):
    """Text columns with a manageable number of groups (regions, products, ...)."""
    _, _, cats = column_groups(df)
    return [c for c in cats if 2 <= df[c].nunique() <= max_unique]


def build_dictionary(df, descriptions=None):
    descriptions = descriptions or {}
    rows = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            kind = "date"
        elif pd.api.types.is_numeric_dtype(s):
            kind = "number"
        else:
            kind = "text"
        if s.notna().any():
            if kind == "number":
                example = f"{s.min():.4g} to {s.max():.4g}"
            elif kind == "date":
                example = f"{s.min():%Y-%m-%d} to {s.max():%Y-%m-%d}"
            else:
                example = str(s.dropna().iloc[0])[:40]
        else:
            example = ""
        rows.append(
            {
                "Column": c,
                "Type": kind,
                "Non-null": int(s.notna().sum()),
                "Missing %": round(float(s.isna().mean()) * 100, 1),
                "Unique": int(s.nunique(dropna=True)),
                "Range or example": example,
                "Description": descriptions.get(c, ""),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------ model pipeline
class DateExpander(BaseEstimator, TransformerMixin):
    """Turns date columns into year / month / weekday / day-of-year numbers and
    makes text columns safe for one-hot encoding. Part of every saved pipeline,
    so new data can be fed in exactly as it appears in your table."""

    def fit(self, X, y=None):
        self.date_cols_ = [c for c in X.columns if pd.api.types.is_datetime64_any_dtype(X[c])]
        return self

    def transform(self, X):
        X = X.copy()
        for c in X.columns:
            if c in self.date_cols_ or pd.api.types.is_datetime64_any_dtype(X[c]):
                d = pd.to_datetime(X[c], errors="coerce")
                X[f"{c}_year"] = d.dt.year.astype("float64")
                X[f"{c}_month"] = d.dt.month.astype("float64")
                X[f"{c}_weekday"] = d.dt.dayofweek.astype("float64")
                X[f"{c}_dayofyear"] = d.dt.dayofyear.astype("float64")
                X = X.drop(columns=c)
            elif not pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]):
                X[c] = X[c].map(lambda v: v if pd.isna(v) else str(v)).astype(object)
        return X


class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    """Small PyTorch neural network with a scikit-learn style interface."""

    def __init__(self, hidden=(64, 32), epochs=60, lr=1e-3, batch_size=64, random_state=42):
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        import torch
        from torch import nn

        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32").reshape(-1, 1)
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) or 1.0
        yn = (y - self.y_mean_) / self.y_std_
        layers, width = [], X.shape[1]
        for h in self.hidden:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        layers.append(nn.Linear(width, 1))
        self.model_ = nn.Sequential(*layers)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        data = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(yn))
        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=True)
        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(self.model_(xb), yb).backward()
                opt.step()
        return self

    def predict(self, X):
        import torch

        self.model_.eval()
        with torch.no_grad():
            out = self.model_(torch.from_numpy(np.asarray(X, dtype="float32"))).numpy().ravel()
        return out * self.y_std_ + self.y_mean_


def available_models():
    """Algorithms offered in the app. The PyTorch network appears if torch is installed."""
    models = {
        "Linear Regression": lambda: LinearRegression(),
        "Ridge Regression": lambda: Ridge(alpha=1.0),
        "Random Forest": lambda: RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": lambda: GradientBoostingRegressor(random_state=42),
        "Neural Network (scikit-learn)": lambda: TransformedTargetRegressor(
            regressor=MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=42),
            transformer=StandardScaler(),
        ),
    }
    if importlib.util.find_spec("torch") is not None:
        models["Neural Network (PyTorch)"] = lambda: TorchMLPRegressor()
    return models


def build_pipeline(estimator):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=30)),
        ]
    )
    prep = ColumnTransformer(
        [
            ("num", numeric, make_column_selector(dtype_include="number")),
            ("cat", categorical, make_column_selector(dtype_exclude="number")),
        ]
    )
    return Pipeline([("dates", DateExpander()), ("prep", prep), ("model", estimator)])


def default_features(df, target):
    """All columns except the target and ID-like text columns."""
    keep = []
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_datetime64_any_dtype(s):
            if s.nunique() > 50 and s.nunique() > 0.5 * len(s):
                continue
        keep.append(c)
    return keep


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mape = np.nan
    mask = y_true != 0
    if mask.any():
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "MAPE": mape,
    }


def train_models(df, target, features, algorithms, date_col=None, test_size=0.2):
    """Train each chosen algorithm and score it on rows it never saw."""
    data = df.dropna(subset=[target]).copy()
    if len(data) < 30:
        raise ValueError("At least 30 rows with a value in the target column are needed to train models.")
    if not features:
        raise ValueError("Choose at least one input column.")
    chronological = bool(date_col and date_col in data.columns)
    if chronological:
        data = data.sort_values(date_col)
    X = data[list(features)]
    y = data[target].astype(float)
    if chronological:
        cut = int(len(data) * (1 - test_size))
        X_tr, X_te, y_tr, y_te = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)
    factories = available_models()
    results = []
    for name in algorithms:
        pipe = build_pipeline(factories[name]())
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        results.append(
            {
                "name": name,
                "pipeline": pipe,
                "metrics": regression_metrics(y_te, pred),
                "y_test": y_te.to_numpy(),
                "pred": np.asarray(pred, dtype=float),
                "X_test": X_te,
            }
        )
    results.sort(key=lambda r: r["metrics"]["RMSE"])
    return results


def feature_importance(pipeline, X_test, y_test, max_rows=2000):
    """Permutation importance on the original input columns (model agnostic)."""
    X_test = X_test.reset_index(drop=True)
    y_test = pd.Series(np.asarray(y_test)).reset_index(drop=True)
    if len(X_test) > max_rows:
        idx = np.random.RandomState(42).choice(len(X_test), max_rows, replace=False)
        X_test, y_test = X_test.iloc[idx], y_test.iloc[idx]
    result = permutation_importance(pipeline, X_test, y_test, n_repeats=5, random_state=42, scoring="r2")
    return pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)


# ------------------------------------------------------------- time series
def aggregate_series(df, date_col, target, freq, agg="sum"):
    """Combine rows into regular time periods. Incomplete first or last periods are
    dropped for totals, because a half-finished month would look like a collapse."""
    d = df[[date_col, target]].dropna(subset=[date_col]).sort_values(date_col)
    s = d.set_index(date_col)[target].astype(float).resample(freq).agg(agg)
    if agg != "sum":
        s = s.interpolate()
    s = s.dropna()
    dropped = []
    if agg == "sum" and freq != "D" and len(s) > 6 and len(d):
        first, last = d[date_col].min().normalize(), d[date_col].max().normalize()
        if freq == "W":  # weeks are labelled by their last day
            start_ok = first <= s.index[0] - pd.Timedelta(days=6)
            end_ok = last >= s.index[-1].normalize()
        else:  # month, quarter and year are labelled by their first day
            off = pd.tseries.frequencies.to_offset(freq)
            start_ok = first <= s.index[0].normalize()
            end_ok = last >= (s.index[-1] + off - pd.Timedelta(days=1)).normalize()
        if not end_ok:
            s = s.iloc[:-1]
            dropped.append("last")
        if not start_ok:
            s = s.iloc[1:]
            dropped.append("first")
    s.attrs["dropped_partial"] = dropped
    return s


def _ts_features(values, i, n_lags, season):
    row = list(values[i - n_lags : i]) + [float(i)]
    if season > 1:
        angle = 2 * np.pi * (i % season) / season
        row += [float(np.sin(angle)), float(np.cos(angle))]
    return row


def _ts_model(name):
    if name == "Linear Regression":
        return LinearRegression()
    if name == "Gradient Boosting":
        return GradientBoostingRegressor(random_state=42)
    return RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)


def forecast_series(series, freq, horizon, algorithm="Random Forest"):
    """Forecast a regular time series with lag features and a backtest on the last periods."""
    s = series.astype(float)
    n = len(s)
    if n < 12:
        raise ValueError(f"Only {n} time periods found. At least 12 are needed. Try a finer frequency such as weekly or daily.")
    season = SEASON.get(freq, 1)
    n_lags = max(2, min(12 if season >= 12 else 6, n // 3))
    vals = s.to_numpy()
    X = np.array([_ts_features(vals, i, n_lags, season) for i in range(n_lags, n)])
    y = vals[n_lags:]
    hold = max(2, min(horizon, len(y) // 4))
    model = _ts_model(algorithm).fit(X[:-hold], y[:-hold])
    back_pred = model.predict(X[-hold:])
    metrics = regression_metrics(y[-hold:], back_pred)
    sigma = max(metrics["RMSE"], 1e-9)
    final = _ts_model(algorithm).fit(X, y)
    hist, out = list(vals), []
    for _ in range(horizon):
        row = _ts_features(np.array(hist), len(hist), n_lags, season)
        p = float(final.predict([row])[0])
        hist.append(p)
        out.append(p)
    idx = pd.date_range(s.index[-1], periods=horizon + 1, freq=freq)[1:]
    out = np.array(out)
    spread = 1.2816 * sigma * np.sqrt(np.arange(1, horizon + 1))
    lower, upper = out - spread, out + spread
    if (vals >= 0).all():
        lower = np.maximum(lower, 0)
    return {
        "history": s,
        "backtest": pd.DataFrame({"actual": y[-hold:], "predicted": back_pred}, index=s.index[n_lags:][-hold:]),
        "forecast": pd.DataFrame({"forecast": out, "lower": lower, "upper": upper}, index=idx),
        "metrics": metrics,
        "algorithm": algorithm,
    }
