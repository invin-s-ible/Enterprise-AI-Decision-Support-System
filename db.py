"""Local storage for the Enterprise AI Decision Support System.

Everything is kept on your own computer:
  data/decision_support.db   SQLite database (your datasets live in tables ds_<id>)
  models/*.joblib            trained model files
"""
import json
import re
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models"
DB_PATH = DATA_DIR / "decision_support.db"


def _connect():
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    with closing(_connect()) as con, con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                source_file TEXT,
                rows INTEGER,
                cols INTEGER,
                created_at TEXT,
                date_cols TEXT DEFAULT '[]',
                config TEXT DEFAULT '{}'
            )"""
        )
        con.execute(
            """CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER,
                algorithm TEXT,
                target TEXT,
                metrics TEXT,
                file_path TEXT,
                created_at TEXT
            )"""
        )


# ------------------------------------------------------------------ datasets
def save_dataset(df, name, source_file="", date_cols=()):
    """Store a DataFrame in its own SQLite table and return the new dataset id."""
    date_cols = [c for c in date_cols if c in df.columns]
    out = df.copy()
    for c in date_cols:
        out[c] = pd.to_datetime(out[c], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    with closing(_connect()) as con, con:
        cur = con.execute(
            "INSERT INTO datasets (name, source_file, rows, cols, created_at, date_cols) VALUES (?,?,?,?,?,?)",
            (name, source_file, len(df), df.shape[1], datetime.now().strftime("%Y-%m-%d %H:%M"), json.dumps(date_cols)),
        )
        ds_id = cur.lastrowid
        out.to_sql(f"ds_{ds_id}", con, index=False, if_exists="replace")
    return ds_id


def list_datasets():
    with closing(_connect()) as con:
        return pd.read_sql_query(
            "SELECT id, name, source_file, rows, cols, created_at FROM datasets ORDER BY id DESC", con
        )


def get_dataset(ds_id):
    with closing(_connect()) as con:
        row = con.execute(
            "SELECT id, name, source_file, rows, cols, created_at, date_cols, config FROM datasets WHERE id=?",
            (int(ds_id),),
        ).fetchone()
    if row is None:
        return None
    keys = ["id", "name", "source_file", "rows", "cols", "created_at", "date_cols", "config"]
    meta = dict(zip(keys, row))
    meta["date_cols"] = json.loads(meta["date_cols"] or "[]")
    meta["config"] = json.loads(meta["config"] or "{}")
    return meta


def load_dataset(ds_id):
    meta = get_dataset(ds_id)
    if meta is None:
        return pd.DataFrame()
    with closing(_connect()) as con:
        df = pd.read_sql_query(f'SELECT * FROM "ds_{int(ds_id)}"', con)
    for c in meta["date_cols"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def update_config(ds_id, config):
    with closing(_connect()) as con, con:
        con.execute("UPDATE datasets SET config=? WHERE id=?", (json.dumps(config), int(ds_id)))


def delete_dataset(ds_id):
    ds_id = int(ds_id)
    for m in list_models(ds_id).to_dict("records"):
        Path(m["file_path"]).unlink(missing_ok=True)
    with closing(_connect()) as con, con:
        con.execute(f'DROP TABLE IF EXISTS "ds_{ds_id}"')
        con.execute(f'DROP TABLE IF EXISTS "forecast_{ds_id}"')
        con.execute("DELETE FROM models WHERE dataset_id=?", (ds_id,))
        con.execute("DELETE FROM datasets WHERE id=?", (ds_id,))


def save_table(table_name, df):
    """Store any result table (for example a forecast) in the local database."""
    safe = re.sub(r"[^A-Za-z0-9_]", "_", table_name)
    with closing(_connect()) as con, con:
        df.to_sql(safe, con, index=False, if_exists="replace")
    return safe


# ----------------------------------------------------------------- SQL console
def run_query(sql, limit=5000):
    """Run one read-only SELECT query against the local database."""
    text = sql.strip().rstrip(";").strip()
    if not re.match(r"^(select|with)\b", text, re.I) or ";" in text:
        raise ValueError("Only a single SELECT query is allowed here.")
    with closing(sqlite3.connect(DB_PATH)) as con:
        con.execute("PRAGMA query_only = ON")
        return pd.read_sql_query(text, con).head(limit)


# ---------------------------------------------------------------------- models
def save_model(dataset_id, algorithm, target, features, metrics, pipeline):
    MODEL_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"[^A-Za-z0-9]+", "_", algorithm).strip("_")
    path = MODEL_DIR / f"model_{int(dataset_id)}_{safe}_{stamp}.joblib"
    joblib.dump(
        {"pipeline": pipeline, "features": list(features), "target": target, "algorithm": algorithm},
        path,
    )
    with closing(_connect()) as con, con:
        con.execute(
            "INSERT INTO models (dataset_id, algorithm, target, metrics, file_path, created_at) VALUES (?,?,?,?,?,?)",
            (int(dataset_id), algorithm, target, json.dumps(metrics), str(path), datetime.now().strftime("%Y-%m-%d %H:%M")),
        )
    return str(path)


def list_models(dataset_id):
    with closing(_connect()) as con:
        df = pd.read_sql_query(
            "SELECT id, algorithm, target, metrics, file_path, created_at FROM models WHERE dataset_id=? ORDER BY id DESC",
            con,
            params=(int(dataset_id),),
        )
    return df


def load_model(path):
    return joblib.load(path)


def read_file_bytes(path):
    p = Path(path)
    return p.read_bytes() if p.exists() else b""


def database_bytes():
    return read_file_bytes(DB_PATH)
