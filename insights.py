"""Turns your data and model results into plain-language insights, risks,
opportunities and recommendations. Every number comes from your own data."""
import html

import numpy as np
import pandas as pd

from ml import aggregate_series, segment_candidates

ORDER = {"Risk": 0, "Opportunity": 1, "Trend": 2, "Driver": 3, "Info": 4}


def fmt(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(x):
        return "n/a"
    ax = abs(x)
    if ax >= 1e9:
        return f"{x / 1e9:.2f}B"
    if ax >= 1e6:
        return f"{x / 1e6:.2f}M"
    if ax >= 1e4:
        return f"{x / 1e3:.1f}K"
    if ax >= 100 or x == int(x):
        return f"{x:,.0f}"
    return f"{x:,.2f}"


def build_insights(df, target, date_col=None, agg="sum", freq="MS", segments=None, forecast=None, importance=None):
    items = []
    ts = None
    if date_col and date_col in df.columns:
        try:
            ts = aggregate_series(df, date_col, target, freq, agg)
        except Exception:
            ts = None
    k = None
    recent_mean = None

    if ts is not None and len(ts) >= 6:
        k = min(3, len(ts) // 2)
        recent_mean = float(ts.iloc[-k:].mean())
        prior_mean = float(ts.iloc[-2 * k : -k].mean())
        if prior_mean:
            change = recent_mean / prior_mean - 1
            direction = "up" if change > 0.02 else "down" if change < -0.02 else "flat"
            action = {
                "up": "Plan capacity, stock and budget for continued growth, and find which segments are driving it.",
                "down": "Find the cause now. Compare segments and time periods to see where the decline starts.",
                "flat": "Performance is steady. Look for growth levers in the segments and drivers below.",
            }[direction]
            items.append(
                {
                    "kind": "Trend",
                    "headline": f"{target} is trending {direction}",
                    "detail": f"The last {k} periods averaged {fmt(recent_mean)} against {fmt(prior_mean)} in the {k} before ({change:+.1%}).",
                    "action": action,
                    "impact": f"Change per period: {fmt(recent_mean - prior_mean)}",
                }
            )
        resid = ts - ts.rolling(5, center=True, min_periods=1).median()
        sd = float(resid.std())
        if sd > 0:
            z = resid / sd
            flagged = z[z.abs() > 2.5].abs().sort_values(ascending=False).head(3)
            if len(flagged):
                where = ", ".join(f"{i:%Y-%m-%d} ({fmt(ts[i])})" for i in flagged.index)
                items.append(
                    {
                        "kind": "Risk",
                        "headline": f"{len(flagged)} unusual period(s) in {target}",
                        "detail": f"These periods sit far from their neighbours: {where}.",
                        "action": "Check whether they are data errors, one-off events or real shocks before trusting forecasts.",
                        "impact": "Unexplained spikes can distort models",
                    }
                )

    seg_cols = [c for c in (segments or segment_candidates(df)) if c in df.columns][:3]
    for seg in seg_cols:
        n_groups = df[seg].nunique()
        if n_groups < 2 or n_groups > 50:
            continue
        totals = df.groupby(seg)[target].agg(agg).sort_values(ascending=False)
        if agg == "sum" and totals.sum() > 0:
            top, share = totals.index[0], totals.iloc[0] / totals.sum()
            if share > 0.5:
                items.append(
                    {
                        "kind": "Risk",
                        "headline": f"{seg} '{top}' carries {share:.0%} of {target}",
                        "detail": f"One {seg} group holds most of the total ({fmt(totals.iloc[0])} of {fmt(totals.sum())}).",
                        "action": f"Reduce dependence on '{top}' by growing the other {seg} groups.",
                        "impact": f"{share:.0%} concentration",
                    }
                )
        if ts is not None and k:
            piv = df.groupby([pd.Grouper(key=date_col, freq=freq), seg])[target].agg(agg).unstack()
            piv = piv.reindex(ts.index)  # same complete periods as the trend
            piv = piv.fillna(0) if agg == "sum" else piv
            if len(piv) >= 2 * k:
                recent = piv.iloc[-k:].mean()
                prior = piv.iloc[-2 * k : -k].mean()
                valid = prior > 0
                chg = recent / prior.where(valid) - 1
                gap = recent - prior
                declining = gap[valid & (chg < -0.10)].sort_values()
                growing = gap[valid & (chg > 0.10)].sort_values(ascending=False)
                if len(declining):
                    g = declining.index[0]
                    items.append(
                        {
                            "kind": "Risk",
                            "headline": f"{seg} '{g}' is declining",
                            "detail": f"Down {abs(chg[g]):.0%}: {fmt(recent[g])} per period recently against {fmt(prior[g])} before.",
                            "action": f"Investigate '{g}' (pricing, supply, competition or data issues) and set a recovery target.",
                            "impact": f"Lost per period: {fmt(-declining.iloc[0])}",
                        }
                    )
                if len(growing):
                    g = growing.index[0]
                    items.append(
                        {
                            "kind": "Opportunity",
                            "headline": f"{seg} '{g}' is growing fast",
                            "detail": f"Up {chg[g]:.0%}: {fmt(recent[g])} per period recently against {fmt(prior[g])} before.",
                            "action": f"Back '{g}' with more budget or capacity while the momentum lasts.",
                            "impact": f"Gain per period: {fmt(growing.iloc[0])}",
                        }
                    )

    if forecast is not None and recent_mean:
        fc = forecast["forecast"]["forecast"]
        chg = float(fc.mean()) / recent_mean - 1 if recent_mean else 0
        mae = forecast["metrics"]["MAE"]
        if chg > 0.05:
            items.append(
                {
                    "kind": "Opportunity",
                    "headline": f"Forecast expects {target} to rise",
                    "detail": f"Next {len(fc)} periods average {fmt(fc.mean())} against {fmt(recent_mean)} recently ({chg:+.1%}). Backtest error: {fmt(mae)} per period.",
                    "action": "Prepare resources for higher volume and set stretch targets.",
                    "impact": f"Expected gain per period: {fmt(fc.mean() - recent_mean)}",
                }
            )
        elif chg < -0.05:
            items.append(
                {
                    "kind": "Risk",
                    "headline": f"Forecast expects {target} to fall",
                    "detail": f"Next {len(fc)} periods average {fmt(fc.mean())} against {fmt(recent_mean)} recently ({chg:+.1%}). Backtest error: {fmt(mae)} per period.",
                    "action": "Build a contingency plan and review costs and commitments for the lower scenario.",
                    "impact": f"Expected loss per period: {fmt(recent_mean - fc.mean())}",
                }
            )

    if importance is not None and len(importance):
        top = [c for c in importance.index[:3]]
        items.append(
            {
                "kind": "Driver",
                "headline": f"{top[0]} has the strongest influence on {target}",
                "detail": "Top drivers by permutation importance on held-out data: " + ", ".join(top) + ".",
                "action": f"Focus monitoring and decisions on {top[0]}, and test changes to it with the prediction form on the Models page.",
                "impact": "Model-based driver ranking",
            }
        )

    if not items:
        items.append(
            {
                "kind": "Info",
                "headline": "Add structure to unlock insights",
                "detail": "No trends, risks or opportunities could be computed yet.",
                "action": "On Data > Set up, choose a date column, a target column and segment columns, then train a model.",
                "impact": "",
            }
        )
    items.sort(key=lambda i: ORDER.get(i["kind"], 9))
    return items


# --------------------------------------------------------------------- report
def build_report(meta, cfg, df, items, model_rows=None, forecast=None):
    blocks = [
        ("h1", f"Decision support report: {meta['name']}"),
        ("p", f"Source file: {meta.get('source_file') or 'n/a'}. Stored locally on {meta['created_at']}."),
        ("h2", "Dataset"),
        ("ul", [f"Rows: {len(df):,}", f"Columns: {df.shape[1]}", f"Target: {cfg.get('target') or 'not set'}", f"Date column: {cfg.get('date_col') or 'not set'}"]),
        ("h2", "Insights and recommendations"),
    ]
    for it in items:
        blocks.append(("ul", [f"[{it['kind']}] {it['headline']}. {it['detail']} Recommended action: {it['action']}"]))
    if model_rows is not None and len(model_rows):
        blocks += [("h2", "Model evaluation (held-out data)"), ("table", model_rows)]
    if forecast is not None:
        f = forecast["forecast"].copy()
        f.index = f.index.strftime("%Y-%m-%d")
        blocks += [
            ("h2", f"Forecast ({forecast['algorithm']})"),
            ("p", f"Backtest MAE {fmt(forecast['metrics']['MAE'])}, RMSE {fmt(forecast['metrics']['RMSE'])}."),
            ("table", f.round(2).reset_index().rename(columns={"index": "date"})),
        ]
    return blocks


def report_markdown(blocks):
    lines = []
    for kind, content in blocks:
        if kind == "h1":
            lines.append(f"# {content}\n")
        elif kind == "h2":
            lines.append(f"\n## {content}\n")
        elif kind == "p":
            lines.append(f"{content}\n")
        elif kind == "ul":
            lines += [f"- {i}" for i in content]
            lines.append("")
        elif kind == "table":
            lines.append(content.to_markdown(index=False) if _has_tabulate() else content.to_csv(index=False))
    return "\n".join(lines)


def _has_tabulate():
    try:
        import tabulate  # noqa: F401

        return True
    except ImportError:
        return False


def report_html(blocks):
    css = (
        "body{font-family:-apple-system,'SF Pro Text','Helvetica Neue',sans-serif;background:#F5F5F7;color:#1D1D1F;"
        "max-width:860px;margin:40px auto;padding:0 24px;line-height:1.5}"
        "h1{font-size:34px;letter-spacing:-.03em}h2{margin-top:36px;letter-spacing:-.02em}"
        "ul{background:#fff;border-radius:18px;padding:18px 18px 18px 38px}"
        "table{border-collapse:collapse;width:100%;background:#fff;border-radius:18px;overflow:hidden}"
        "th,td{padding:10px 14px;border-bottom:1px solid #ECECEF;text-align:left;font-size:14px}"
    )
    out = [f"<!doctype html><html><head><meta charset='utf-8'><title>Report</title><style>{css}</style></head><body>"]
    for kind, content in blocks:
        if kind in ("h1", "h2"):
            out.append(f"<{kind}>{html.escape(content)}</{kind}>")
        elif kind == "p":
            out.append(f"<p>{html.escape(content)}</p>")
        elif kind == "ul":
            out.append("<ul>" + "".join(f"<li>{html.escape(i)}</li>" for i in content) + "</ul>")
        elif kind == "table":
            out.append(content.to_html(index=False, border=0, escape=True))
    out.append("</body></html>")
    return "\n".join(out)
