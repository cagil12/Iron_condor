"""Dual-module VIX x range x breach analysis for XSP 0DTE Iron Condor research.

Module A (market-data only): SPX (^GSPC) + VIX (^VIX) daily OHLC analysis.
Module B (model-dependent): synthetic backtester outcomes from synthetic_backtest.py.

Standalone CLI:
    python -m src.research.vix_breach_eda
    python src/research/vix_breach_eda.py
"""

from __future__ import annotations

import copy
import math
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2_contingency, kruskal

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

try:
    import yfinance as yf
except Exception as exc:  # pragma: no cover
    raise RuntimeError("yfinance is required for vix_breach_eda.py") from exc

try:  # optional dependency; fallback exists
    from sklearn.metrics import auc as skl_auc
    from sklearn.metrics import roc_curve as skl_roc_curve
except Exception:  # pragma: no cover
    skl_auc = None
    skl_roc_curve = None

try:
    from . import synthetic_backtest as sb
except ImportError:
    REPO_ROOT_FALLBACK = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT_FALLBACK) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT_FALLBACK))
    from src.research import synthetic_backtest as sb


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "backtest_config.yaml"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vix_breach_eda"

MODULE_A_START = "2005-01-01"
VIX_BUCKETS = [0.0, 15.0, 20.0, 25.0, 30.0, 40.0, np.inf]
VIX_BUCKET_LABELS = ["0-15", "15-20", "20-25", "25-30", "30-40", "40+"]
VIX_BUCKET_CATEGORIES = pd.CategoricalDtype(categories=VIX_BUCKET_LABELS, ordered=True)
THRESHOLDS = [0.7, 1.0, 1.3, 1.5, 2.0]
THRESHOLD_LABELS = [f"{x:.1f}%" for x in THRESHOLDS]
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
PNL_COL = "outcome_tp50_sl_capped"

ERA_SPECS: List[Tuple[str, str, Optional[str]]] = [
    ("2005-09", "2005-01-01", "2009-12-31"),
    ("2010-14", "2010-01-01", "2014-12-31"),
    ("2015-19", "2015-01-01", "2019-12-31"),
    ("2020-24", "2020-01-01", "2024-12-31"),
    ("2025+", "2025-01-01", None),
]

BUCKET_COLORS = {
    "0-15": "#4C78A8",
    "15-20": "#59A14F",
    "20-25": "#F28E2B",
    "25-30": "#E15759",
    "30-40": "#B07AA1",
    "40+": "#9C755F",
}


class Reporter:
    def __init__(self) -> None:
        self.lines: List[str] = []

    def line(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def section(self, title: str) -> None:
        self.line()
        self.line(title)
        self.line("-" * len(title))

    def table(self, df: Optional[pd.DataFrame], index: bool = True, float_fmt: str = "{:.4f}") -> None:
        if df is None or df.empty:
            self.line("(no rows)")
            return
        fmt_df = df.copy()
        for col in fmt_df.columns:
            if pd.api.types.is_float_dtype(fmt_df[col]):
                fmt_df[col] = fmt_df[col].map(lambda x: "" if pd.isna(x) else float_fmt.format(float(x)))
        self.line(fmt_df.to_string(index=index))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def _safe_div(num: float, den: float) -> float:
    if den in (0, 0.0) or pd.isna(den):
        return float("nan")
    return float(num) / float(den)


def _quantile_or_nan(series: pd.Series, q: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.quantile(q)) if not s.empty else float("nan")


def _set_plot_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.frameon": True,
        }
    )


def _format_pct(x: float) -> str:
    return "NA" if not np.isfinite(x) else f"{x:.1%}"


def _bucketize_vix(values: pd.Series) -> pd.Categorical:
    return pd.cut(values, bins=VIX_BUCKETS, labels=VIX_BUCKET_LABELS, right=False, include_lowest=True)


def _extract_yf_ohlc(frame: pd.DataFrame, symbol: str, prefix: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError(f"No data returned for {symbol}")
    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        lvl0 = list(data.columns.get_level_values(0))
        lvl1 = list(data.columns.get_level_values(1))
        if symbol in lvl0:
            data = data[symbol]
        elif symbol in lvl1:
            data = data.xs(symbol, axis=1, level=1)
        else:
            data.columns = data.columns.get_level_values(-1)
    data = data.rename(columns={c: str(c).lower() for c in data.columns})
    need = ["open", "high", "low", "close"]
    for col in need:
        if col not in data.columns:
            raise ValueError(f"Missing '{col}' in Yahoo frame for {symbol}")
    out = data[need].rename(columns={c: f"{prefix}_{c}" for c in need})
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.sort_index()


def _download_symbol_prefixed(symbol: str, prefix: str, start_date: str, end_date: str, attempts: int = 3) -> pd.DataFrame:
    end_plus = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            raw = yf.download(
                symbol,
                start=start_date,
                end=end_plus,
                auto_adjust=False,
                progress=False,
                interval="1d",
                threads=False,
            )
            return _extract_yf_ohlc(raw, symbol, prefix=prefix)
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            if attempt < attempts:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"Failed Yahoo download for {symbol} after {attempts} attempts: {last_exc}")


def _cramers_v_from_contingency(table: pd.DataFrame) -> Tuple[float, float, int, float]:
    if table.empty or float(np.asarray(table.values).sum()) <= 0:
        return (np.nan, np.nan, 0, np.nan)
    chi2, p_value, dof, _ = chi2_contingency(table.values)
    n = float(table.values.sum())
    r, k = table.shape
    denom = max(1.0, min(r - 1, k - 1))
    cramers_v = math.sqrt((chi2 / n) / denom) if denom > 0 else np.nan
    return (float(chi2), float(p_value), int(dof), float(cramers_v))


def _kruskal_safe(groups: Sequence[np.ndarray]) -> Tuple[float, float]:
    groups_nonempty = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
    if len(groups_nonempty) < 2:
        return (np.nan, np.nan)
    try:
        h, p = kruskal(*groups_nonempty)
        return (float(h), float(p))
    except Exception:
        return (np.nan, np.nan)


def _roc_manual(y_true: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.size == 0:
        return np.array([]), np.array([]), np.array([]), float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    s = s[order]
    positives = float((y == 1).sum())
    negatives = float((y == 0).sum())
    if positives == 0 or negatives == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf]), float("nan")
    unique_scores = np.unique(s)[::-1]
    tpr_list: List[float] = [0.0]
    fpr_list: List[float] = [0.0]
    thr_list: List[float] = [np.inf]
    for thr in unique_scores:
        pred = s >= thr
        tp = float(((pred) & (y == 1)).sum())
        fp = float(((pred) & (y == 0)).sum())
        tpr_list.append(tp / positives)
        fpr_list.append(fp / negatives)
        thr_list.append(float(thr))
    tpr_list.append(1.0)
    fpr_list.append(1.0)
    thr_list.append(float("-inf"))
    fpr = np.asarray(fpr_list, dtype=float)
    tpr = np.asarray(tpr_list, dtype=float)
    thr = np.asarray(thr_list, dtype=float)
    return fpr, tpr, thr, float(np.trapz(tpr, fpr))


def _roc_curve_with_fallback(y_true: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if skl_roc_curve is not None and skl_auc is not None:
        fpr, tpr, thr = skl_roc_curve(y_true, scores)
        return np.asarray(fpr), np.asarray(tpr), np.asarray(thr), float(skl_auc(fpr, tpr))
    return _roc_manual(np.asarray(y_true), np.asarray(scores))


def _youden_best(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> Dict[str, float]:
    if len(fpr) == 0:
        return {"threshold": np.nan, "tpr": np.nan, "fpr": np.nan, "j": np.nan}
    j = tpr - fpr
    idx = int(np.nanargmax(j)) if np.isfinite(j).any() else 0
    return {
        "threshold": float(thresholds[idx]) if np.isfinite(thresholds[idx]) else np.nan,
        "tpr": float(tpr[idx]),
        "fpr": float(fpr[idx]),
        "j": float(j[idx]),
    }


def _point_from_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pred = s >= float(threshold)
    pos = y == 1
    neg = y == 0
    tp = float((pred & pos).sum())
    fp = float((pred & neg).sum())
    return {
        "threshold": float(threshold),
        "tpr": _safe_div(tp, float(pos.sum())),
        "fpr": _safe_div(fp, float(neg.sum())),
    }


def _pearsonr_safe(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 2:
        return float("nan")
    x_use = x_arr[mask]
    y_use = y_arr[mask]
    if np.allclose(x_use, x_use[0]) or np.allclose(y_use, y_use[0]):
        return float("nan")
    return float(np.corrcoef(x_use, y_use)[0, 1])


def _write_csv(df: pd.DataFrame, path: Path, index: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return cfg


def _apply_backtest_overrides(base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("default", {})
    if not isinstance(cfg["default"], dict):
        raise ValueError("config.default must be a mapping")
    cfg["default"]["wing_width"] = 2.0
    cfg["default"]["tp_pct"] = 0.50
    cfg["default"]["sl_mult"] = 3.0
    return cfg


def _load_cached_backtest_data(cache_file: Path, start: str, end: str) -> pd.DataFrame:
    data = pd.read_csv(cache_file, parse_dates=["date"])
    data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)
    data = data[(data["date"] >= pd.to_datetime(start)) & (data["date"] <= pd.to_datetime(end))].copy()
    data = data.sort_values("date").reset_index(drop=True)
    vix_cols = [c for c in ["vix_open", "vix_high", "vix_low", "vix_close"] if c in data.columns]
    for col in vix_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    if vix_cols:
        data[vix_cols] = data[vix_cols].ffill()
    if {"spx_high", "spx_low"}.issubset(data.columns):
        data = data[data["spx_high"] >= data["spx_low"]].copy()
    if {"spx_open", "spx_close"}.issubset(data.columns):
        data = data.dropna(subset=["spx_open", "spx_close"])
    return data.reset_index(drop=True)


def _data_has_usable_vix(data: pd.DataFrame) -> bool:
    if data is None or data.empty or "vix_close" not in data.columns:
        return False
    return int(pd.to_numeric(data["vix_close"], errors="coerce").notna().sum()) > 0


def _load_backtester_source_data(cfg: Dict[str, Any], report: Reporter) -> pd.DataFrame:
    params = sb._backtest_params(cfg)
    start_date = str(params["start_date"])
    end_date = str(params["end_date"])
    cache_file = Path(str(params["cache_file"]))

    report.section("Module B Setup: Backtester Data Load")
    report.line(f"Config date range: {start_date} -> {end_date}")
    report.line(f"Backtester cache (read-only preference): {cache_file}")

    data = pd.DataFrame()
    if cache_file.exists():
        try:
            cached = _load_cached_backtest_data(cache_file, start_date, end_date)
            if _data_has_usable_vix(cached):
                data = cached
                report.line("Using existing backtester cache (read-only).")
            else:
                report.line("Cache exists but VIX invalid/empty. Will download in-memory only.")
        except Exception as exc:
            report.line(f"WARNING: cache read failed ({exc.__class__.__name__}: {exc}). Downloading in-memory.")

    if data.empty:
        data = sb._download_spx_vix(start_date=start_date, end_date=end_date)
        data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)
        data = data[(data["date"] >= pd.to_datetime(start_date)) & (data["date"] <= pd.to_datetime(end_date))].copy()
        data = data.sort_values("date").reset_index(drop=True)
        if not _data_has_usable_vix(data):
            raise ValueError("Backtester source download returned no usable VIX values.")
        report.line("Downloaded backtester source data in-memory via synthetic_backtest._download_spx_vix().")
        local_copy = OUTPUT_DIR / "_module_b_source_data_snapshot.csv"
        data.to_csv(local_copy, index=False)
        report.line(f"Saved local snapshot: {local_copy}")

    summary = sb.validate_data(data)
    report.line(
        "Source rows: {rows} | Date range: {d0} -> {d1}".format(
            rows=len(data), d0=summary["date_range"][0], d1=summary["date_range"][1]
        )
    )
    return data


def _run_backtester(cfg: Dict[str, Any], data: pd.DataFrame, report: Reporter) -> pd.DataFrame:
    params = sb._backtest_params(cfg)
    report.line(
        "Running backtester with wing_width={w:.1f}, tp_pct={tp:.2f}, sl_mult={sl:.1f}, entry_hour={eh:.2f}".format(
            w=float(params["wing_width"]),
            tp=float(params["take_profit_pct"]),
            sl=float(params["stop_loss_mult"]),
            eh=float(params["entry_hour"]),
        )
    )
    out = sb.run_backtest(cfg, data=data)
    tradable = int((~out["skipped"].astype(bool)).sum()) if "skipped" in out.columns else len(out)
    report.line(f"Backtester output rows: {len(out)} | Tradable rows: {tradable}")
    return out


def _market_cache_path() -> Path:
    return OUTPUT_DIR / "_module_a_spx_vix_cache.csv"


def _load_module_a_market_data(report: Reporter) -> pd.DataFrame:
    report.section("MODULE A.0: Data Acquisition (SPX + VIX)")
    end_date = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    report.line(f"Target range: {MODULE_A_START} -> {end_date}")

    cache_path = _market_cache_path()
    xsp_note = "not attempted"
    last_exc: Optional[Exception] = None

    try:
        spx = _download_symbol_prefixed("^GSPC", "spx", MODULE_A_START, end_date, attempts=3)
        vix = _download_symbol_prefixed("^VIX", "vix", MODULE_A_START, end_date, attempts=3)
        merged = spx.join(vix, how="inner").sort_index()
        merged = merged[merged.index.dayofweek < 5].copy()
        merged["date"] = merged.index
        merged = merged.reset_index(drop=True)
        try:
            xsp = _download_symbol_prefixed("^XSP", "xsp", MODULE_A_START, end_date, attempts=2)
            xsp_reset = xsp.reset_index().rename(columns={"index": "date"})
            xsp_reset["date"] = pd.to_datetime(xsp_reset["date"]).dt.tz_localize(None)
            merged = merged.merge(xsp_reset, on="date", how="left")
            xsp_rows = int(pd.to_numeric(merged.get("xsp_close"), errors="coerce").notna().sum()) if "xsp_close" in merged.columns else 0
            xsp_note = f"downloaded (usable rows={xsp_rows})"
        except Exception as exc_xsp:  # pragma: no cover
            xsp_note = f"unavailable ({exc_xsp.__class__.__name__})"

        required = ["spx_open", "spx_high", "spx_low", "spx_close", "vix_close"]
        for col in required:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
        merged = merged.dropna(subset=required).copy()
        merged = merged[merged["spx_high"] >= merged["spx_low"]].copy()

        merged.to_csv(cache_path, index=False)
        report.line("Data source: Yahoo Finance (^GSPC, ^VIX)")
        report.line(f"Optional XSP status: {xsp_note}")
        report.line(f"Cached merged dataset: {cache_path}")
    except Exception as exc:  # pragma: no cover
        last_exc = exc
        if not cache_path.exists():
            raise
        report.line(f"WARNING: Yahoo download failed ({exc.__class__.__name__}: {exc}). Using local cache.")
        merged = pd.read_csv(cache_path, parse_dates=["date"])
        merged["date"] = pd.to_datetime(merged["date"]).dt.tz_localize(None)
        report.line(f"Data source: local cache {cache_path}")

    merged = merged.sort_values("date").reset_index(drop=True)
    merged["vix_prev_close"] = pd.to_numeric(merged["vix_close"], errors="coerce").shift(1)
    merged["ret_pct"] = pd.to_numeric(merged["spx_close"], errors="coerce").pct_change() * 100.0
    report.line(
        "Final merged rows: {n} | Date range: {d0} -> {d1}".format(
            n=len(merged),
            d0=merged["date"].min().date() if not merged.empty else "N/A",
            d1=merged["date"].max().date() if not merged.empty else "N/A",
        )
    )
    if last_exc is not None:
        report.line(f"Note: live download failed this run; cache fallback used ({last_exc.__class__.__name__}).")
    return merged


def _prepare_module_a_frame(raw: pd.DataFrame, report: Reporter) -> pd.DataFrame:
    report.section("MODULE A.1: Intraday Range Metrics")
    df = raw.copy()
    for col in ["spx_open", "spx_high", "spx_low", "spx_close", "vix_close", "vix_prev_close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["range_pct"] = (df["spx_high"] - df["spx_low"]) / df["spx_open"] * 100.0
    df["up_range_pct"] = (df["spx_high"] - df["spx_open"]) / df["spx_open"] * 100.0
    df["down_range_pct"] = (df["spx_open"] - df["spx_low"]) / df["spx_open"] * 100.0
    df["max_directional"] = np.maximum(df["up_range_pct"], df["down_range_pct"])
    df["weekday"] = pd.to_datetime(df["date"]).dt.day_name()
    df["weekday"] = pd.Categorical(df["weekday"], categories=WEEKDAY_ORDER, ordered=True)
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["vix_regime"] = df["vix_prev_close"]  # no look-ahead
    df["vix_bucket"] = _bucketize_vix(df["vix_regime"]).astype(VIX_BUCKET_CATEGORIES)
    mask = (
        np.isfinite(df["spx_open"])
        & np.isfinite(df["spx_high"])
        & np.isfinite(df["spx_low"])
        & np.isfinite(df["spx_close"])
        & np.isfinite(df["vix_regime"])
        & (~df["vix_bucket"].isna())
    )
    df = df.loc[mask].copy()
    report.line(f"Usable rows for VIX-conditioned analyses (vix_prev_close): {len(df)}")
    report.line(
        f"Median max_directional={df['max_directional'].median():.3f}% | P90={df['max_directional'].quantile(0.9):.3f}% | P99={df['max_directional'].quantile(0.99):.3f}%"
    )
    return df


def _module_a_a2_exceedance(df: pd.DataFrame, report: Reporter) -> Dict[str, Any]:
    report.section("MODULE A.2: Threshold Exceedance by VIX Bucket")
    rows: List[Dict[str, Any]] = []
    for bucket in VIX_BUCKET_LABELS:
        sub = df[df["vix_bucket"] == bucket]
        row: Dict[str, Any] = {"vix_bucket": bucket, "n_days": int(len(sub))}
        for thr in THRESHOLDS:
            row[f"p_exceed_{str(thr).replace('.', 'p')}"] = float((sub["max_directional"] > thr).mean()) if len(sub) else np.nan
        rows.append(row)
    table = pd.DataFrame(rows).set_index("vix_bucket")
    _write_csv(table, OUTPUT_DIR / "A2_exceedance_table.csv")
    report.table(table, index=True)

    heat_cols = [f"p_exceed_{str(t).replace('.', 'p')}" for t in THRESHOLDS]
    heat_df = table[heat_cols].copy()
    heat_df.columns = THRESHOLD_LABELS

    fig, ax = plt.subplots(figsize=(11, 6))
    if sns is not None:
        sns.heatmap(
            heat_df,
            annot=True,
            fmt=".1%",
            cmap="YlOrRd",
            cbar_kws={"label": "Exceedance probability"},
            linewidths=0.5,
            ax=ax,
        )
    else:
        data = heat_df.to_numpy(dtype=float)
        vmax = np.nanmax(data) if np.isfinite(data).any() else 1.0
        im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=max(0.01, float(vmax)))
        fig.colorbar(im, ax=ax, label="Exceedance probability")
        ax.set_xticks(np.arange(len(heat_df.columns)))
        ax.set_xticklabels(list(heat_df.columns))
        ax.set_yticks(np.arange(len(heat_df.index)))
        ax.set_yticklabels(list(heat_df.index))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if np.isfinite(data[i, j]):
                    ax.text(j, i, f"{data[i, j]:.1%}", ha="center", va="center", fontsize=8)
    ax.set_title("A.2 Exceedance Probability: P(max_directional > threshold | VIX bucket)")
    ax.set_xlabel("Threshold (percent move from open)")
    ax.set_ylabel("VIX bucket (previous close)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart_A2_exceedance_heatmap.png", dpi=180)
    plt.close(fig)

    one_pct = (df["max_directional"] > 1.0).astype(int)
    agg = (
        df.assign(exceed_1p0=one_pct)
        .groupby("vix_bucket", observed=True)["exceed_1p0"]
        .agg(total="count", exceed="sum")
        .reindex(VIX_BUCKET_LABELS)
        .fillna(0)
    )
    agg["total"] = agg["total"].astype(int)
    agg["exceed"] = agg["exceed"].astype(int)
    agg["no_exceed"] = (agg["total"] - agg["exceed"]).astype(int)
    ct = agg.loc[agg["total"] > 0, ["no_exceed", "exceed"]]
    chi2, p, dof, cv = _cramers_v_from_contingency(ct)
    report.line(f"A.2 (1.0% exceed): chi2={chi2:.4f}, dof={dof}, p={p:.6f}, Cramer's V={cv:.4f}")
    return {"table": table, "heat_df": heat_df, "one_pct_stats": {"chi2": chi2, "p_value": p, "dof": dof, "cramers_v": cv}}


def _module_a_a3_distributions(df: pd.DataFrame, report: Reporter) -> Dict[str, Any]:
    report.section("MODULE A.3: Distribution of Ranges by VIX Bucket")
    rows: List[Dict[str, Any]] = []
    for bucket in VIX_BUCKET_LABELS:
        sub = df[df["vix_bucket"] == bucket]
        md = sub["max_directional"]
        rg = sub["range_pct"]
        rows.append(
            {
                "vix_bucket": bucket,
                "n_days": int(len(sub)),
                "maxdir_mean": float(md.mean()) if len(sub) else np.nan,
                "maxdir_median": float(md.median()) if len(sub) else np.nan,
                "maxdir_p75": _quantile_or_nan(md, 0.75),
                "maxdir_p90": _quantile_or_nan(md, 0.90),
                "maxdir_p95": _quantile_or_nan(md, 0.95),
                "maxdir_p99": _quantile_or_nan(md, 0.99),
                "range_mean": float(rg.mean()) if len(sub) else np.nan,
                "range_median": float(rg.median()) if len(sub) else np.nan,
                "range_p75": _quantile_or_nan(rg, 0.75),
                "range_p90": _quantile_or_nan(rg, 0.90),
                "range_p95": _quantile_or_nan(rg, 0.95),
                "range_p99": _quantile_or_nan(rg, 0.99),
            }
        )
    table = pd.DataFrame(rows).set_index("vix_bucket")
    _write_csv(table, OUTPUT_DIR / "A3_range_percentiles.csv")
    report.table(table, index=True)

    groups = [df.loc[df["vix_bucket"] == b, "max_directional"].dropna().to_numpy() for b in VIX_BUCKET_LABELS]
    kw_h, kw_p = _kruskal_safe(groups)
    report.line(f"A.3 Kruskal-Wallis on max_directional across VIX buckets: H={kw_h:.4f}, p={kw_p:.6f}")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    x_max = max(0.5, float(df["max_directional"].quantile(0.995)))
    bins = np.linspace(0, x_max, 45)
    for bucket in VIX_BUCKET_LABELS:
        vals = df.loc[df["vix_bucket"] == bucket, "max_directional"].dropna()
        if vals.empty:
            continue
        color = BUCKET_COLORS[bucket]
        if sns is not None and len(vals) >= 10:
            try:
                sns.kdeplot(vals, ax=ax, label=f"{bucket} (n={len(vals)})", color=color, linewidth=1.8)
                continue
            except Exception:
                pass
        ax.hist(vals, bins=bins, density=True, alpha=0.25, color=color, label=f"{bucket} (n={len(vals)})")
    ax.set_title("A.3 Distribution of Max Directional Range by VIX Bucket")
    ax.set_xlabel("Max directional move from open (%)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart_A3_range_distributions.png", dpi=180)
    plt.close(fig)
    return {"table": table, "kruskal_h": kw_h, "kruskal_p": kw_p}


def _module_a_a4_asymmetry(df: pd.DataFrame, report: Reporter, threshold: float = 1.0) -> Dict[str, Any]:
    report.section("MODULE A.4: Directional Asymmetry (Put vs Call Risk)")
    down = df["down_range_pct"] > threshold
    up = df["up_range_pct"] > threshold
    work = df.assign(down_exceeds=down, up_exceeds=up)
    rows: List[Dict[str, Any]] = []
    for bucket in VIX_BUCKET_LABELS:
        sub = work[work["vix_bucket"] == bucket]
        n = int(len(sub))
        if n == 0:
            rows.append({"vix_bucket": bucket, "n_days": 0, "p_down_only": np.nan, "p_up_only": np.nan, "p_both": np.nan, "p_neither": np.nan, "down_share_of_total_breaches": np.nan, "total_breach_rate": np.nan})
            continue
        d = sub["down_exceeds"].astype(bool)
        u = sub["up_exceeds"].astype(bool)
        any_b = d | u
        rows.append(
            {
                "vix_bucket": bucket,
                "n_days": n,
                "p_down_only": float((d & ~u).mean()),
                "p_up_only": float((u & ~d).mean()),
                "p_both": float((d & u).mean()),
                "p_neither": float((~any_b).mean()),
                "down_share_of_total_breaches": _safe_div(float(d.sum()), float(any_b.sum())),
                "total_breach_rate": float(any_b.mean()),
            }
        )
    table = pd.DataFrame(rows).set_index("vix_bucket")
    _write_csv(table, OUTPUT_DIR / "A4_directional_asymmetry.csv")
    report.table(table, index=True)

    fig, ax1 = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(VIX_BUCKET_LABELS))
    bottom = np.zeros(len(VIX_BUCKET_LABELS), dtype=float)
    stack_cols = [
        ("p_neither", "Neither", "#D0D0D0"),
        ("p_down_only", "Down only (>1.0%)", "#E15759"),
        ("p_up_only", "Up only (>1.0%)", "#4C78A8"),
        ("p_both", "Both", "#B07AA1"),
    ]
    for col, label, color in stack_cols:
        vals = table[col].fillna(0).to_numpy(dtype=float)
        ax1.bar(x, vals, bottom=bottom, label=label, color=color, edgecolor="white", linewidth=0.5)
        bottom += vals
    ax1.set_xticks(x)
    ax1.set_xticklabels(VIX_BUCKET_LABELS)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Probability")
    ax1.set_title("A.4 Directional Asymmetry by VIX Bucket (threshold = 1.0%)")
    ax2 = ax1.twinx()
    ax2.plot(x, table["down_share_of_total_breaches"].to_numpy(dtype=float), marker="o", color="#7A1E1E", linewidth=2, label="Down-side share of total breaches")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Down-side share among breached days")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart_A4_asymmetry_by_vix.png", dpi=180)
    plt.close(fig)

    overall_down_share = _safe_div(float(down.sum()), float((down | up).sum()))
    report.line(f"Overall down-side share of 1.0% breaches (all buckets): {_format_pct(overall_down_share)}")
    return {"table": table, "overall_down_share": overall_down_share}


def _module_a_a5_dow_given_vix(df: pd.DataFrame, report: Reporter) -> Dict[str, Any]:
    report.section("MODULE A.5: Day-of-Week Effect (Conditional on VIX)")
    rows: List[Dict[str, Any]] = []
    for bucket in VIX_BUCKET_LABELS:
        sub = df[df["vix_bucket"] == bucket]
        groups = [sub.loc[sub["weekday"] == wd, "max_directional"].dropna().to_numpy() for wd in WEEKDAY_ORDER]
        h, p = _kruskal_safe(groups)
        rows.append(
            {
                "vix_bucket": bucket,
                "n_days": int(len(sub)),
                "weekdays_present": int(sum(len(g) > 0 for g in groups)),
                "kruskal_h": h,
                "p_value": p,
                "p_gt_0_10": bool(np.isfinite(p) and p > 0.10),
            }
        )
    table = pd.DataFrame(rows).set_index("vix_bucket")
    _write_csv(table, OUTPUT_DIR / "A5_dow_given_vix.csv")
    report.table(table, index=True)
    subsumed = bool((table["p_value"].dropna() > 0.10).all()) if not table["p_value"].dropna().empty else False
    report.line(f"A.5 conclusion (all buckets p>0.10): {'YES' if subsumed else 'NO'}")
    return {"table": table, "subsumed": subsumed}


def _assign_era(date_series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(date_series)
    out = pd.Series(pd.Categorical([None] * len(dates), categories=[x[0] for x in ERA_SPECS], ordered=True), index=dates.index)
    for label, start, end in ERA_SPECS:
        start_ts = pd.to_datetime(start)
        mask = dates >= start_ts if end is None else ((dates >= start_ts) & (dates <= pd.to_datetime(end)))
        out.loc[mask] = label
    return out.astype(pd.CategoricalDtype(categories=[x[0] for x in ERA_SPECS], ordered=True))


def _module_a_a6_temporal_stability(df: pd.DataFrame, report: Reporter) -> Dict[str, Any]:
    report.section("MODULE A.6: Temporal Stability (5-year windows)")
    work = df.copy()
    work["era"] = _assign_era(work["date"])
    work["exceed_1p0"] = (work["max_directional"] > 1.0).astype(int)
    rows: List[Dict[str, Any]] = []
    for era_label, _, _ in ERA_SPECS:
        for bucket in VIX_BUCKET_LABELS:
            sub = work[(work["era"] == era_label) & (work["vix_bucket"] == bucket)]
            rows.append({"era": era_label, "vix_bucket": bucket, "n_days": int(len(sub)), "exceed_rate_1p0": float(sub["exceed_1p0"].mean()) if len(sub) else np.nan})
    long_df = pd.DataFrame(rows)
    _write_csv(long_df.set_index(["era", "vix_bucket"]), OUTPUT_DIR / "A6_temporal_stability.csv")
    report.table(long_df.pivot(index="era", columns="vix_bucket", values="exceed_rate_1p0"), index=True)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    eras = [e[0] for e in ERA_SPECS]
    x = np.arange(len(eras))
    for bucket in VIX_BUCKET_LABELS:
        sub = long_df[long_df["vix_bucket"] == bucket].set_index("era").reindex(eras)
        ax.plot(x, sub["exceed_rate_1p0"].to_numpy(dtype=float), marker="o", label=bucket, color=BUCKET_COLORS[bucket])
    ax.set_xticks(x)
    ax.set_xticklabels(eras)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(max_directional > 1.0%)")
    ax.set_xlabel("Era")
    ax.set_title("A.6 Temporal Stability of 1.0% Exceedance by VIX Bucket")
    ax.legend(title="VIX bucket", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart_A6_stability_by_era.png", dpi=180)
    plt.close(fig)

    per_bucket_ranges: Dict[str, float] = {}
    for bucket in VIX_BUCKET_LABELS:
        sub = long_df[(long_df["vix_bucket"] == bucket) & (long_df["n_days"] >= 30)]
        vals = sub["exceed_rate_1p0"].dropna().to_numpy(dtype=float)
        per_bucket_ranges[bucket] = float(np.nanmax(vals) - np.nanmin(vals)) if len(vals) >= 2 else np.nan
    valid_ranges = [v for v in per_bucket_ranges.values() if np.isfinite(v)]
    stability_label = "STABLE"
    if valid_ranges and (max(valid_ranges) > 0.15 or np.nanmedian(valid_ranges) > 0.10):
        stability_label = "REGIME-DEPENDENT"
    report.line(f"A.6 stability heuristic: {stability_label} (bucket range max={(max(valid_ranges) if valid_ranges else float('nan')):.3f})")
    return {"table_long": long_df, "per_bucket_ranges": per_bucket_ranges, "stability_label": stability_label}


def _module_a_a7_roc(df: pd.DataFrame, report: Reporter) -> Dict[str, Any]:
    report.section("MODULE A.7: VIX as Predictor (ROC for max_directional > 1.0%)")
    work = df[["vix_regime", "max_directional"]].dropna().copy()
    y_true = (work["max_directional"] > 1.0).astype(int).to_numpy(dtype=int)
    scores = work["vix_regime"].to_numpy(dtype=float)
    fpr, tpr, thresholds, auc_val = _roc_curve_with_fallback(y_true, scores)
    youden = _youden_best(fpr, tpr, thresholds)
    current_l3 = _point_from_threshold(y_true, scores, threshold=30.0)
    if not np.isfinite(youden["threshold"]):
        finite_mask = np.isfinite(thresholds)
        if finite_mask.any():
            youden = _youden_best(fpr[finite_mask], tpr[finite_mask], thresholds[finite_mask])

    rec_threshold = float(round((youden["threshold"] if np.isfinite(youden["threshold"]) else 30.0) * 2.0) / 2.0)
    if not np.isfinite(auc_val):
        gate_reco = "KEEP 30"
    elif auc_val < 0.60:
        gate_reco = "KEEP 30 (weak classifier)"
    elif abs(rec_threshold - 30.0) <= 2.0:
        gate_reco = "KEEP 30"
    elif rec_threshold < 30.0:
        gate_reco = f"LOWER TO {rec_threshold:.1f}"
    else:
        gate_reco = f"RAISE TO {rec_threshold:.1f}"

    summary = pd.DataFrame([{
        "n_days": int(len(work)),
        "positive_rate_gt_1p0": float(y_true.mean()) if len(y_true) else np.nan,
        "auc": auc_val,
        "optimal_threshold_youden": youden["threshold"],
        "optimal_tpr": youden["tpr"],
        "optimal_fpr": youden["fpr"],
        "optimal_youden_j": youden["j"],
        "current_l3_threshold": 30.0,
        "current_l3_tpr": current_l3["tpr"],
        "current_l3_fpr": current_l3["fpr"],
        "recommendation": gate_reco,
    }])
    _write_csv(summary, OUTPUT_DIR / "A7_roc_analysis.csv", index=False)
    report.table(summary, index=False)
    report.line(f"Optimal VIX threshold by Youden: {youden['threshold']:.2f} (AUC={auc_val:.3f}). Current L3=30. Recommendation: {gate_reco}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fpr, tpr, color="#4C78A8", linewidth=2, label=f"ROC (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random")
    if np.isfinite(youden["fpr"]) and np.isfinite(youden["tpr"]):
        ax.scatter([youden["fpr"]], [youden["tpr"]], color="#E15759", s=60, label=f"Youden @ {youden['threshold']:.1f}")
    if np.isfinite(current_l3["fpr"]) and np.isfinite(current_l3["tpr"]):
        ax.scatter([current_l3["fpr"]], [current_l3["tpr"]], color="#59A14F", s=60, label="Current L3=30")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("A.7 ROC: VIX_prev_close predicting max_directional > 1.0%")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart_A7_roc_curve.png", dpi=180)
    plt.close(fig)
    return {"summary": summary.iloc[0].to_dict(), "fpr": fpr, "tpr": tpr, "thresholds": thresholds}


def _run_module_a(report: Reporter) -> Dict[str, Any]:
    raw = _load_module_a_market_data(report)
    df = _prepare_module_a_frame(raw, report)
    return {
        "raw": raw,
        "df": df,
        "A2": _module_a_a2_exceedance(df, report),
        "A3": _module_a_a3_distributions(df, report),
        "A4": _module_a_a4_asymmetry(df, report, threshold=1.0),
        "A5": _module_a_a5_dow_given_vix(df, report),
        "A6": _module_a_a6_temporal_stability(df, report),
        "A7": _module_a_a7_roc(df, report),
    }


def _prepare_module_b_frame(results: pd.DataFrame, report: Reporter) -> pd.DataFrame:
    report.section("MODULE B Prep: Normalize Backtester Output")
    if PNL_COL not in results.columns:
        raise ValueError(f"Expected PnL column '{PNL_COL}' not found in backtest output")

    df = results.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    for col in ["vix", "spx_open", "spx_high", "spx_low", "entry_credit", "entry_credit_usd", PNL_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    mask = (
        (~df["skipped"].astype(bool))
        & np.isfinite(df["vix"])
        & np.isfinite(df["spx_open"])
        & np.isfinite(df["spx_high"])
        & np.isfinite(df["spx_low"])
        & np.isfinite(df[PNL_COL])
    )
    df = df.loc[mask].copy()
    if df.empty:
        raise ValueError("No tradable rows in backtester output after filtering")

    for col in ["put_breached", "call_breached", "put_wing_breached", "call_wing_breached"]:
        df[col] = df[col].astype(bool)

    df["vix_bucket"] = _bucketize_vix(df["vix"]).astype(VIX_BUCKET_CATEGORIES)
    df = df[~df["vix_bucket"].isna()].copy()
    df["any_short_touch"] = df["put_breached"] | df["call_breached"]
    df["any_wing_breach"] = df["put_wing_breached"] | df["call_wing_breached"]
    df["pnl_tp50_sl"] = df[PNL_COL].astype(float)
    df["loss_trade"] = df["pnl_tp50_sl"] < 0

    credit_usd = pd.to_numeric(df.get("entry_credit_usd"), errors="coerce") if "entry_credit_usd" in df.columns else pd.Series(np.nan, index=df.index)
    if credit_usd.isna().all() and "entry_credit" in df.columns:
        credit_usd = pd.to_numeric(df["entry_credit"], errors="coerce") * 100.0
    df["entry_credit_usd_eff"] = credit_usd
    df["abs_loss_x_entry_credit"] = np.where(
        (df["loss_trade"]) & (df["entry_credit_usd_eff"] > 0),
        np.abs(df["pnl_tp50_sl"]) / df["entry_credit_usd_eff"],
        np.nan,
    )

    spot = df["spx_open"].astype(float)
    df["mae_put_pct"] = np.maximum(0.0, df["short_put"].astype(float) - df["spx_low"].astype(float)) / spot * 100.0
    df["mae_call_pct"] = np.maximum(0.0, df["spx_high"].astype(float) - df["short_call"].astype(float)) / spot * 100.0
    df["mae_max_pct"] = np.maximum(df["mae_put_pct"], df["mae_call_pct"])

    report.line(f"Module B analysis rows: {len(df)}")
    counts = df.groupby("vix_bucket", observed=True).size().reindex(VIX_BUCKET_LABELS, fill_value=0).rename("n_trades").to_frame()
    report.table(counts, index=True, float_fmt="{:.0f}")
    return df


def _module_b_b1_breach_by_vix(df: pd.DataFrame, report: Reporter) -> Dict[str, Any]:
    report.section("MODULE B.1: Breach Rate by VIX Bucket")
    grouped = df.groupby("vix_bucket", observed=True)
    table = pd.DataFrame(
        {
            "n_trades": grouped.size(),
            "short_touch_rate_put": grouped["put_breached"].mean(),
            "short_touch_rate_call": grouped["call_breached"].mean(),
            "wing_breach_rate_put": grouped["put_wing_breached"].mean(),
            "wing_breach_rate_call": grouped["call_wing_breached"].mean(),
            "any_short_touch_rate": grouped["any_short_touch"].mean(),
            "any_wing_breach_rate": grouped["any_wing_breach"].mean(),
        }
    ).reindex(VIX_BUCKET_LABELS)
    _write_csv(table, OUTPUT_DIR / "B1_breach_by_vix.csv")
    report.table(table, index=True)

    stats: Dict[str, float] = {}
    for label, col in [("any_short_touch", "any_short_touch"), ("any_wing_breach", "any_wing_breach")]:
        agg = (
            df.groupby("vix_bucket", observed=True)[col]
            .agg(total="count", breach="sum")
            .reindex(VIX_BUCKET_LABELS)
            .fillna(0)
        )
        agg["total"] = agg["total"].astype(int)
        agg["breach"] = agg["breach"].astype(int)
        agg["no_breach"] = (agg["total"] - agg["breach"]).astype(int)
        ct = agg.loc[agg["total"] > 0, ["no_breach", "breach"]]
        chi2, p, dof, cv = _cramers_v_from_contingency(ct)
        stats[f"{label}_chi2"] = chi2
        stats[f"{label}_p_value"] = p
        stats[f"{label}_dof"] = dof
        stats[f"{label}_cramers_v"] = cv
        report.line(f"{label}: chi2={chi2:.4f}, dof={dof}, p={p:.6f}, Cramer's V={cv:.4f}")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(VIX_BUCKET_LABELS))
    width = 0.35
    ax.bar(x - width / 2, table["any_short_touch_rate"].fillna(0).to_numpy(dtype=float), width, label="Any short touch", color="#4C78A8")
    ax.bar(x + width / 2, table["any_wing_breach_rate"].fillna(0).to_numpy(dtype=float), width, label="Any wing breach", color="#E15759")
    ax.set_xticks(x)
    ax.set_xticklabels(VIX_BUCKET_LABELS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_xlabel("VIX bucket")
    ax.set_title("B.1 Backtester Breach Rates by VIX Bucket (wing=2, tp50_sl_capped)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart_B1_breach_rates.png", dpi=180)
    plt.close(fig)
    return {"table": table, "stats": stats}


def _module_b_b2_loss_severity(df: pd.DataFrame, report: Reporter) -> Dict[str, Any]:
    report.section("MODULE B.2: Loss Severity by VIX Bucket")
    touched = df[df["any_short_touch"]].copy()
    losses = touched[touched["loss_trade"]].copy()
    report.line(f"Touched-short trades: {len(touched)}")
    report.line(f"Actual losses ({PNL_COL} < 0): {len(losses)}")

    touch_counts = touched.groupby("vix_bucket", observed=True).size().rename("n_touched")
    grouped = losses.groupby("vix_bucket", observed=True)
    table = pd.DataFrame(
        {
            "n_touched": touch_counts,
            "n_losses": grouped.size(),
            "mean_pnl_usd": grouped["pnl_tp50_sl"].mean(),
            "median_pnl_usd": grouped["pnl_tp50_sl"].median(),
            "worst_pnl_usd": grouped["pnl_tp50_sl"].min(),
            "pct_losses_wing_breached": grouped["any_wing_breach"].mean(),
            "mean_abs_loss_x_entry_credit": grouped["abs_loss_x_entry_credit"].mean(),
        }
    ).reindex(VIX_BUCKET_LABELS)
    _write_csv(table, OUTPUT_DIR / "B2_loss_severity.csv")
    report.table(table, index=True)

    fig, ax1 = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(VIX_BUCKET_LABELS))
    width = 0.36
    ax1.bar(x - width / 2, table["mean_pnl_usd"].fillna(0).to_numpy(dtype=float), width, label="Mean loss PnL", color="#E15759")
    ax1.bar(x + width / 2, table["median_pnl_usd"].fillna(0).to_numpy(dtype=float), width, label="Median loss PnL", color="#F28E2B")
    ax1.set_xticks(x)
    ax1.set_xticklabels(VIX_BUCKET_LABELS)
    ax1.set_ylabel("PnL USD (losses only)")
    ax1.set_title("B.2 Loss Severity by VIX Bucket (tp50_sl_capped)")

    ax2 = ax1.twinx()
    ax2.plot(x, table["pct_losses_wing_breached"].to_numpy(dtype=float), marker="o", linewidth=2, color="#4C78A8", label="% losses with wing breach")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Wing breach rate among losses")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart_B2_loss_severity.png", dpi=180)
    plt.close(fig)
    return {"table": table, "losses": losses}


def _module_b_b3_mae_proxy(df: pd.DataFrame, report: Reporter) -> Dict[str, Any]:
    report.section("MODULE B.3: MAE Proxy (Underlying)")
    grouped = df.groupby("vix_bucket", observed=True)["mae_max_pct"]
    table = grouped.agg(
        n="count",
        mean="mean",
        median="median",
        p90=lambda s: s.quantile(0.90),
        p99=lambda s: s.quantile(0.99),
    ).reindex(VIX_BUCKET_LABELS)
    _write_csv(table, OUTPUT_DIR / "B3_mae_proxy.csv")
    report.table(table, index=True)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(VIX_BUCKET_LABELS))
    ax.bar(x, table["mean"].fillna(0).to_numpy(dtype=float), color=[BUCKET_COLORS[b] for b in VIX_BUCKET_LABELS], alpha=0.8, label="Mean")
    ax.plot(x, table["p90"].to_numpy(dtype=float), color="#1F1F1F", marker="o", linewidth=2, label="P90")
    ax.plot(x, table["p99"].to_numpy(dtype=float), color="#7A1E1E", marker="s", linewidth=2, label="P99")
    ax.set_xticks(x)
    ax.set_xticklabels(VIX_BUCKET_LABELS)
    ax.set_ylabel("MAE proxy (% intrusion past short strike)")
    ax.set_xlabel("VIX bucket")
    ax.set_title("B.3 MAE Proxy by VIX Bucket")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart_B3_mae_by_vix.png", dpi=180)
    plt.close(fig)
    return {"table": table}


def _module_b_hour_feasibility(results: pd.DataFrame, report: Reporter) -> Dict[str, Any]:
    report.section("Module B (Feasibility): Entry Hour Availability Check")
    trade_fields = [f.name for f in fields(sb.TradeResult)]
    result_cols = list(results.columns)
    time_cols = [c for c in result_cols if c != "date" and any(tok in c.lower() for tok in ("hour", "time", "timestamp"))]
    trade_time_fields = [f for f in trade_fields if f != "date" and any(tok in f.lower() for tok in ("hour", "time", "timestamp"))]
    viable_cols: List[str] = []
    for c in time_cols:
        s = pd.to_numeric(results[c], errors="coerce")
        if s.notna().sum() > 0 and s.nunique(dropna=True) > 1:
            viable_cols.append(c)
    if not viable_cols:
        msg = (
            "Hour analysis not feasible with current backtester outputs. "
            f"TradeResult fields: {trade_fields}. "
            "Recommend adding per-trade entry_hour or intraday path timestamps."
        )
        report.line(msg)
        return {"feasible": False, "message": msg, "trade_fields": trade_fields, "result_time_fields": time_cols, "trade_time_fields": trade_time_fields}
    report.line(f"Hour analysis feasible with columns: {viable_cols}")
    return {"feasible": True, "columns": viable_cols, "trade_fields": trade_fields}


def _module_b_b4_cross_validation(a_ctx: Optional[Dict[str, Any]], b1: Dict[str, Any], report: Reporter) -> Dict[str, Any]:
    report.section("MODULE B.4: Cross-Validation (Module A vs Module B)")
    if not a_ctx:
        report.line("Skipped: Module A unavailable, so no raw-data proxy to compare.")
        empty = pd.DataFrame(columns=["vix_bucket", "module_a_pred_exceed_1p0", "module_b_actual_any_short_touch", "abs_diff"])
        empty.to_csv(OUTPUT_DIR / "B4_cross_validation.csv", index=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Module A unavailable\nCross-validation skipped", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "chart_B4_predicted_vs_actual.png", dpi=180)
        plt.close(fig)
        return {"table": empty, "pearson_r": np.nan}

    a2_table: pd.DataFrame = a_ctx["A2"]["table"]
    b1_table: pd.DataFrame = b1["table"]
    merged = pd.DataFrame(index=VIX_BUCKET_LABELS)
    merged["n_days_A"] = a2_table.get("n_days")
    merged["module_a_pred_exceed_1p0"] = a2_table["p_exceed_1p0"]
    merged["n_trades_B"] = b1_table.get("n_trades")
    merged["module_b_actual_any_short_touch"] = b1_table.get("any_short_touch_rate")
    merged["module_b_actual_any_wing_breach"] = b1_table.get("any_wing_breach_rate")
    merged["abs_diff"] = (merged["module_a_pred_exceed_1p0"] - merged["module_b_actual_any_short_touch"]).abs()
    merged = merged.reset_index().rename(columns={"index": "vix_bucket"})
    merged.to_csv(OUTPUT_DIR / "B4_cross_validation.csv", index=False)
    report.table(merged.set_index("vix_bucket"), index=True)

    r_val = _pearsonr_safe(merged["module_a_pred_exceed_1p0"], merged["module_b_actual_any_short_touch"])
    report.line(f"Pearson r (A predicted exceedance vs B actual any short touch): {r_val:.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    xs = merged["module_a_pred_exceed_1p0"].to_numpy(dtype=float)
    ys = merged["module_b_actual_any_short_touch"].to_numpy(dtype=float)
    for _, row in merged.iterrows():
        x = row["module_a_pred_exceed_1p0"]
        y = row["module_b_actual_any_short_touch"]
        if np.isfinite(x) and np.isfinite(y):
            bucket = str(row["vix_bucket"])
            ax.scatter(x, y, s=70, color=BUCKET_COLORS.get(bucket, "#333333"))
            ax.text(x, y, bucket, fontsize=8, ha="left", va="bottom")
    finite = np.isfinite(xs) & np.isfinite(ys)
    if finite.any():
        lo = float(min(xs[finite].min(), ys[finite].min()))
        hi = float(max(xs[finite].max(), ys[finite].max()))
        pad = max(0.02, 0.05 * (hi - lo if hi > lo else 1.0))
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="gray", linewidth=1, label="y=x")
        ax.set_xlim(max(0, lo - pad), min(1, hi + pad))
        ax.set_ylim(max(0, lo - pad), min(1, hi + pad))
    ax.set_xlabel("Module A: P(max_directional > 1.0%)")
    ax.set_ylabel("Module B: any short strike touch rate")
    ax.set_title(f"B.4 Module A vs B Cross-Validation (r={r_val:.2f})")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart_B4_predicted_vs_actual.png", dpi=180)
    plt.close(fig)
    return {"table": merged, "pearson_r": r_val}


def _run_module_b(report: Reporter, a_ctx: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    report.section("MODULE B: Backtester Strategy Outcomes")
    cfg_raw = _load_yaml_config(CONFIG_PATH)
    cfg = _apply_backtest_overrides(cfg_raw)
    source_data = _load_backtester_source_data(cfg, report)
    results = _run_backtester(cfg, source_data, report)
    df = _prepare_module_b_frame(results, report)
    b1 = _module_b_b1_breach_by_vix(df, report)
    b2 = _module_b_b2_loss_severity(df, report)
    b3 = _module_b_b3_mae_proxy(df, report)
    hour_info = _module_b_hour_feasibility(results, report)
    b4 = _module_b_b4_cross_validation(a_ctx, b1, report)
    return {
        "cfg": cfg,
        "source_data": source_data,
        "results": results,
        "df": df,
        "B1": b1,
        "B2": b2,
        "B3": b3,
        "B4": b4,
        "hour_info": hour_info,
    }


def _weighted_bucket_rate(table: pd.DataFrame, rate_col: str, buckets: Iterable[str], n_col: str = "n_trades") -> float:
    num = 0.0
    den = 0.0
    for b in buckets:
        if b not in table.index:
            continue
        n = table.at[b, n_col] if n_col in table.columns else np.nan
        r = table.at[b, rate_col] if rate_col in table.columns else np.nan
        if pd.notna(n) and pd.notna(r):
            num += float(n) * float(r)
            den += float(n)
    return _safe_div(num, den)


def _classify_regime_viability(rate: float) -> str:
    if not np.isfinite(rate):
        return "UNKNOWN"
    if rate < 0.05:
        return "VIABLE"
    if rate < 0.12:
        return "MARGINAL"
    return "UNVIABLE"


def _actionable_conclusions(a_ctx: Optional[Dict[str, Any]], b_ctx: Optional[Dict[str, Any]], report: Reporter) -> None:
    report.line("============================================================")
    report.line("VIX x BREACH ANALYSIS - CONSOLIDATED FINDINGS")
    report.line("============================================================")
    report.line()

    if a_ctx is None:
        report.line("MODULE A: SPX MARKET DATA")
        report.line("---------------------------------------------------")
        report.line("FAILED (see report sections above for error)")
        report.line()
    else:
        a_df = a_ctx["df"]
        a2_table = a_ctx["A2"]["table"]
        a7_summary = a_ctx["A7"]["summary"]
        report.line(f"MODULE A: SPX MARKET DATA (N={len(a_df)} days, 2005-present, VIX_prev_close buckets)")
        report.line("---------------------------------------------------")
        report.line("[A.2] Exceedance probability at 1.0% threshold:")
        for bucket in VIX_BUCKET_LABELS:
            val = a2_table.at[bucket, "p_exceed_1p0"] if bucket in a2_table.index else np.nan
            report.line(f"      VIX {bucket}: {_format_pct(val)}")
        down_share = a_ctx["A4"]["overall_down_share"]
        report.line(f"[A.4] Directional asymmetry (down/total at 1.0%): {_format_pct(down_share)}")
        report.line(f"      Confirms put-side dominance: {'YES' if (np.isfinite(down_share) and down_share >= 0.60) else 'NO'}")
        auc_val = float(a7_summary.get("auc", np.nan))
        opt_thr = float(a7_summary.get("optimal_threshold_youden", np.nan))
        a7_reco = str(a7_summary.get("recommendation", "N/A"))
        report.line(f"[A.7] Optimal VIX threshold (ROC): {opt_thr:.2f}")
        if np.isfinite(opt_thr):
            if abs(opt_thr - 30.0) <= 2.0:
                current_label = "OPTIMAL"
            elif opt_thr < 28.0:
                current_label = "TOO HIGH"
            else:
                current_label = "TOO LOW"
        else:
            current_label = "UNKNOWN"
        report.line(f"      Current L3=30 is: {current_label} (AUC={auc_val:.3f}; {a7_reco})")
        report.line()

    if b_ctx is None:
        report.line("MODULE B: BACKTESTER OUTCOMES")
        report.line("---------------------------------------------------")
        report.line("FAILED (see report sections above for error)")
        report.line()
    else:
        b_df = b_ctx["df"]
        b1_table = b_ctx["B1"]["table"]
        b4_r = float(b_ctx["B4"].get("pearson_r", np.nan))
        report.line(f"MODULE B: BACKTESTER OUTCOMES (N={len(b_df)} trades, wing=2)")
        report.line("---------------------------------------------------")
        report.line("[B.1] Wing breach rate:")
        report.line(f"      VIX <20:  {_format_pct(_weighted_bucket_rate(b1_table, 'any_wing_breach_rate', ['0-15', '15-20']))}")
        report.line(f"      VIX 20-25: {_format_pct(_weighted_bucket_rate(b1_table, 'any_wing_breach_rate', ['20-25']))}")
        report.line(f"      VIX 25-30: {_format_pct(_weighted_bucket_rate(b1_table, 'any_wing_breach_rate', ['25-30']))}")
        report.line(f"      VIX 30+:   {_format_pct(_weighted_bucket_rate(b1_table, 'any_wing_breach_rate', ['30-40', '40+']))}")
        losses = b_ctx["B2"]["losses"]
        wing_among_losses = float(losses["any_wing_breach"].mean()) if len(losses) else np.nan
        hv_losses = losses[losses["vix"] > 25] if len(losses) else losses
        wing_hv = float(hv_losses["any_wing_breach"].mean()) if len(hv_losses) else np.nan
        sl_reliability = "UNRELIABLE in VIX>25" if (np.isfinite(wing_hv) and wing_hv >= 0.50) else "RELIABLE"
        report.line(f"[B.2] When loss occurs, wing also breached: {_format_pct(wing_among_losses)} of losses")
        report.line(f"      -> SL reliability: {sl_reliability}")
        report.line(f"[B.4] Module A predicts Module B: r={b4_r:.2f}")
        report.line(f"      -> Raw SPX data {'SUFFICIENT' if (np.isfinite(b4_r) and b4_r > 0.80) else 'INSUFFICIENT'} as proxy")
        report.line()

    report.line("============================================================")
    report.line("ACTIONABLE CONCLUSIONS")
    report.line("============================================================")

    if a_ctx is None:
        report.line("[1] VIX gate threshold: KEEP 30")
        report.line("    Evidence: Module A unavailable; cannot estimate ROC threshold this run")
    else:
        a7 = a_ctx["A7"]["summary"]
        opt = float(a7.get("optimal_threshold_youden", np.nan))
        auc_val = float(a7.get("auc", np.nan))
        high_vs_low_note = ""
        if b_ctx is not None:
            b1_table = b_ctx["B1"]["table"]
            low_rate = _weighted_bucket_rate(b1_table, "any_wing_breach_rate", ["0-15", "15-20"])
            high_rate = _weighted_bucket_rate(b1_table, "any_wing_breach_rate", ["25-30", "30-40", "40+"])
            high_vs_low_note = f"; B1 any wing breach VIX>=25 {_format_pct(high_rate)} vs VIX<20 {_format_pct(low_rate)}"
        if not np.isfinite(auc_val) or auc_val < 0.60:
            gate_conclusion = "KEEP 30"
        elif np.isfinite(opt) and opt < 28.0:
            rounded = round(opt * 2.0) / 2.0
            gate_conclusion = "LOWER TO 25" if rounded == 25.0 else f"LOWER TO {rounded:.1f}"
        elif np.isfinite(opt) and opt > 32.0:
            gate_conclusion = f"RAISE TO {round(opt * 2.0) / 2.0:.1f}"
        else:
            gate_conclusion = "KEEP 30"
        report.line(f"[1] VIX gate threshold: {gate_conclusion}")
        report.line(f"    Evidence: A7 ROC optimal={opt:.2f}, AUC={auc_val:.3f}{high_vs_low_note}")

    if b_ctx is None:
        report.line("[2] Wing=2 viability by regime:")
        report.line("    Module B unavailable")
    else:
        b1_table = b_ctx["B1"]["table"]
        report.line("[2] Wing=2 viability by regime:")
        for label, buckets in [("VIX <20", ["0-15", "15-20"]), ("VIX 20-25", ["20-25"]), ("VIX 25-30", ["25-30"])]:
            r = _weighted_bucket_rate(b1_table, "any_wing_breach_rate", buckets)
            report.line(f"    {label}: {_classify_regime_viability(r)} (wing breach {_format_pct(r)})")

    if a_ctx is None:
        report.line("[3] Put-side asymmetry: UNKNOWN")
        report.line("    Implication for future: none (Module A unavailable)")
    else:
        down_share = float(a_ctx["A4"]["overall_down_share"])
        asym_label = "CONFIRMED" if np.isfinite(down_share) and down_share >= 0.60 else "NOT CONFIRMED"
        implication = "consider asymmetric wings/delta on put side in future research" if asym_label == "CONFIRMED" else "no asymmetry change justified yet"
        report.line(f"[3] Put-side asymmetry: {asym_label}")
        report.line(f"    Implication for future: {implication} (down-side share={_format_pct(down_share)})")

    if a_ctx is None:
        report.line("[4] Day-of-week after VIX: UNKNOWN")
    else:
        dow_table = a_ctx["A5"]["table"]
        valid_p = dow_table["p_value"].dropna()
        report.line(f"[4] Day-of-week after VIX: {'SUBSUMED' if (not valid_p.empty and bool((valid_p > 0.10).all())) else 'INDEPENDENT'}")

    if a_ctx is None:
        report.line("[5] Temporal stability: UNKNOWN")
        report.line("    If unstable: n/a")
    else:
        a6 = a_ctx["A6"]
        stability_label = str(a6.get('stability_label', 'UNKNOWN'))
        ranges = a6.get("per_bucket_ranges", {})
        worst_bucket = None
        worst_val = -np.inf
        for b, val in ranges.items():
            if np.isfinite(val) and val > worst_val:
                worst_bucket = b
                worst_val = float(val)
        reason = "insufficient sample per era" if worst_bucket is None else f"largest bucket drift in {worst_bucket} (range={worst_val:.3f} in P(exceed>1.0%))"
        report.line(f"[5] Temporal stability: {stability_label}")
        report.line(f"    If unstable: {reason}")

    if b_ctx is not None and bool(b_ctx.get("hour_info", {}).get("feasible", False)):
        report.line("[6] Hour analysis: FEASIBLE")
    else:
        report.line("[6] Hour analysis: NOT FEASIBLE - requires intraday data")
    report.line("============================================================")


def _save_artifact_manifest(report: Reporter) -> None:
    report.line()
    report.line("Artifacts")
    report.line("---------")
    for name in [
        "A2_exceedance_table.csv",
        "chart_A2_exceedance_heatmap.png",
        "A3_range_percentiles.csv",
        "chart_A3_range_distributions.png",
        "A4_directional_asymmetry.csv",
        "chart_A4_asymmetry_by_vix.png",
        "A5_dow_given_vix.csv",
        "A6_temporal_stability.csv",
        "chart_A6_stability_by_era.png",
        "A7_roc_analysis.csv",
        "chart_A7_roc_curve.png",
        "B1_breach_by_vix.csv",
        "chart_B1_breach_rates.png",
        "B2_loss_severity.csv",
        "chart_B2_loss_severity.png",
        "B3_mae_proxy.csv",
        "chart_B3_mae_by_vix.png",
        "B4_cross_validation.csv",
        "chart_B4_predicted_vs_actual.png",
        "vix_breach_eda_report.txt",
    ]:
        report.line(f"- {OUTPUT_DIR / name}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _set_plot_style()
    report = Reporter()
    report.line("VIX x Intraday Range x Breach Analysis (Dual Module)")
    report.line(f"Repo root: {REPO_ROOT}")
    report.line(f"Output dir: {OUTPUT_DIR}")

    a_ctx: Optional[Dict[str, Any]] = None
    b_ctx: Optional[Dict[str, Any]] = None
    try:
        a_ctx = _run_module_a(report)
    except Exception as exc:
        report.section("MODULE A ERROR")
        report.line(f"Module A failed: {exc.__class__.__name__}: {exc}")
    try:
        b_ctx = _run_module_b(report, a_ctx=a_ctx)
    except Exception as exc:
        report.section("MODULE B ERROR")
        report.line(f"Module B failed: {exc.__class__.__name__}: {exc}")
        report.line("Module A outputs (if any) remain valid and were not rolled back.")

    report.section("Final Summary")
    _actionable_conclusions(a_ctx=a_ctx, b_ctx=b_ctx, report=report)
    _save_artifact_manifest(report)

    report_path = OUTPUT_DIR / "vix_breach_eda_report.txt"
    report.save(report_path)
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
