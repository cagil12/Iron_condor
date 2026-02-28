"""Intraday range timing EDA for SPX/XSP 0DTE risk timing decisions.

Standalone CLI:
    python -m src.research.hourly_range_eda
    python src/research/hourly_range_eda.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

try:
    import yfinance as yf
except Exception as exc:  # pragma: no cover
    raise RuntimeError("yfinance is required for hourly_range_eda.py") from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "outputs" / "hourly_range_eda"
CACHE_PATH = OUTPUT_DIR / "_intraday_cache.csv"
ET_TZ = ZoneInfo("America/New_York")

VIX_BUCKETS = [0.0, 15.0, 20.0, 25.0, np.inf]
VIX_BUCKET_LABELS = ["0-15", "15-20", "20-25", "25+"]
VIX_BUCKET_DTYPE = pd.CategoricalDtype(categories=VIX_BUCKET_LABELS, ordered=True)
BUCKET_COLORS = {
    "0-15": "#4C78A8",
    "15-20": "#59A14F",
    "20-25": "#F28E2B",
    "25+": "#E15759",
}

CHECKPOINTS = ["10:00", "10:30", "11:00", "11:30", "12:00", "12:30", "13:00", "13:30", "14:00", "14:30", "15:00", "15:30", "16:00"]
ENTRY_TIMES = ["10:00", "10:30", "11:00", "11:30", "12:00"]
BLOCKS = [
    ("09:30", "10:00", "09:30-10:00", "open_volatility"),
    ("10:00", "10:30", "10:00-10:30", "early_entry_window"),
    ("10:30", "11:00", "10:30-11:00", "late_entry_window"),
    ("11:00", "12:00", "11:00-12:00", "mid_morning"),
    ("12:00", "13:00", "12:00-13:00", "lunch"),
    ("13:00", "14:00", "13:00-14:00", "early_afternoon"),
    ("14:00", "15:00", "14:00-15:00", "fomc_window"),
    ("15:00", "16:00", "15:00-16:00", "close_gamma"),
]

RESIDUAL_THRESHOLDS = [1.0, 1.3]


@dataclass
class IntradayDownloadResult:
    symbol_used: str
    bars: pd.DataFrame
    notes: List[str]
    interval_days: Dict[int, int]
    interval_bars: Dict[int, int]


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
        out = df.copy()
        for c in out.columns:
            if pd.api.types.is_float_dtype(out[c]):
                out[c] = out[c].map(lambda x: "" if pd.isna(x) else float_fmt.format(float(x)))
        self.line(out.to_string(index=index))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def _set_plot_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass
    plt.rcParams.update(
        {
            "figure.figsize": (12, 6),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.frameon": True,
        }
    )


def _fmt_pct(x: float) -> str:
    return "NA" if not np.isfinite(x) else f"{x:.1%}"


def _safe_div(num: float, den: float) -> float:
    if den == 0 or not np.isfinite(den):
        return float("nan")
    return float(num) / float(den)


def _write_csv(df: pd.DataFrame, path: Path, index: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def _bucketize_vix(vix_series: pd.Series) -> pd.Categorical:
    return pd.cut(vix_series, bins=VIX_BUCKETS, labels=VIX_BUCKET_LABELS, right=False, include_lowest=True)


def _time_str_to_time(s: str) -> pd.Timestamp:
    return pd.Timestamp(f"2000-01-01 {s}")


def _time_to_minutes(s: str) -> int:
    t = _time_str_to_time(s)
    return int(t.hour * 60 + t.minute)


def _decimal_hour(ts: pd.Timestamp) -> float:
    return float(ts.hour + ts.minute / 60.0)


def _extract_ohlcv(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
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
    mapping = {"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}
    for col in ["open", "high", "low", "close"]:
        if col not in data.columns:
            raise ValueError(f"Missing {col} for {symbol}")
    if "volume" not in data.columns:
        data["volume"] = np.nan
    out = data[list(mapping.keys())].rename(columns=mapping)
    return out


def _download_intraday(symbol: str, interval: str, period: str) -> pd.DataFrame:
    raw = yf.download(symbol, interval=interval, period=period, auto_adjust=False, progress=False, threads=False)
    out = _extract_ohlcv(raw, symbol=symbol)
    if out.empty:
        return out
    idx = pd.to_datetime(out.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx_et = idx.tz_convert(ET_TZ).tz_localize(None)
    out = out.copy()
    out["timestamp_et"] = idx_et
    out = out.reset_index(drop=True)
    return out


def _prepare_intraday_bars(bars: pd.DataFrame, symbol: str, interval_min: int, label: str) -> pd.DataFrame:
    if bars.empty:
        return bars.copy()
    df = bars.copy()
    df["symbol"] = symbol
    df["interval_min"] = int(interval_min)
    df["source_label"] = label
    df["timestamp_et"] = pd.to_datetime(df["timestamp_et"])
    df = df.sort_values("timestamp_et").drop_duplicates(subset=["timestamp_et"], keep="last").reset_index(drop=True)
    df["date"] = df["timestamp_et"].dt.normalize()
    df["time_str"] = df["timestamp_et"].dt.strftime("%H:%M")
    # Bar timestamps are bar-start timestamps from Yahoo. bar_end_et helps checkpoint logic.
    df["bar_end_et"] = df["timestamp_et"] + pd.to_timedelta(interval_min, unit="m")
    # Filter regular trading hours by bar-start within [09:30,16:00). Final 5m bar is 15:55.
    mins = df["timestamp_et"].dt.hour * 60 + df["timestamp_et"].dt.minute
    rth_mask = (mins >= 570) & (mins < 960)  # 09:30 to <16:00 ET
    df = df.loc[rth_mask].copy()
    df = df[df["timestamp_et"].dt.dayofweek < 5].copy()
    return df.reset_index(drop=True)


def _download_symbol_bundle(symbol: str, report: Reporter) -> IntradayDownloadResult:
    notes: List[str] = []
    errors: List[str] = []

    # 5m longer-history attempt (Yahoo currently often limits to 60d despite historical claims)
    five_candidates = ["730d", "60d"]
    five_df = pd.DataFrame()
    for period in five_candidates:
        try:
            raw = _download_intraday(symbol, interval="5m", period=period)
            if raw.empty:
                notes.append(f"{symbol} 5m period={period}: empty")
                continue
            five_df = _prepare_intraday_bars(raw, symbol=symbol, interval_min=5, label=f"5m_{period}")
            notes.append(f"{symbol} 5m period={period}: {len(five_df)} RTH bars")
            break
        except Exception as exc:  # pragma: no cover
            errors.append(f"{symbol} 5m period={period}: {exc.__class__.__name__}: {exc}")

    # 1m recent-detail attempt
    one_candidates = ["60d", "30d", "7d"]
    one_df = pd.DataFrame()
    for period in one_candidates:
        try:
            raw = _download_intraday(symbol, interval="1m", period=period)
            if raw.empty:
                notes.append(f"{symbol} 1m period={period}: empty")
                continue
            one_df = _prepare_intraday_bars(raw, symbol=symbol, interval_min=1, label=f"1m_{period}")
            notes.append(f"{symbol} 1m period={period}: {len(one_df)} RTH bars")
            break
        except Exception as exc:  # pragma: no cover
            errors.append(f"{symbol} 1m period={period}: {exc.__class__.__name__}: {exc}")

    if five_df.empty and one_df.empty:
        raise RuntimeError(" | ".join(errors or [f"No intraday data returned for {symbol}"]))

    # Combine with 1m overriding overlapping dates from 5m.
    if not one_df.empty:
        one_dates = set(pd.to_datetime(one_df["date"]).dt.normalize())
        if not five_df.empty:
            five_df = five_df[~five_df["date"].isin(one_dates)].copy()
    combined = pd.concat([d for d in [five_df, one_df] if not d.empty], ignore_index=True)
    combined = combined.sort_values(["date", "timestamp_et"]).reset_index(drop=True)

    interval_days = (
        combined.groupby("interval_min", observed=True)["date"].nunique().to_dict()
        if not combined.empty
        else {}
    )
    interval_bars = (
        combined.groupby("interval_min", observed=True).size().to_dict()
        if not combined.empty
        else {}
    )
    if errors:
        notes.extend([f"ERROR: {e}" for e in errors])
    return IntradayDownloadResult(
        symbol_used=symbol,
        bars=combined,
        notes=notes,
        interval_days={int(k): int(v) for k, v in interval_days.items()},
        interval_bars={int(k): int(v) for k, v in interval_bars.items()},
    )


def _download_daily_vix(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    # Daily bars; Yahoo end is exclusive -> add 1 day.
    start = (pd.to_datetime(start_date) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end = (pd.to_datetime(end_date) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    raw = yf.download("^VIX", start=start, end=end, interval="1d", auto_adjust=False, progress=False, threads=False)
    vix = _extract_ohlcv(raw, "^VIX")
    if vix.empty:
        raise ValueError("No daily VIX data returned")
    idx = pd.to_datetime(vix.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx_et = idx.tz_convert(ET_TZ).tz_localize(None).normalize()
    vix = vix.copy()
    vix["date"] = idx_et
    vix = vix.reset_index(drop=True)
    vix = vix.rename(columns={"close": "vix_close", "open": "vix_open", "high": "vix_high", "low": "vix_low"})
    vix = vix[["date", "vix_open", "vix_high", "vix_low", "vix_close"]].sort_values("date").drop_duplicates("date", keep="last")
    vix["vix_prev_close"] = pd.to_numeric(vix["vix_close"], errors="coerce").shift(1)
    return vix.reset_index(drop=True)


def _merge_intraday_with_vix(intraday: pd.DataFrame, vix_daily: pd.DataFrame) -> pd.DataFrame:
    df = intraday.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    merged = df.merge(vix_daily, on="date", how="left")
    merged["vix_bucket"] = _bucketize_vix(pd.to_numeric(merged["vix_prev_close"], errors="coerce")).astype(VIX_BUCKET_DTYPE)
    return merged


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    # Store ISO timestamps (naive ET) for portability.
    out["timestamp_et"] = pd.to_datetime(out["timestamp_et"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    out["bar_end_et"] = pd.to_datetime(out["bar_end_et"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)


def _load_cache(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    for c in ["timestamp_et", "bar_end_et", "date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    num_cols = ["open", "high", "low", "close", "volume", "vix_close", "vix_prev_close", "vix_open", "vix_high", "vix_low", "interval_min"]
    for c in [c for c in num_cols if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "interval_min" in df.columns:
        df["interval_min"] = df["interval_min"].astype(int)
    if "vix_bucket" in df.columns:
        df["vix_bucket"] = pd.Categorical(df["vix_bucket"], categories=VIX_BUCKET_LABELS, ordered=True)
    return df


def _validate_intraday(df: pd.DataFrame, report: Reporter) -> Dict[str, Any]:
    report.section("Data Validation")
    if df.empty:
        report.line("No intraday rows available.")
        return {"n_days": 0, "n_bars": 0}

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d["timestamp_et"] = pd.to_datetime(d["timestamp_et"])
    d["bar_end_et"] = pd.to_datetime(d["bar_end_et"])

    min_ts = d["timestamp_et"].min()
    max_ts = d["timestamp_et"].max()
    n_days = int(d["date"].nunique())
    n_bars = int(len(d))
    report.line(f"Date range (ET): {min_ts.date()} -> {max_ts.date()}")
    report.line(f"Trading days: {n_days}")
    report.line(f"Bars: {n_bars}")
    report.line(f"Timezone handling: timestamps converted to ET and stored as ET-naive values")

    by_interval = (
        d.groupby("interval_min", observed=True)
        .agg(n_bars=("date", "size"), n_days=("date", "nunique"))
        .reset_index()
        .sort_values("interval_min")
    )
    if not by_interval.empty:
        report.line("Bars by interval:")
        report.table(by_interval, index=False, float_fmt="{:.0f}")

    # Gap diagnostics per day (expected bars within full RTH session).
    per_day = (
        d.groupby(["date", "interval_min"], observed=True)
        .size()
        .rename("bars")
        .reset_index()
    )
    per_day["expected_bars"] = np.where(per_day["interval_min"] == 1, 390, np.where(per_day["interval_min"] == 5, 78, np.nan))
    per_day["gap_bars"] = per_day["expected_bars"] - per_day["bars"]
    gap_days = per_day[per_day["gap_bars"] > 0].copy()
    report.line(f"Days with missing RTH bars (vs full session expectation): {len(gap_days)}")
    if not gap_days.empty:
        report.line("Sample gap days:")
        sample = gap_days.sort_values(["gap_bars", "date"], ascending=[False, True]).head(10)
        report.table(sample, index=False, float_fmt="{:.0f}")

    if n_days < 20:
        report.line("WARNING: Fewer than 20 trading days available. Results may be unstable.")

    return {
        "n_days": n_days,
        "n_bars": n_bars,
        "date_start": min_ts.date(),
        "date_end": max_ts.date(),
        "gap_days": int(len(gap_days)),
        "by_interval": by_interval,
    }


def _load_or_download_intraday(report: Reporter) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if CACHE_PATH.exists():
        try:
            cached = _load_cache(CACHE_PATH)
            if not cached.empty and {"timestamp_et", "bar_end_et", "date", "vix_close", "vix_prev_close"}.issubset(cached.columns):
                report.section("Data Acquisition")
                report.line(f"Using cache: {CACHE_PATH}")
                meta = _validate_intraday(cached, report)
                return cached, {"source": "cache", **meta}
        except Exception as exc:
            report.section("Data Acquisition")
            report.line(f"WARNING: failed to load cache ({exc.__class__.__name__}: {exc}). Re-downloading.")

    report.section("Data Acquisition")
    attempts: List[str] = []
    last_error: Optional[Exception] = None
    result: Optional[IntradayDownloadResult] = None
    for symbol in ["^GSPC", "^XSP"]:
        try:
            report.line(f"Attempting intraday download for {symbol} (5m long + 1m recent)")
            result = _download_symbol_bundle(symbol, report)
            break
        except Exception as exc:
            attempts.append(f"{symbol}: {exc.__class__.__name__}: {exc}")
            last_error = exc
            report.line(f"Failed {symbol}: {exc.__class__.__name__}: {exc}")

    if result is None:
        report.line("ERROR: Intraday download failed for SPX and XSP.")
        for msg in attempts:
            report.line(f"  - {msg}")
        raise RuntimeError("Intraday data acquisition failed") from last_error

    for note in result.notes:
        report.line(f"  {note}")

    bars = result.bars
    if bars.empty:
        raise RuntimeError("Downloaded intraday bars are empty after RTH filtering")

    start_date = pd.to_datetime(bars["date"]).min()
    end_date = pd.to_datetime(bars["date"]).max()
    vix_daily = _download_daily_vix(start_date, end_date)
    merged = _merge_intraday_with_vix(bars, vix_daily)
    merged["symbol_used"] = result.symbol_used

    # Primary dataset note (if 1m coverage is limited).
    days_1m = int(result.interval_days.get(1, 0))
    days_5m = int(result.interval_days.get(5, 0))
    if days_1m < 30 and days_5m > 0:
        report.line(f"1m coverage is limited ({days_1m} days). 5m bars are the primary history source ({days_5m} days).")
    else:
        report.line(f"Using mixed intraday dataset: 5m={days_5m} days, 1m={days_1m} days (1m overrides overlap).")

    _save_cache(merged, CACHE_PATH)
    report.line(f"Saved cache: {CACHE_PATH}")

    meta = _validate_intraday(merged, report)
    return merged, {"source": "download", "symbol_used": result.symbol_used, **meta}


def _enrich_day_running_metrics(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["timestamp_et"] = pd.to_datetime(d["timestamp_et"])
    d["bar_end_et"] = pd.to_datetime(d["bar_end_et"])
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d = d.sort_values(["date", "timestamp_et"]).reset_index(drop=True)

    parts: List[pd.DataFrame] = []
    for date, g in d.groupby("date", sort=True):
        day = g.sort_values("timestamp_et").copy()
        open_px = float(day.iloc[0]["open"])
        day["day_open"] = open_px
        day["running_high"] = pd.to_numeric(day["high"], errors="coerce").cummax()
        day["running_low"] = pd.to_numeric(day["low"], errors="coerce").cummin()
        day["cum_range_pct"] = (day["running_high"] - day["running_low"]) / open_px * 100.0
        final_range = float(day["cum_range_pct"].iloc[-1]) if not day.empty else np.nan
        day["final_range_pct"] = final_range
        day["range_fraction"] = np.where(final_range > 0, day["cum_range_pct"] / final_range, np.nan)
        parts.append(day)
    return pd.concat(parts, ignore_index=True) if parts else d.iloc[0:0].copy()


def _checkpoint_minutes(checkpoints: Sequence[str]) -> Dict[str, int]:
    return {cp: _time_to_minutes(cp) for cp in checkpoints}


def _block_minutes() -> List[Tuple[str, str, str, str, int, int]]:
    out = []
    for start_s, end_s, block_label, block_key in BLOCKS:
        out.append((start_s, end_s, block_label, block_key, _time_to_minutes(start_s), _time_to_minutes(end_s)))
    return out


def _last_value_by_bar_end(day: pd.DataFrame, target_minute: int, column: str) -> float:
    # bar_end_et is ET-naive datetime marking the end of the bar.
    if day.empty:
        return float("nan")
    bar_end_min = day["bar_end_et"].dt.hour * 60 + day["bar_end_et"].dt.minute
    eligible = day.loc[bar_end_min <= target_minute, column]
    if eligible.empty:
        return float("nan")
    return float(pd.to_numeric(eligible.iloc[-1], errors="coerce"))


def _build_checkpoint_frame(df: pd.DataFrame, report: Reporter) -> pd.DataFrame:
    report.section("Build Intraday Checkpoint Panel")
    enriched = _enrich_day_running_metrics(df)
    cp_minutes = _checkpoint_minutes(CHECKPOINTS)

    rows: List[Dict[str, Any]] = []
    for date, g in enriched.groupby("date", sort=True):
        day = g.sort_values("timestamp_et").copy()
        final_range = float(day["final_range_pct"].iloc[-1]) if not day.empty else np.nan
        day_open = float(day["day_open"].iloc[0]) if not day.empty else np.nan
        vix_prev = float(pd.to_numeric(day["vix_prev_close"], errors="coerce").iloc[0]) if "vix_prev_close" in day.columns else np.nan
        vix_bucket = str(day["vix_bucket"].iloc[0]) if "vix_bucket" in day.columns and pd.notna(day["vix_bucket"].iloc[0]) else None
        interval_min = int(day["interval_min"].iloc[0]) if "interval_min" in day.columns else np.nan

        for cp in CHECKPOINTS:
            cp_min = cp_minutes[cp]
            cum_range = _last_value_by_bar_end(day, cp_min, "cum_range_pct")
            frac = _last_value_by_bar_end(day, cp_min, "range_fraction")
            rows.append(
                {
                    "date": date,
                    "checkpoint": cp,
                    "checkpoint_minute": cp_min,
                    "cum_range_pct": cum_range,
                    "range_fraction": frac,
                    "final_range_pct": final_range,
                    "day_open": day_open,
                    "interval_min": interval_min,
                    "vix_prev_close": vix_prev,
                    "vix_bucket": vix_bucket,
                }
            )

    panel = pd.DataFrame(rows)
    if panel.empty:
        raise ValueError("Checkpoint panel is empty")
    panel["vix_bucket"] = pd.Categorical(panel["vix_bucket"], categories=VIX_BUCKET_LABELS, ordered=True)
    report.line(f"Checkpoint panel rows: {len(panel)} ({panel['date'].nunique()} days x {panel['checkpoint'].nunique()} checkpoints)")
    return panel


def _analysis_a1(checkpoints_df: pd.DataFrame, report: Reporter) -> pd.DataFrame:
    report.section("Analysis 1: Cumulative Range by Hour")
    grouped = checkpoints_df.groupby(["checkpoint", "checkpoint_minute"], observed=True)["range_fraction"]
    out = grouped.agg(
        n_days="count",
        mean="mean",
        median="median",
        p25=lambda s: s.quantile(0.25),
        p75=lambda s: s.quantile(0.75),
    ).reset_index()
    out = out.sort_values("checkpoint_minute").reset_index(drop=True)
    _write_csv(out, OUTPUT_DIR / "A1_cumulative_range_fraction.csv", index=False)
    report.table(out, index=False, float_fmt="{:.4f}")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(out))
    med = out["median"].to_numpy(dtype=float)
    p25 = out["p25"].to_numpy(dtype=float)
    p75 = out["p75"].to_numpy(dtype=float)
    ax.plot(x, med, marker="o", color="#4C78A8", linewidth=2, label="Median")
    ax.fill_between(x, p25, p75, color="#4C78A8", alpha=0.20, label="P25-P75")
    ax.axhline(0.50, color="#F28E2B", linestyle="--", linewidth=1, label="50%")
    ax.axhline(0.80, color="#E15759", linestyle="--", linewidth=1, label="80%")
    ax.set_xticks(x)
    ax.set_xticklabels(out["checkpoint"].tolist(), rotation=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of final daily range consumed")
    ax.set_xlabel("Time (ET)")
    ax.set_title("A1. Cumulative Daily Range Consumption by Time")
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart_A1_range_consumption.png", dpi=180)
    plt.close(fig)
    return out


def _analysis_a2(checkpoints_df: pd.DataFrame, report: Reporter) -> pd.DataFrame:
    report.section("Analysis 2: Cumulative Range by Hour x VIX Bucket")
    sub = checkpoints_df.dropna(subset=["vix_bucket"]).copy()
    out = (
        sub.groupby(["vix_bucket", "checkpoint", "checkpoint_minute"], observed=True)["range_fraction"]
        .agg(
            n_days="count",
            mean="mean",
            median="median",
            p25=lambda s: s.quantile(0.25),
            p75=lambda s: s.quantile(0.75),
        )
        .reset_index()
        .sort_values(["vix_bucket", "checkpoint_minute"])
        .reset_index(drop=True)
    )
    _write_csv(out, OUTPUT_DIR / "A2_range_by_hour_vix.csv", index=False)
    report.table(out.head(20), index=False, float_fmt="{:.4f}")

    fig, ax = plt.subplots(figsize=(12, 6))
    for bucket in VIX_BUCKET_LABELS:
        g = out[out["vix_bucket"] == bucket].sort_values("checkpoint_minute")
        if g.empty:
            continue
        x = np.arange(len(g))
        ax.plot(x, g["median"].to_numpy(dtype=float), marker="o", linewidth=2, label=bucket, color=BUCKET_COLORS.get(bucket))
    if not out.empty:
        ticks_src = out[out["vix_bucket"] == out["vix_bucket"].dropna().unique()[0]] if out["vix_bucket"].notna().any() else pd.DataFrame()
        if not ticks_src.empty:
            ax.set_xticks(np.arange(len(ticks_src)))
            ax.set_xticklabels(ticks_src["checkpoint"].tolist())
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Median fraction of final daily range consumed")
    ax.set_xlabel("Time (ET)")
    ax.set_title("A2. Range Consumption by Time and VIX Bucket")
    ax.legend(title="VIX bucket")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart_A2_range_by_vix.png", dpi=180)
    plt.close(fig)
    return out


def _analysis_a3(checkpoints_df: pd.DataFrame, report: Reporter) -> pd.DataFrame:
    report.section("Analysis 3: Incremental Range per Hour Block")
    cp_map = checkpoints_df.set_index(["date", "checkpoint"])["cum_range_pct"].to_dict()
    frac_map = checkpoints_df.set_index(["date", "checkpoint"])["range_fraction"].to_dict()
    dates = sorted(checkpoints_df["date"].dropna().unique())
    rows: List[Dict[str, Any]] = []

    for date in dates:
        final_range = checkpoints_df.loc[checkpoints_df["date"] == date, "final_range_pct"].dropna()
        final_val = float(final_range.iloc[0]) if not final_range.empty else np.nan
        for start_s, end_s, block_label, block_key, _, _ in _block_minutes():
            if start_s == "09:30":
                start_cum = 0.0
                start_frac = 0.0
            else:
                start_cum = cp_map.get((date, start_s), np.nan)
                start_frac = frac_map.get((date, start_s), np.nan)
            end_cum = cp_map.get((date, end_s), np.nan)
            end_frac = frac_map.get((date, end_s), np.nan)
            inc_cum = end_cum - start_cum if np.isfinite(end_cum) and np.isfinite(start_cum) else np.nan
            inc_frac = end_frac - start_frac if np.isfinite(end_frac) and np.isfinite(start_frac) else np.nan
            rows.append(
                {
                    "date": date,
                    "block": block_label,
                    "block_key": block_key,
                    "incremental_range_pct": inc_cum,
                    "incremental_fraction": inc_frac,
                    "final_range_pct": final_val,
                }
            )

    block_df = pd.DataFrame(rows)
    out = (
        block_df.groupby(["block", "block_key"], observed=True)["incremental_fraction"]
        .agg(
            n_days="count",
            mean="mean",
            median="median",
            p25=lambda s: s.quantile(0.25),
            p75=lambda s: s.quantile(0.75),
        )
        .reset_index()
    )
    block_order = [b[2] for b in BLOCKS]
    out["block"] = pd.Categorical(out["block"], categories=block_order, ordered=True)
    out = out.sort_values("block").reset_index(drop=True)
    _write_csv(out, OUTPUT_DIR / "A3_incremental_range.csv", index=False)
    report.table(out, index=False, float_fmt="{:.4f}")

    fig, ax = plt.subplots(figsize=(12, 6))
    cum_bottom = 0.0
    colors = sns.color_palette("viridis", len(out)) if sns is not None else None
    for i, row in out.iterrows():
        val = float(row["mean"]) if pd.notna(row["mean"]) else 0.0
        ax.bar([0], [val], bottom=[cum_bottom], width=0.6, label=str(row["block"]), color=(colors[i] if colors is not None else None))
        cum_bottom += val
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0])
    ax.set_xticklabels(["Mean composition of final daily range"])
    ax.set_ylabel("Fraction of final daily range")
    ax.set_title("A3. Where the Daily Range Is Added (Incremental Fraction by Block)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart_A3_incremental_bars.png", dpi=180)
    plt.close(fig)
    return out


def _price_at_entry(day: pd.DataFrame, entry_minute: int) -> Tuple[Optional[pd.Series], str]:
    ts_min = day["timestamp_et"].dt.hour * 60 + day["timestamp_et"].dt.minute
    exact = day.loc[ts_min == entry_minute]
    if not exact.empty:
        return exact.iloc[0], "exact"
    later = day.loc[ts_min > entry_minute]
    if not later.empty:
        return later.iloc[0], "next_bar"
    return None, "missing"


def _analysis_a4(df: pd.DataFrame, report: Reporter) -> pd.DataFrame:
    report.section('Analysis 4: "Residual Risk" After Entry Time')
    d = _enrich_day_running_metrics(df)
    rows: List[Dict[str, Any]] = []
    entry_minutes = {et: _time_to_minutes(et) for et in ENTRY_TIMES}

    for date, g in d.groupby("date", sort=True):
        day = g.sort_values("timestamp_et").copy()
        if day.empty:
            continue
        vix_prev = float(pd.to_numeric(day["vix_prev_close"], errors="coerce").iloc[0]) if "vix_prev_close" in day.columns else np.nan
        vix_bucket = str(day["vix_bucket"].iloc[0]) if "vix_bucket" in day.columns and pd.notna(day["vix_bucket"].iloc[0]) else None
        day_open = float(day["day_open"].iloc[0])
        final_range = float(day["final_range_pct"].iloc[-1]) if not day.empty else np.nan
        for et in ENTRY_TIMES:
            row_entry, entry_source = _price_at_entry(day, entry_minutes[et])
            if row_entry is None:
                rows.append(
                    {
                        "date": date,
                        "entry_time": et,
                        "entry_source": entry_source,
                        "entry_price": np.nan,
                        "residual_up_pct": np.nan,
                        "residual_down_pct": np.nan,
                        "residual_max_pct": np.nan,
                        "vix_prev_close": vix_prev,
                        "vix_bucket": vix_bucket,
                        "day_open": day_open,
                        "final_range_pct": final_range,
                    }
                )
                continue
            entry_ts = pd.to_datetime(row_entry["timestamp_et"])
            entry_price = float(row_entry["open"])
            post = day[day["timestamp_et"] >= entry_ts].copy()
            if post.empty or not np.isfinite(entry_price) or entry_price <= 0:
                ru = rd = rm = np.nan
            else:
                post_high = float(pd.to_numeric(post["high"], errors="coerce").max())
                post_low = float(pd.to_numeric(post["low"], errors="coerce").min())
                ru = (post_high - entry_price) / entry_price * 100.0
                rd = (entry_price - post_low) / entry_price * 100.0
                rm = max(ru, rd)
            rows.append(
                {
                    "date": date,
                    "entry_time": et,
                    "entry_source": entry_source,
                    "entry_price": entry_price,
                    "residual_up_pct": ru,
                    "residual_down_pct": rd,
                    "residual_max_pct": rm,
                    "vix_prev_close": vix_prev,
                    "vix_bucket": vix_bucket,
                    "day_open": day_open,
                    "final_range_pct": final_range,
                }
            )

    residual = pd.DataFrame(rows)
    residual["vix_bucket"] = pd.Categorical(residual["vix_bucket"], categories=VIX_BUCKET_LABELS, ordered=True)
    residual = residual.sort_values(["date", "entry_time"]).reset_index(drop=True)

    # Aggregate by entry_time x vix_bucket and all-VIX summary rows
    agg_rows: List[Dict[str, Any]] = []
    combos: List[Tuple[str, Optional[str]]] = []
    for et in ENTRY_TIMES:
        combos.append((et, None))  # all VIX
        for vb in VIX_BUCKET_LABELS:
            combos.append((et, vb))

    for et, vb in combos:
        sub = residual[residual["entry_time"] == et].copy()
        if vb is not None:
            sub = sub[sub["vix_bucket"] == vb]
        vals = pd.to_numeric(sub["residual_max_pct"], errors="coerce").dropna()
        row = {
            "entry_time": et,
            "vix_bucket": vb if vb is not None else "ALL",
            "n_days": int(len(vals)),
            "mean_residual_max_pct": float(vals.mean()) if not vals.empty else np.nan,
            "median_residual_max_pct": float(vals.median()) if not vals.empty else np.nan,
            "p90_residual_max_pct": float(vals.quantile(0.90)) if not vals.empty else np.nan,
            "p95_residual_max_pct": float(vals.quantile(0.95)) if not vals.empty else np.nan,
            "p_residual_gt_1p0": float((vals > 1.0).mean()) if not vals.empty else np.nan,
            "p_residual_gt_1p3": float((vals > 1.3).mean()) if not vals.empty else np.nan,
            "entry_source_exact_rate": float((sub["entry_source"] == "exact").mean()) if len(sub) else np.nan,
        }
        agg_rows.append(row)

    a4_out = pd.DataFrame(agg_rows)
    a4_out["entry_time"] = pd.Categorical(a4_out["entry_time"], categories=ENTRY_TIMES, ordered=True)
    _write_csv(a4_out.sort_values(["vix_bucket", "entry_time"]), OUTPUT_DIR / "A4_residual_risk.csv", index=False)
    report.table(a4_out[a4_out["vix_bucket"] == "ALL"].sort_values("entry_time"), index=False, float_fmt="{:.4f}")

    # Heatmap: entry_time x vix_bucket -> P(residual > 1.0%)
    heat = (
        a4_out[a4_out["vix_bucket"] != "ALL"]
        .pivot(index="entry_time", columns="vix_bucket", values="p_residual_gt_1p0")
        .reindex(index=ENTRY_TIMES, columns=VIX_BUCKET_LABELS)
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    if sns is not None:
        sns.heatmap(heat, annot=True, fmt=".1%", cmap="YlOrRd", linewidths=0.5, cbar_kws={"label": "P(residual > 1.0%)"}, ax=ax)
    else:
        data = heat.to_numpy(dtype=float)
        vmax = np.nanmax(data) if np.isfinite(data).any() else 1.0
        im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
        fig.colorbar(im, ax=ax, label="P(residual > 1.0%)")
        ax.set_xticks(np.arange(len(heat.columns))); ax.set_xticklabels(heat.columns.tolist())
        ax.set_yticks(np.arange(len(heat.index))); ax.set_yticklabels(heat.index.tolist())
    ax.set_title("A4. Residual Risk Heatmap: P(residual_max > 1.0%)")
    ax.set_xlabel("VIX bucket (previous close)")
    ax.set_ylabel("Entry time (ET)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart_A4_residual_by_entry.png", dpi=180)
    plt.close(fig)

    # Boxplot all VIX
    fig, ax = plt.subplots(figsize=(12, 6))
    box_df = residual.dropna(subset=["residual_max_pct"]).copy()
    if sns is not None and not box_df.empty:
        sns.boxplot(data=box_df, x="entry_time", y="residual_max_pct", order=ENTRY_TIMES, ax=ax, color="#4C78A8", showfliers=False)
    else:
        arrays = [box_df.loc[box_df["entry_time"] == et, "residual_max_pct"].dropna().to_numpy(dtype=float) for et in ENTRY_TIMES]
        ax.boxplot(arrays, labels=ENTRY_TIMES, showfliers=False)
    ax.set_title("A4. Residual Max Directional Risk by Entry Time (All VIX)")
    ax.set_xlabel("Entry time (ET)")
    ax.set_ylabel("Residual max move after entry (%)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart_A4_residual_boxplot.png", dpi=180)
    plt.close(fig)

    return a4_out


def _analysis_a5(a4_out: pd.DataFrame, report: Reporter) -> pd.DataFrame:
    report.section("Analysis 5: Premium Decay Profile (Theoretical)")
    total_hours = 6.5  # 09:30 -> 16:00
    rows: List[Dict[str, Any]] = []

    a4_all = a4_out[a4_out["vix_bucket"] == "ALL"].copy()
    a4_vix_2025 = a4_out[a4_out["vix_bucket"] == "20-25"].copy()
    for et in ENTRY_TIMES:
        t = _time_str_to_time(et)
        current_hours = t.hour + t.minute / 60.0
        t_remaining = max(0.0, 16.0 - current_hours)
        premium_fraction = math.sqrt(t_remaining / total_hours) if total_hours > 0 and t_remaining > 0 else 0.0
        all_row = a4_all[a4_all["entry_time"] == et]
        vix_row = a4_vix_2025[a4_vix_2025["entry_time"] == et]
        p90_all = float(all_row["p90_residual_max_pct"].iloc[0]) if not all_row.empty else np.nan
        p90_2025 = float(vix_row["p90_residual_max_pct"].iloc[0]) if not vix_row.empty else np.nan
        p_gt1_all = float(all_row["p_residual_gt_1p0"].iloc[0]) if not all_row.empty else np.nan
        p_gt1_2025 = float(vix_row["p_residual_gt_1p0"].iloc[0]) if not vix_row.empty else np.nan
        rows.append(
            {
                "entry_time": et,
                "hours_to_expiry": t_remaining,
                "premium_fraction_theoretical": premium_fraction,
                "premium_lost_by_waiting": 1.0 - premium_fraction,
                "p90_residual_all_vix": p90_all,
                "p90_residual_vix_20_25": p90_2025,
                "p_residual_gt_1p0_all_vix": p_gt1_all,
                "p_residual_gt_1p0_vix_20_25": p_gt1_2025,
                "risk_reward_ratio_p90_all_vix": _safe_div(p90_all, premium_fraction),
                "risk_reward_ratio_p90_vix_20_25": _safe_div(p90_2025, premium_fraction),
            }
        )
    out = pd.DataFrame(rows)
    out["entry_time"] = pd.Categorical(out["entry_time"], categories=ENTRY_TIMES, ordered=True)
    out = out.sort_values("entry_time").reset_index(drop=True)
    _write_csv(out, OUTPUT_DIR / "A5_premium_decay.csv", index=False)
    report.table(out, index=False, float_fmt="{:.4f}")

    # Sweet spot = minimum risk/reward ratio (all-VIX), tie-break earlier time.
    sweet_idx = out["risk_reward_ratio_p90_all_vix"].astype(float).replace([np.inf, -np.inf], np.nan).idxmin()
    sweet_time = str(out.loc[sweet_idx, "entry_time"]) if pd.notna(sweet_idx) else "NA"
    report.line(f"Sweet spot (min P90 residual / premium fraction, all-VIX): {sweet_time}")

    fig, ax1 = plt.subplots(figsize=(12, 6))
    x = np.arange(len(out))
    ax1.plot(x, out["p90_residual_all_vix"].to_numpy(dtype=float), marker="o", linewidth=2, color="#4C78A8", label="P90 residual risk (all VIX)")
    ax1.plot(x, out["p90_residual_vix_20_25"].to_numpy(dtype=float), marker="s", linewidth=2, color="#E15759", label="P90 residual risk (VIX 20-25)")
    ax1.set_ylabel("Residual max move P90 (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(out["entry_time"].astype(str).tolist())
    ax1.set_xlabel("Entry time (ET)")
    ax1.set_title("A5. Risk-Reward Tradeoff: Residual Risk vs Theoretical Premium Remaining")

    ax2 = ax1.twinx()
    ax2.bar(x, out["premium_fraction_theoretical"].to_numpy(dtype=float), alpha=0.25, color="#59A14F", label="Premium fraction remaining")
    ax2.set_ylabel("Premium fraction remaining (theoretical)")
    ax2.set_ylim(0, 1.05)

    if pd.notna(sweet_idx):
        ax1.axvline(float(sweet_idx), color="gray", linestyle="--", linewidth=1)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart_A5_risk_reward_tradeoff.png", dpi=180)
    plt.close(fig)
    return out


def _analysis_a6(df: pd.DataFrame, report: Reporter) -> Tuple[pd.DataFrame, pd.DataFrame]:
    report.section("Analysis 6: Worst Moves - When Do They Start?")
    d = df.copy()
    d["timestamp_et"] = pd.to_datetime(d["timestamp_et"])
    d["bar_end_et"] = pd.to_datetime(d["bar_end_et"])
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d = d.sort_values(["date", "timestamp_et"]).reset_index(drop=True)

    rows: List[Dict[str, Any]] = []
    for date, g in d.groupby("date", sort=True):
        day = g.sort_values("timestamp_et").copy()
        if day.empty:
            continue
        open_px = float(pd.to_numeric(day["open"], errors="coerce").iloc[0])
        if not np.isfinite(open_px) or open_px <= 0:
            continue
        up_level = open_px * 1.01
        down_level = open_px * 0.99

        high_hit = pd.to_numeric(day["high"], errors="coerce") >= up_level
        low_hit = pd.to_numeric(day["low"], errors="coerce") <= down_level
        any_hit = high_hit | low_hit
        if not bool(any_hit.any()):
            continue
        first_idx = int(np.flatnonzero(any_hit.to_numpy(dtype=bool))[0])
        first_row = day.iloc[first_idx]
        side = "up" if bool(high_hit.iloc[first_idx]) and not bool(low_hit.iloc[first_idx]) else "down" if bool(low_hit.iloc[first_idx]) and not bool(high_hit.iloc[first_idx]) else "both_same_bar"
        breach_bar_end = pd.to_datetime(first_row["bar_end_et"])
        hour_dec = _decimal_hour(breach_bar_end)
        if hour_dec < 11.0:
            timing_bucket = "before_11"
        elif hour_dec < 14.0:
            timing_bucket = "11_to_14"
        else:
            timing_bucket = "after_14"
        rows.append(
            {
                "date": date,
                "breach_time_et": breach_bar_end,
                "breach_time_str": breach_bar_end.strftime("%H:%M"),
                "breach_hour_decimal": hour_dec,
                "breach_side": side,
                "timing_bucket": timing_bucket,
                "vix_prev_close": float(pd.to_numeric(day["vix_prev_close"], errors="coerce").iloc[0]) if "vix_prev_close" in day.columns else np.nan,
                "vix_bucket": str(day["vix_bucket"].iloc[0]) if "vix_bucket" in day.columns and pd.notna(day["vix_bucket"].iloc[0]) else None,
            }
        )

    raw = pd.DataFrame(rows)
    if raw.empty:
        out = pd.DataFrame(columns=["vix_bucket", "timing_bucket", "n_breaches", "share_within_vix"])
        _write_csv(out, OUTPUT_DIR / "A6_breach_timing.csv", index=False)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "No breach days (>1.0% from open) in dataset", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "chart_A6_breach_timing.png", dpi=180)
        plt.close(fig)
        return out, raw

    raw["vix_bucket"] = pd.Categorical(raw["vix_bucket"], categories=VIX_BUCKET_LABELS, ordered=True)
    agg = (
        raw.groupby(["vix_bucket", "timing_bucket"], observed=True)
        .size()
        .rename("n_breaches")
        .reset_index()
    )
    totals = raw.groupby("vix_bucket", observed=True).size().rename("n_total_vix").reset_index()
    agg = agg.merge(totals, on="vix_bucket", how="left")
    agg["share_within_vix"] = agg["n_breaches"] / agg["n_total_vix"]
    _write_csv(agg, OUTPUT_DIR / "A6_breach_timing.csv", index=False)
    report.table(agg.sort_values(["vix_bucket", "timing_bucket"]), index=False, float_fmt="{:.4f}")

    fig, ax = plt.subplots(figsize=(12, 6))
    # Histogram with hue by VIX bucket; fallback to stacked bars if seaborn unavailable.
    if sns is not None:
        plot_df = raw.dropna(subset=["vix_bucket"]).copy()
        if not plot_df.empty:
            bins = np.arange(10.0, 16.5, 0.5)
            sns.histplot(data=plot_df, x="breach_hour_decimal", hue="vix_bucket", bins=bins, multiple="stack", palette=BUCKET_COLORS, ax=ax)
        else:
            ax.text(0.5, 0.5, "No VIX bucket data for breach timing", ha="center", va="center")
    else:
        bins = np.arange(10.0, 16.5, 0.5)
        centers = bins[:-1] + 0.25
        width = 0.45
        bottom = np.zeros_like(centers, dtype=float)
        for bucket in VIX_BUCKET_LABELS:
            vals = raw.loc[raw["vix_bucket"] == bucket, "breach_hour_decimal"].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            counts, _ = np.histogram(vals, bins=bins)
            ax.bar(centers, counts, bottom=bottom, width=width, label=bucket, color=BUCKET_COLORS.get(bucket))
            bottom += counts
    ax.set_title("A6. Breach Timing (first 1.0% move from open) by VIX bucket")
    ax.set_xlabel("Breach time (ET, bar-end approximation)")
    ax.set_ylabel("Count of breach days")
    if sns is None:
        ax.legend(title="VIX bucket")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart_A6_breach_timing.png", dpi=180)
    plt.close(fig)
    return agg, raw


def _range_consumption_profile_label(a1: pd.DataFrame) -> str:
    if a1.empty:
        return "UNKNOWN"
    get_med = lambda cp: float(a1.loc[a1["checkpoint"] == cp, "median"].iloc[0]) if (a1["checkpoint"] == cp).any() else np.nan
    by_11 = get_med("11:00")
    by_14 = get_med("14:00")
    if np.isfinite(by_11) and by_11 >= 0.65:
        return "FRONT-LOADED"
    if np.isfinite(by_14) and by_14 <= 0.70:
        return "BACK-LOADED"
    return "UNIFORM"


def _vix_hour_interaction_label(a2: pd.DataFrame) -> str:
    if a2.empty:
        return "DOES NOT EXIST"
    # Compare median range fraction at 11:00 and 12:00 between high and low VIX.
    sub = a2[a2["checkpoint"].isin(["11:00", "12:00"])].copy()
    if sub.empty:
        return "DOES NOT EXIST"
    low = sub[sub["vix_bucket"].isin(["0-15", "15-20"])]
    high = sub[sub["vix_bucket"] == "25+"]
    if low.empty or high.empty:
        return "DOES NOT EXIST"
    low_med = float(low["median"].mean())
    high_med = float(high["median"].mean())
    diff = high_med - low_med
    if diff > 0.07:
        return "EXISTS (high-VIX earlier/faster)"
    if diff < -0.07:
        return "EXISTS (high-VIX later/slower)"
    return "DOES NOT EXIST"


def _write_summary_report(
    report: Reporter,
    meta: Dict[str, Any],
    a1: pd.DataFrame,
    a2: pd.DataFrame,
    a4: pd.DataFrame,
    a5: pd.DataFrame,
    a6_agg: pd.DataFrame,
    a6_raw: pd.DataFrame,
) -> None:
    report.section("CONSOLIDATED SUMMARY")
    report.line("============================================================")
    report.line("INTRADAY RANGE ANALYSIS - WHEN DOES THE MOVE HAPPEN?")
    report.line("============================================================")
    report.line()
    report.line(
        "DATA: Intraday bars (mixed 5m/1m), N={n} trading days ({d0} to {d1}), times in ET".format(
            n=meta.get("n_days", "NA"),
            d0=meta.get("date_start", "NA"),
            d1=meta.get("date_end", "NA"),
        )
    )

    report.line()
    report.line("RANGE CONSUMPTION PROFILE")
    report.line("============================================================")
    for cp in ["10:00", "10:30", "11:00", "12:00", "14:00"]:
        row = a1[a1["checkpoint"] == cp]
        med = float(row["median"].iloc[0]) if not row.empty else np.nan
        report.line(f"By {cp}: {_fmt_pct(med)} of daily range consumed (median)")
    profile = _range_consumption_profile_label(a1)
    report.line(f"-> {profile}")

    # High-VIX timing vs low-VIX timing at 11:00 and 12:00.
    low_vix_rows = a2[(a2["vix_bucket"].isin(["0-15", "15-20"])) & (a2["checkpoint"].isin(["11:00", "12:00"]))]
    high_vix_rows = a2[(a2["vix_bucket"] == "25+") & (a2["checkpoint"].isin(["11:00", "12:00"]))]
    low_mean = float(low_vix_rows["median"].mean()) if not low_vix_rows.empty else np.nan
    high_mean = float(high_vix_rows["median"].mean()) if not high_vix_rows.empty else np.nan
    faster_label = "SIMILAR"
    if np.isfinite(low_mean) and np.isfinite(high_mean):
        if high_mean - low_mean > 0.07:
            faster_label = "FASTER"
        elif low_mean - high_mean > 0.07:
            faster_label = "SLOWER"
    report.line()
    report.line(f"HIGH-VIX DAYS: range consumption is {faster_label} than low-VIX")

    report.line()
    report.line("RESIDUAL RISK BY ENTRY TIME (all VIX)")
    report.line("============================================================")
    a4_all = a4[a4["vix_bucket"] == "ALL"].copy().sort_values("entry_time")
    for et in ENTRY_TIMES:
        row = a4_all[a4_all["entry_time"] == et]
        if row.empty:
            continue
        r = row.iloc[0]
        report.line(
            f"Entry {et}: P(residual>1%) = {_fmt_pct(float(r['p_residual_gt_1p0']))}, "
            f"P(residual>1.3%) = {_fmt_pct(float(r['p_residual_gt_1p3']))}, "
            f"P90 residual = {float(r['p90_residual_max_pct']):.2f}%"
        )

    report.line()
    report.line("RESIDUAL RISK BY ENTRY TIME (VIX 20-25 only)")
    report.line("============================================================")
    a4_2025 = a4[a4["vix_bucket"] == "20-25"].copy().sort_values("entry_time")
    for et in ENTRY_TIMES:
        row = a4_2025[a4_2025["entry_time"] == et]
        if row.empty:
            continue
        r = row.iloc[0]
        report.line(
            f"Entry {et}: P(residual>1%) = {_fmt_pct(float(r['p_residual_gt_1p0']))}, "
            f"P(residual>1.3%) = {_fmt_pct(float(r['p_residual_gt_1p3']))}, "
            f"P90 residual = {float(r['p90_residual_max_pct']):.2f}%"
        )

    report.line()
    report.line("RISK-REWARD SWEET SPOT")
    report.line("============================================================")
    sweet_metric = a5["risk_reward_ratio_p90_all_vix"].replace([np.inf, -np.inf], np.nan)
    if sweet_metric.notna().any():
        idx = int(sweet_metric.idxmin())
        sweet = a5.iloc[idx]
        report.line(f"Optimal entry time (proxy): {sweet['entry_time']} ET")
        report.line(
            f"Rationale: P90 residual risk = {float(sweet['p90_residual_all_vix']):.2f}% while premium remains at {_fmt_pct(float(sweet['premium_fraction_theoretical']))}"
        )
    else:
        report.line("Optimal entry time: NA (insufficient data)")
    report.line("Current bot behavior: enters at first valid fill after 10:00")

    # Recommendation heuristic comparing 10:00 vs 11:00
    r10 = a4_all[a4_all["entry_time"] == "10:00"]
    r11 = a4_all[a4_all["entry_time"] == "11:00"]
    p10 = a5[a5["entry_time"] == "10:00"]
    p11 = a5[a5["entry_time"] == "11:00"]
    if not r10.empty and not r11.empty and not p10.empty and not p11.empty:
        breach10 = float(r10.iloc[0]["p_residual_gt_1p0"])
        breach11 = float(r11.iloc[0]["p_residual_gt_1p0"])
        prem10 = float(p10.iloc[0]["premium_fraction_theoretical"])
        prem11 = float(p11.iloc[0]["premium_fraction_theoretical"])
        risk_drop = _safe_div(breach10 - breach11, breach10)
        premium_cost = _safe_div(prem10 - prem11, prem10)
        if np.isfinite(risk_drop) and np.isfinite(premium_cost) and risk_drop > premium_cost + 0.05:
            reco = "DELAY ENTRY TO 11:00"
        elif np.isfinite(risk_drop) and np.isfinite(premium_cost) and premium_cost > risk_drop + 0.05:
            reco = "KEEP CURRENT"
        else:
            reco = "ADD ENTRY WINDOW FILTER (conditional on VIX)"
        report.line(f"Recommendation: {reco}")
    else:
        report.line("Recommendation: KEEP CURRENT (insufficient comparison data)")

    report.line()
    report.line("BREACH TIMING (days where range > 1%)")
    report.line("============================================================")
    if a6_raw.empty:
        report.line("No breach days found in dataset.")
    else:
        n_breach = len(a6_raw)
        before_11 = float((a6_raw["timing_bucket"] == "before_11").mean())
        between_11_14 = float((a6_raw["timing_bucket"] == "11_to_14").mean())
        after_14 = float((a6_raw["timing_bucket"] == "after_14").mean())
        report.line(f"{_fmt_pct(before_11)} of breaches occur before 11:00")
        report.line(f"{_fmt_pct(between_11_14)} of breaches occur between 11:00-14:00")
        report.line(f"{_fmt_pct(after_14)} of breaches occur after 14:00")
        if before_11 >= 0.55:
            timing_label = "CONCENTRATED EARLY"
        elif after_14 >= 0.45:
            timing_label = "CONCENTRATED LATE"
        else:
            timing_label = "SPREAD THROUGHOUT"
        report.line(f"-> Dangerous moves are {timing_label}")
        report.line(f"Breach-day count: {n_breach}")

    report.line()
    report.line("============================================================")
    report.line("ACTIONABLE CONCLUSIONS")
    report.line("============================================================")

    # [1] Entry window
    if not r10.empty and not r11.empty and not p10.empty and not p11.empty:
        breach10 = float(r10.iloc[0]["p_residual_gt_1p0"])
        breach11 = float(r11.iloc[0]["p_residual_gt_1p0"])
        prem10 = float(p10.iloc[0]["premium_fraction_theoretical"])
        prem11 = float(p11.iloc[0]["premium_fraction_theoretical"])
        risk_reduction = _safe_div(breach10 - breach11, breach10)
        premium_lost = _safe_div(prem10 - prem11, prem10)
        if np.isfinite(risk_reduction) and np.isfinite(premium_lost) and risk_reduction > premium_lost + 0.05:
            entry_reco = "SHIFT TO 11:00-12:00"
        elif np.isfinite(risk_reduction) and np.isfinite(premium_lost) and premium_lost > risk_reduction + 0.05:
            entry_reco = "KEEP 10:00"
        else:
            entry_reco = "KEEP 10:00 but test conditional delay on high-VIX days"
        report.line(f"[1] Entry window: {entry_reco}")
        report.line(
            f"    Evidence: waiting 10:00->11:00 changes P(breach) by {_fmt_pct(breach10 - breach11)} "
            f"(relative {_fmt_pct(risk_reduction)}) while premium cost is ~{_fmt_pct(premium_lost)}"
        )
    else:
        report.line("[1] Entry window: KEEP 10:00")
        report.line("    Evidence: insufficient data")

    # [2] VIX interaction with hour
    interaction = _vix_hour_interaction_label(a2)
    report.line(f"[2] VIX interaction with hour: {interaction}")
    if "earlier" in interaction:
        report.line("    High-VIX days concentrate risk EARLIER than low-VIX")
    elif "later" in interaction:
        report.line("    High-VIX days concentrate risk LATER than low-VIX")
    else:
        report.line("    High-VIX timing profile appears similar to low-VIX timing profile")

    # [3] Wing=2 proxy from residual thresholds
    if not r10.empty and not r11.empty and not p10.empty and not p11.empty:
        breach10 = float(r10.iloc[0]["p_residual_gt_1p0"])
        breach11 = float(r11.iloc[0]["p_residual_gt_1p0"])
        wing10 = float(r10.iloc[0]["p_residual_gt_1p3"])
        wing11 = float(r11.iloc[0]["p_residual_gt_1p3"])
        prem10 = float(p10.iloc[0]["premium_fraction_theoretical"])
        prem11 = float(p11.iloc[0]["premium_fraction_theoretical"])
        premium_lost = _safe_div(prem10 - prem11, prem10)
        effect = "POSITIVE" if (breach10 - breach11) > (premium_lost * 0.5) else "NEUTRAL"
        report.line("[3] For wing=2 specifically (range-threshold proxy):")
        report.line(f"    Entry at 10:00 -> P(short breach ~1.0%) = {_fmt_pct(breach10)} | P(wing breach ~1.3%) = {_fmt_pct(wing10)}")
        report.line(f"    Entry at 11:00 -> P(short breach ~1.0%) = {_fmt_pct(breach11)} | P(wing breach ~1.3%) = {_fmt_pct(wing11)}")
        report.line(f"    Premium lost by waiting: ~{_fmt_pct(premium_lost)}")
        report.line(f"    Net effect (risk proxy vs premium proxy): {effect}")
    else:
        report.line("[3] For wing=2 specifically: insufficient data")

    # [4] FOMC-specific
    report.line("[4] FOMC days specifically: NOT IDENTIFIABLE in current dataset")
    report.line("    Requires external event calendar (e.g., FOMC schedule) to separate pre/post-14:00 behavior on actual FOMC dates.")
    report.line("============================================================")


def _collect_artifacts(report: Reporter) -> None:
    report.line()
    report.line("Artifacts")
    report.line("---------")
    for name in [
        "_intraday_cache.csv",
        "A1_cumulative_range_fraction.csv",
        "chart_A1_range_consumption.png",
        "A2_range_by_hour_vix.csv",
        "chart_A2_range_by_vix.png",
        "A3_incremental_range.csv",
        "chart_A3_incremental_bars.png",
        "A4_residual_risk.csv",
        "chart_A4_residual_by_entry.png",
        "chart_A4_residual_boxplot.png",
        "A5_premium_decay.csv",
        "chart_A5_risk_reward_tradeoff.png",
        "A6_breach_timing.csv",
        "chart_A6_breach_timing.png",
        "hourly_range_report.txt",
    ]:
        report.line(f"- {OUTPUT_DIR / name}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _set_plot_style()
    report = Reporter()

    report.line("Intraday Range Analysis - When Does the Move Happen?")
    report.line(f"Repo root: {REPO_ROOT}")
    report.line(f"Output dir: {OUTPUT_DIR}")

    try:
        df, meta = _load_or_download_intraday(report)
    except Exception as exc:
        report.section("ERROR")
        report.line(f"Failed to acquire intraday data: {exc.__class__.__name__}: {exc}")
        report.line("Attempted SPX (^GSPC) and XSP (^XSP) intraday downloads via yfinance.")
        report_path = OUTPUT_DIR / "hourly_range_report.txt"
        report.save(report_path)
        print(f"\nSaved error report: {report_path}")
        return

    checkpoints_df = _build_checkpoint_frame(df, report)
    a1 = _analysis_a1(checkpoints_df, report)
    a2 = _analysis_a2(checkpoints_df, report)
    _ = _analysis_a3(checkpoints_df, report)
    a4 = _analysis_a4(df, report)
    a5 = _analysis_a5(a4, report)
    a6_agg, a6_raw = _analysis_a6(df, report)

    _write_summary_report(report, meta=meta, a1=a1, a2=a2, a4=a4, a5=a5, a6_agg=a6_agg, a6_raw=a6_raw)
    _collect_artifacts(report)

    report_path = OUTPUT_DIR / "hourly_range_report.txt"
    report.save(report_path)
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
