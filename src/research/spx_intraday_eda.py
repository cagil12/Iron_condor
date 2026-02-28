"""SPX intraday return EDA for XSP 0DTE Iron Condor risk framing.

Run:
    python -m src.research.spx_intraday_eda
    python src/research/spx_intraday_eda.py

This is a research/EDA script (not a backtester). It uses SPX daily + intraday
bars and VIX daily bars from Yahoo Finance, then exports CSV outputs and a
plain-text report under outputs/spx_intraday_eda/.
"""

from __future__ import annotations

import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import stats

try:
    import yfinance as yf
except Exception as exc:  # pragma: no cover
    raise RuntimeError("yfinance is required (pip install yfinance)") from exc

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "outputs" / "spx_intraday_eda"
CACHE_DIR = OUTPUT_DIR / "_cache"
REPORT_PATH = OUTPUT_DIR / "spx_intraday_eda_report.txt"

ET_TZ = ZoneInfo("America/New_York")

START_DATE = "2020-01-01"
END_DATE = "2026-02-27"

# Current reference only for illustrative point conversion in printed examples.
XSP_REF_EXAMPLE = 690.0
IC_CREDIT = 0.25
WING_WIDTH = 2.0
XSP_MULTIPLIER = 100.0

DAILY_SPX_TICKER = "^GSPC"
VIX_TICKER = "^VIX"


class Reporter:
    def __init__(self) -> None:
        self.lines: List[str] = []

    def line(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def section(self, title: str) -> None:
        self.line()
        self.line("=" * 70)
        self.line(title)
        self.line("=" * 70)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def _ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _flatten_ohlc_columns(frame: pd.DataFrame, symbol_hint: Optional[str] = None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        if symbol_hint:
            lvl0 = list(out.columns.get_level_values(0))
            lvl1 = list(out.columns.get_level_values(1))
            if symbol_hint in lvl0:
                out = out[symbol_hint]
            elif symbol_hint in lvl1:
                out = out.xs(symbol_hint, axis=1, level=1)
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = out.columns.get_level_values(-1)
    out = out.rename(columns={c: str(c).title() for c in out.columns})
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in out.columns]
    return out[keep].copy()


def _download_daily(symbol: str, start: str, end: str, cache_name: str) -> pd.DataFrame:
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists():
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if not cached.empty:
            return cached
    raw = yf.download(
        symbol,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    df = _flatten_ohlc_columns(raw, symbol_hint=symbol)
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.to_csv(cache_path, index=True)
    return df


@dataclass
class IntradayDownload:
    bars: pd.DataFrame
    source_desc: str
    notes: List[str]


def _download_intraday_1h_rth(report: Reporter) -> IntradayDownload:
    """Download 1h SPX intraday bars, convert to ET, and filter to RTH.

    Yahoo's retention on intraday data can vary. We attempt 730d and fall back
    to 60d. The output remains useful for hour-of-day pattern EDA even when
    history is shorter than requested.
    """
    notes: List[str] = []
    bars = pd.DataFrame()
    used_period = ""
    for period in ("730d", "365d", "180d", "60d"):
        try:
            raw = yf.download(
                DAILY_SPX_TICKER,
                period=period,
                interval="1h",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            df = _flatten_ohlc_columns(raw, symbol_hint=DAILY_SPX_TICKER)
            if df.empty:
                notes.append(f"period={period}: empty")
                continue
            idx = pd.to_datetime(df.index)
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            idx_et = idx.tz_convert(ET_TZ).tz_localize(None)
            df = df.copy()
            df["timestamp_et"] = idx_et
            df = df.reset_index(drop=True)
            # Regular trading hours by bar-start time. 1h Yahoo bars typically start
            # at 09:30, 10:30, ... 15:30 ET.
            mins = df["timestamp_et"].dt.hour * 60 + df["timestamp_et"].dt.minute
            rth_mask = (mins >= 570) & (mins < 960)
            df = df.loc[rth_mask].copy()
            df = df[df["timestamp_et"].dt.dayofweek < 5].copy()
            if df.empty:
                notes.append(f"period={period}: no RTH bars after filtering")
                continue
            df = df.sort_values("timestamp_et").drop_duplicates("timestamp_et", keep="last").reset_index(drop=True)
            df["date"] = df["timestamp_et"].dt.date
            df["hour"] = df["timestamp_et"].dt.hour
            df["minute"] = df["timestamp_et"].dt.minute
            df["bar_end_et"] = df["timestamp_et"] + pd.Timedelta(hours=1)
            bars = df
            used_period = period
            notes.append(f"period={period}: {len(df)} RTH hourly bars accepted")
            break
        except Exception as exc:  # pragma: no cover
            notes.append(f"period={period}: error={exc}")

    cache_path = CACHE_DIR / "spx_hourly_1h_rth.csv"
    if not bars.empty:
        bars.to_csv(cache_path, index=False)
    elif cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["timestamp_et", "bar_end_et"])
        if not cached.empty:
            notes.append("using cached spx_hourly_1h_rth.csv")
            bars = cached
            used_period = "cache"
    return IntradayDownload(bars=bars, source_desc=f"1h intraday ({used_period or 'none'})", notes=notes)


def _safe_jb(series: pd.Series) -> Tuple[float, float]:
    x = series.dropna()
    if len(x) < 8:
        return float("nan"), float("nan")
    jb = stats.jarque_bera(x)
    return float(jb.statistic), float(jb.pvalue)


def _fmt_pct(x: float, ndigits: int = 1) -> str:
    if not np.isfinite(x):
        return "NA"
    return f"{x:.{ndigits}f}%"


def _xsp_points_from_pct(pct_series: pd.Series, xsp_ref_series: pd.Series) -> pd.Series:
    return pct_series.astype(float) / 100.0 * xsp_ref_series.astype(float)


def _save_csv(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    path = OUTPUT_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path


def _hour_bar(count_pct: float, scale: float = 2.0) -> str:
    blocks = int(max(count_pct, 0.0) / scale)
    return "#" * blocks


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def main() -> int:
    _ensure_dirs()
    report = Reporter()

    report.line("=" * 70)
    report.line("SPX INTRADAY RETURN EDA FOR IRON CONDOR 0DTE (PRODUCTIONIZED)")
    report.line("=" * 70)

    report.line()
    report.line("Assumptions / framing:")
    report.line(f"  Example XSP reference (display only): {XSP_REF_EXAMPLE:.0f}")
    report.line(f"  Typical IC credit (for SL survival proxy): ${IC_CREDIT:.2f}")
    report.line(f"  Wing width cap for spread approximation: {WING_WIDTH:.1f} pts")
    report.line("  Intraday timing layer uses 1h Yahoo bars -> coarse timing proxy (not minute-accurate).")

    # Downloads
    report.section("DATA DOWNLOAD")
    report.line(f"Downloading {DAILY_SPX_TICKER} daily OHLC ({START_DATE} to {END_DATE})...")
    spx_daily = _download_daily(DAILY_SPX_TICKER, START_DATE, END_DATE, "spx_daily_1d.csv")
    report.line(f"  Daily SPX bars: {len(spx_daily)}")

    report.line(f"Downloading {VIX_TICKER} daily OHLC ({START_DATE} to {END_DATE})...")
    vix_daily = _download_daily(VIX_TICKER, START_DATE, END_DATE, "vix_daily_1d.csv")
    report.line(f"  Daily VIX bars: {len(vix_daily)}")

    intraday = _download_intraday_1h_rth(report)
    report.line(f"Hourly source: {intraday.source_desc}")
    for note in intraday.notes:
        report.line(f"  - {note}")
    spx_hourly = intraday.bars
    report.line(f"  Hourly RTH bars retained: {len(spx_hourly)}")

    if len(spx_daily) <= 100:
        raise RuntimeError(f"Insufficient daily SPX data: {len(spx_daily)}")
    if len(spx_hourly) <= 50:
        raise RuntimeError(f"Insufficient hourly SPX data after RTH filter: {len(spx_hourly)}")

    # Layer 1: Daily MAE
    report.section("LAYER 1: DAILY MAX ADVERSE EXCURSION (2020-2026)")
    df = _coerce_numeric(spx_daily.copy(), ["Open", "High", "Low", "Close"])
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    df["mae_down"] = (df["Open"] - df["Low"]) / df["Open"] * 100.0
    df["mae_up"] = (df["High"] - df["Open"]) / df["Open"] * 100.0
    df["range_pct"] = (df["High"] - df["Low"]) / df["Open"] * 100.0
    df["close_ret"] = (df["Close"] - df["Open"]) / df["Open"] * 100.0
    df["xsp_open_ref"] = df["Open"] / 10.0

    report.line("MAE Down (Open→Low, proxy for PUT side risk):")
    report.line(f"  Mean   {_fmt_pct(df['mae_down'].mean(), 3)}")
    report.line(f"  Median {_fmt_pct(df['mae_down'].median(), 3)}")
    report.line(f"  Std    {_fmt_pct(df['mae_down'].std(), 3)}")
    report.line(f"  P90    {_fmt_pct(df['mae_down'].quantile(0.90), 3)}")
    report.line(f"  P95    {_fmt_pct(df['mae_down'].quantile(0.95), 3)}")
    report.line(f"  P99    {_fmt_pct(df['mae_down'].quantile(0.99), 3)}")
    report.line(f"  Max    {_fmt_pct(df['mae_down'].max(), 3)}")

    report.line()
    report.line("MAE Up (Open→High, proxy for CALL side risk):")
    report.line(f"  Mean   {_fmt_pct(df['mae_up'].mean(), 3)}")
    report.line(f"  Median {_fmt_pct(df['mae_up'].median(), 3)}")
    report.line(f"  Std    {_fmt_pct(df['mae_up'].std(), 3)}")
    report.line(f"  P90    {_fmt_pct(df['mae_up'].quantile(0.90), 3)}")
    report.line(f"  P95    {_fmt_pct(df['mae_up'].quantile(0.95), 3)}")
    report.line(f"  P99    {_fmt_pct(df['mae_up'].quantile(0.99), 3)}")
    report.line(f"  Max    {_fmt_pct(df['mae_up'].max(), 3)}")

    mae_down_pts = _xsp_points_from_pct(df["mae_down"], df["xsp_open_ref"])
    mae_up_pts = _xsp_points_from_pct(df["mae_up"], df["xsp_open_ref"])
    report.line()
    report.line("Converted to XSP-equivalent points (dynamic daily reference = SPX Open / 10):")
    report.line(
        "  MAE Down (put): "
        f"Mean={mae_down_pts.mean():.1f} | Median={mae_down_pts.median():.1f} | "
        f"P90={mae_down_pts.quantile(0.90):.1f} | P95={mae_down_pts.quantile(0.95):.1f} | "
        f"P99={mae_down_pts.quantile(0.99):.1f} | Max={mae_down_pts.max():.1f} pts"
    )
    report.line(
        "  MAE Up (call): "
        f"Mean={mae_up_pts.mean():.1f} | Median={mae_up_pts.median():.1f} | "
        f"P90={mae_up_pts.quantile(0.90):.1f} | P95={mae_up_pts.quantile(0.95):.1f} | "
        f"P99={mae_up_pts.quantile(0.99):.1f} | Max={mae_up_pts.max():.1f} pts"
    )

    report.line()
    report.line("How often does intraday MAE reach a strike distance? (dynamic point threshold per day)")
    report.line(" Dist | Put breach | Call breach | Either")
    report.line("----- | ---------- | ----------- | ------")
    breach_rows: List[Dict[str, float]] = []
    for dist_pts in [5, 6, 7, 8, 9, 10, 12, 15, 20]:
        dist_pct_series = (dist_pts / df["xsp_open_ref"]) * 100.0
        breach_down = (df["mae_down"] >= dist_pct_series).mean() * 100.0
        breach_up = (df["mae_up"] >= dist_pct_series).mean() * 100.0
        breach_either = ((df["mae_down"] >= dist_pct_series) | (df["mae_up"] >= dist_pct_series)).mean() * 100.0
        report.line(f"{dist_pts:4d} | {breach_down:9.1f}% | {breach_up:10.1f}% | {breach_either:6.1f}%")
        breach_rows.append(
            {
                "distance_pts": dist_pts,
                "put_breach_pct": breach_down,
                "call_breach_pct": breach_up,
                "either_breach_pct": breach_either,
            }
        )

    report.line()
    report.line("Distribution shape (fat tails diagnostics):")
    jb_stat, jb_p = _safe_jb(df["mae_down"])
    close_rets = df["Close"].pct_change().dropna() * 100.0
    jb_close_stat, jb_close_p = _safe_jb(close_rets)
    report.line("  MAE Down:")
    report.line(f"    Skewness   {df['mae_down'].skew():.2f}")
    report.line(f"    Kurtosis   {df['mae_down'].kurtosis():.2f} (excess)")
    report.line(f"    Jarque-Bera stat={jb_stat:.1f}, p={jb_p:.2e}")
    report.line("  Daily close-to-close returns:")
    report.line(f"    Skewness   {close_rets.skew():.2f}")
    report.line(f"    Kurtosis   {close_rets.kurtosis():.2f} (excess)")
    report.line(f"    Jarque-Bera stat={jb_close_stat:.1f}, p={jb_close_p:.2e}")

    # VIX regime segmentation using previous close (avoids same-day look-ahead complaints).
    report.line()
    report.line("MAE by VIX regime (using VIX previous close for regime assignment):")
    vix_prev_close = vix_daily["Close"].shift(1) if "Close" in vix_daily.columns else pd.Series(index=vix_daily.index, dtype=float)
    df["vix_prev_close"] = vix_prev_close.reindex(df.index).ffill()
    df["vix_open"] = vix_daily["Open"].reindex(df.index).ffill() if "Open" in vix_daily.columns else np.nan

    vix_bins = [(0, 15), (15, 20), (20, 25), (25, 30), (30, 100)]
    regime_rows: List[Dict[str, float]] = []
    for lo, hi in vix_bins:
        mask = (df["vix_prev_close"] >= lo) & (df["vix_prev_close"] < hi)
        sub = df.loc[mask].copy()
        if len(sub) < 5:
            continue
        pts_down = _xsp_points_from_pct(sub["mae_down"], sub["xsp_open_ref"])
        pts_up = _xsp_points_from_pct(sub["mae_up"], sub["xsp_open_ref"])
        dist_8_pct = (8.0 / sub["xsp_open_ref"]) * 100.0
        breach_8 = ((sub["mae_down"] >= dist_8_pct) | (sub["mae_up"] >= dist_8_pct)).mean() * 100.0
        label = f"VIX {lo:>2d}-{hi:>2d}"
        report.line(f"  {label} (n={len(sub):4d}):")
        report.line(
            "    MAE down pts: "
            f"mean={pts_down.mean():.1f} P90={pts_down.quantile(0.90):.1f} "
            f"P95={pts_down.quantile(0.95):.1f} P99={pts_down.quantile(0.99):.1f}"
        )
        report.line(
            "    MAE up pts:   "
            f"mean={pts_up.mean():.1f} P90={pts_up.quantile(0.90):.1f} "
            f"P95={pts_up.quantile(0.95):.1f} P99={pts_up.quantile(0.99):.1f}"
        )
        report.line(f"    Breach 8 pts (either side): {breach_8:.1f}%")
        regime_rows.append(
            {
                "regime": label,
                "vix_lo": lo,
                "vix_hi": hi,
                "n": len(sub),
                "mae_down_mean_pts": pts_down.mean(),
                "mae_down_p90_pts": pts_down.quantile(0.90),
                "mae_down_p95_pts": pts_down.quantile(0.95),
                "mae_down_p99_pts": pts_down.quantile(0.99),
                "mae_up_mean_pts": pts_up.mean(),
                "mae_up_p90_pts": pts_up.quantile(0.90),
                "mae_up_p95_pts": pts_up.quantile(0.95),
                "mae_up_p99_pts": pts_up.quantile(0.99),
                "breach_8pts_pct": breach_8,
            }
        )

    regime_df = pd.DataFrame(regime_rows)

    # Layer 2: Hourly intraday dynamics (1h proxy)
    report.section("LAYER 2: HOURLY INTRADAY DYNAMICS (1H BARS, ET/RTH-FILTERED)")
    report.line("Note: 1h Yahoo bars are coarse. '10:00 onward' is proxied using the first bar ending after ~10:00.")

    h = _coerce_numeric(spx_hourly.copy(), ["Open", "High", "Low", "Close"])
    h = h.dropna(subset=["Open", "High", "Low", "Close", "timestamp_et"]).copy()
    h["date"] = pd.to_datetime(h["timestamp_et"]).dt.date
    h["hour"] = pd.to_datetime(h["timestamp_et"]).dt.hour
    h["minute"] = pd.to_datetime(h["timestamp_et"]).dt.minute
    sessions = h.groupby("date", sort=True)

    session_data: List[Dict[str, float]] = []
    gap_rows: List[Dict[str, float]] = []
    for date, group in sessions:
        if len(group) < 5:
            continue
        group = group.sort_values("timestamp_et").copy()
        open_price = float(group["Open"].iloc[0])
        close_price = float(group["Close"].iloc[-1])
        running_low = group["Low"].cummin()
        running_high = group["High"].cummax()
        day_mae_down = float((open_price - float(running_low.min())) / open_price * 100.0)
        day_mae_up = float((float(running_high.max()) - open_price) / open_price * 100.0)
        worst_down_idx = (open_price - running_low).idxmax()
        worst_up_idx = (running_high - open_price).idxmax()
        worst_down_hour = int(pd.Timestamp(group.loc[worst_down_idx, "timestamp_et"]).hour)
        worst_up_hour = int(pd.Timestamp(group.loc[worst_up_idx, "timestamp_et"]).hour)
        close_ret = float((close_price - open_price) / open_price * 100.0)
        session_data.append(
            {
                "date": str(date),
                "open": open_price,
                "close": close_price,
                "mae_down_pct": day_mae_down,
                "mae_up_pct": day_mae_up,
                "mae_down_pts_xsp": day_mae_down / 100.0 * (open_price / 10.0),
                "mae_up_pts_xsp": day_mae_up / 100.0 * (open_price / 10.0),
                "worst_down_hour": worst_down_hour,
                "worst_up_hour": worst_up_hour,
                "close_ret_pct": close_ret,
                "recovered_up": int(close_price >= open_price),
                "n_bars": int(len(group)),
            }
        )
        gap_rows.append({"date": str(date), "n_bars": int(len(group))})

    sdf = pd.DataFrame(session_data)
    gap_df = pd.DataFrame(gap_rows)
    report.line(f"Sessions with >=5 hourly RTH bars: {len(sdf)}")
    if not gap_df.empty:
        report.line(
            f"Hourly session bar-count quality: min={gap_df['n_bars'].min()} "
            f"median={gap_df['n_bars'].median():.0f} max={gap_df['n_bars'].max()}"
        )

    if not sdf.empty:
        report.line()
        report.line("Hour-of-day: when does the worst drawdown occur? (put side proxy)")
        for hour in sorted(int(x) for x in sdf["worst_down_hour"].dropna().unique()):
            count = int((sdf["worst_down_hour"] == hour).sum())
            pct = count / len(sdf) * 100.0
            report.line(f"  {hour:02d}:00  {count:4d} ({pct:5.1f}%) {_hour_bar(pct)}")

        report.line()
        report.line("Hour-of-day: when does the worst upside move occur? (call side proxy)")
        for hour in sorted(int(x) for x in sdf["worst_up_hour"].dropna().unique()):
            count = int((sdf["worst_up_hour"] == hour).sum())
            pct = count / len(sdf) * 100.0
            report.line(f"  {hour:02d}:00  {count:4d} ({pct:5.1f}%) {_hour_bar(pct)}")

    # Recovery analysis (hourly proxy)
    report.line()
    report.line("Recovery analysis (hourly proxy): if price drops > N pts from open by hour H,")
    report.line("what % of the time does it close above open?")
    recovery_detail_rows: List[Dict[str, float]] = []
    for threshold_pts in [3, 5, 7, 9, 12]:
        recovery_by_hour: Dict[int, Dict[str, int]] = {}
        threshold_summary_stressed = 0
        threshold_summary_recovered = 0
        for date, group in sessions:
            if len(group) < 5:
                continue
            group = group.sort_values("timestamp_et").copy()
            open_price = float(group["Open"].iloc[0])
            close_price = float(group["Close"].iloc[-1])
            recovered = close_price >= open_price
            running_low = group["Low"].cummin()
            threshold_pct = threshold_pts / (open_price / 10.0) * 100.0
            breached = False
            for idx, _ in group.iterrows():
                up_to_idx = group.loc[:idx]
                mae_at_bar = float((open_price - float(up_to_idx["Low"].min())) / open_price * 100.0)
                if mae_at_bar >= threshold_pct:
                    hour = int(pd.Timestamp(group.loc[idx, "timestamp_et"]).hour)
                    slot = recovery_by_hour.setdefault(hour, {"stressed": 0, "recovered": 0})
                    slot["stressed"] += 1
                    slot["recovered"] += int(recovered)
                    threshold_summary_stressed += 1
                    threshold_summary_recovered += int(recovered)
                    breached = True
                    break
            if not breached:
                pass

        if threshold_summary_stressed == 0:
            report.line(f"  Drop > {threshold_pts:2d} pts: no sessions in hourly sample")
            continue
        overall_recovery = threshold_summary_recovered / threshold_summary_stressed * 100.0
        report.line(
            f"  Drop > {threshold_pts:2d} pts (n={threshold_summary_stressed}, "
            f"overall recovery={overall_recovery:.1f}%):"
        )
        for hour in sorted(recovery_by_hour.keys()):
            v = recovery_by_hour[hour]
            rec_pct = v["recovered"] / v["stressed"] * 100.0 if v["stressed"] else float("nan")
            report.line(
                f"    Breach at {hour:02d}:00: n={v['stressed']:4d} | "
                f"recovered={v['recovered']:4d} ({rec_pct:5.1f}%) {_hour_bar(rec_pct, 5.0)}"
            )
            recovery_detail_rows.append(
                {
                    "threshold_pts": threshold_pts,
                    "first_breach_hour": hour,
                    "stressed_sessions": v["stressed"],
                    "recovered_sessions": v["recovered"],
                    "recovery_pct": rec_pct,
                }
            )

    # MAE from ~10:00 onward proxy with 1h bars
    report.line()
    report.line("MAE from ~10:00 onward (1h proxy): uses first hourly bar with bar_end_et >= 10:00 ET")
    mae_from_10_rows: List[Dict[str, float]] = []
    for date, group in sessions:
        if len(group) < 5:
            continue
        group = group.sort_values("timestamp_et").copy()
        bars_10 = group[group["bar_end_et"].dt.hour >= 10].copy()
        if len(bars_10) < 3:
            continue
        open_10 = float(bars_10["Open"].iloc[0])
        mae_d = float((open_10 - float(bars_10["Low"].min())) / open_10 * 100.0)
        mae_u = float((float(bars_10["High"].max()) - open_10) / open_10 * 100.0)
        close_price = float(bars_10["Close"].iloc[-1])
        mae_from_10_rows.append(
            {
                "date": str(date),
                "open_proxy": open_10,
                "mae_down_pct": mae_d,
                "mae_up_pct": mae_u,
                "mae_down_pts_xsp": mae_d / 100.0 * (open_10 / 10.0),
                "mae_up_pts_xsp": mae_u / 100.0 * (open_10 / 10.0),
                "close_ret_pct_from_proxy_open": (close_price - open_10) / open_10 * 100.0,
                "n_bars_from_proxy": int(len(bars_10)),
            }
        )
    mdf = pd.DataFrame(mae_from_10_rows)
    if not mdf.empty:
        report.line(f"  Sessions: {len(mdf)}")
        for col, label in [("mae_down_pts_xsp", "Down (put)"), ("mae_up_pts_xsp", "Up (call)")]:
            report.line(
                f"  {label}: Mean={mdf[col].mean():.1f} pts | "
                f"P90={mdf[col].quantile(0.90):.1f} | P95={mdf[col].quantile(0.95):.1f} | "
                f"P99={mdf[col].quantile(0.99):.1f} | Max={mdf[col].max():.1f}"
            )
    else:
        report.line("  No sufficient sessions for 10:00 onward proxy subset.")

    # Layer 3: SL threshold survival analysis (linear proxy, capped)
    report.section("LAYER 3: SL THRESHOLD SURVIVAL ANALYSIS (LINEAR PROXY, CAPPED)")
    report.line(f"Assumptions: credit=${IC_CREDIT:.2f}, wing={WING_WIDTH:.1f} pts, cost-basis SL semantics")
    report.line("Using daily Open→Low MAE as put-side proxy. Linear spread approximation with wing cap:")
    report.line("  est_spread = min(credit + intrusion_pts, wing_width)")

    mae_pts_all = mae_down_pts.copy()
    results: List[Dict[str, float]] = []
    for distance in [6, 7, 8, 9, 10]:
        report.line(f"Strike distance: {distance} pts from open")
        intrusion = (mae_pts_all - distance).clip(lower=0.0)
        est_spread = (IC_CREDIT + intrusion).clip(upper=WING_WIDTH)
        for sl_mult in [2.0, 2.5, 3.0, 3.5, 4.0]:
            threshold = sl_mult * IC_CREDIT
            sl_triggered = int((est_spread >= threshold).sum())
            survival = (1.0 - sl_triggered / len(df)) * 100.0
            pnl_at_trigger = -(threshold - IC_CREDIT) * XSP_MULTIPLIER
            report.line(
                f"  SL={sl_mult:.1f}x (spread>=${threshold:.2f}, PnL≈-{abs(pnl_at_trigger):.0f}) "
                f"triggers={sl_triggered:4d}/{len(df):4d} | survival={survival:5.1f}%"
            )
            results.append(
                {
                    "distance_pts": distance,
                    "sl_mult": sl_mult,
                    "threshold_cost": threshold,
                    "triggers": sl_triggered,
                    "survival_pct": survival,
                    "pnl_at_trigger_usd": pnl_at_trigger,
                }
            )
        report.line()
    results_df = pd.DataFrame(results)

    report.line("Intraday asymmetry (put vs call side)")
    mean_down = float(df["mae_down"].mean())
    mean_up = float(df["mae_up"].mean())
    p95_down_pts = float(mae_down_pts.quantile(0.95))
    p95_up_pts = float(mae_up_pts.quantile(0.95))
    report.line(f"  Mean MAE down: {mean_down:.3f}% ({mae_down_pts.mean():.1f} XSP pts)")
    report.line(f"  Mean MAE up:   {mean_up:.3f}% ({mae_up_pts.mean():.1f} XSP pts)")
    report.line(
        f"  Ratio down/up: {mean_down / mean_up:.2f} "
        f"({'put riskier' if mean_down > mean_up else 'call riskier'})"
    )
    report.line(f"  P95 down: {p95_down_pts:.1f} pts")
    report.line(f"  P95 up:   {p95_up_pts:.1f} pts")
    report.line(f"  Supports left-biased IC? {'YES' if mean_down > mean_up * 1.1 else 'MARGINAL'}")

    # Export CSVs
    report.section("SAVE OUTPUTS")
    daily_export = df[
        [
            "Open",
            "High",
            "Low",
            "Close",
            "mae_down",
            "mae_up",
            "range_pct",
            "close_ret",
            "xsp_open_ref",
            "vix_prev_close",
            "vix_open",
        ]
    ].copy()
    p_daily = _save_csv(daily_export.reset_index(names="date"), "spx_daily_mae.csv", index=False)
    p_regime = _save_csv(regime_df, "spx_mae_by_vix_regime.csv", index=False) if not regime_df.empty else None
    p_sl = _save_csv(results_df, "spx_sl_survival_analysis.csv", index=False)
    p_breach = _save_csv(pd.DataFrame(breach_rows), "spx_breach_prob_by_distance.csv", index=False)
    p_hourly_summary = _save_csv(sdf, "spx_hourly_session_summary.csv", index=False) if not sdf.empty else None
    p_hourly_recovery = _save_csv(pd.DataFrame(recovery_detail_rows), "spx_hourly_recovery_by_breach_hour.csv", index=False) if recovery_detail_rows else None
    p_mae10 = _save_csv(mdf, "spx_mae_from_10am_proxy_1h.csv", index=False) if not mdf.empty else None
    p_hourly_raw = _save_csv(h, "spx_hourly_rth_1h_used.csv", index=False)
    p_gap = _save_csv(gap_df, "spx_hourly_session_bar_counts.csv", index=False) if not gap_df.empty else None

    report.line(f"Saved: {p_daily}")
    if p_regime:
        report.line(f"Saved: {p_regime}")
    report.line(f"Saved: {p_sl}")
    report.line(f"Saved: {p_breach}")
    if p_hourly_summary:
        report.line(f"Saved: {p_hourly_summary}")
    if p_hourly_recovery:
        report.line(f"Saved: {p_hourly_recovery}")
    if p_mae10:
        report.line(f"Saved: {p_mae10}")
    report.line(f"Saved: {p_hourly_raw}")
    if p_gap:
        report.line(f"Saved: {p_gap}")

    report.section("KEY QUESTIONS THIS ANSWERS")
    report.line("1. Are delta~0.10 strike distances (~7-9 XSP pts) body or tail of MAE distribution?")
    report.line("2. How fat are intraday MAE tails vs normal assumptions?")
    report.line("3. Does a VIX gate (e.g., 25) exclude materially worse MAE regimes?")
    report.line("4. If stressed early (hourly proxy), how often does price recover by close?")
    report.line("5. Is there down/up asymmetry supporting left-biased condors?")
    report.line("6. At cost-basis SL multiples (2x-4x), what is proxy trigger frequency?")
    report.line()
    report.line("EDA complete.")

    report.save(REPORT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())

