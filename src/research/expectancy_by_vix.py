"""Expectancy by VIX bucket for XSP 0DTE Iron Condor strategy economics.

Standalone CLI:
    python -m src.research.expectancy_by_vix
    python src/research/expectancy_by_vix.py
"""

from __future__ import annotations

import copy
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

try:
    from . import synthetic_backtest as sb
except ImportError:
    REPO_ROOT_FALLBACK = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT_FALLBACK) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT_FALLBACK))
    from src.research import synthetic_backtest as sb


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "backtest_config.yaml"
OUTPUT_DIR = REPO_ROOT / "outputs" / "expectancy_by_vix"
VIX_EDA_DIR = REPO_ROOT / "outputs" / "vix_breach_eda"

VIX_BUCKETS = [0.0, 15.0, 20.0, 25.0, 30.0, np.inf]
VIX_BUCKET_LABELS = ["0-15", "15-20", "20-25", "25-30", "30+"]
VIX_BUCKET_DTYPE = pd.CategoricalDtype(categories=VIX_BUCKET_LABELS, ordered=True)

SCENARIO_ORDER = ["hold_to_expiry", "tp50_or_expiry", "tp50_sl_capped", "worst_case"]
SCENARIO_LABELS = {
    "hold_to_expiry": "hold_to_expiry",
    "tp50_or_expiry": "tp50_or_expiry",
    "tp50_sl_capped": "tp50_sl_capped",
    "worst_case": "worst_case",
}
TARGET_SCENARIO = "tp50_sl_capped"
L3_VALUES = [15, 18, 20, 22, 25, 28, 30, 35, 999]
WING_PROJECTION_VALUES = [2, 3, 5, 10]

# Commission constants (configurable, do not hardcode throughout the script)
COMMISSION_WIN_OPEN_ONLY = 4.92
COMMISSION_LOSS_ROUNDTRIP = 9.84
COMMISSION_ALWAYS_CLOSE = 9.84

COMMISSION_MODELS = {
    "hold_to_expiry_like": {
        "win_commission": COMMISSION_WIN_OPEN_ONLY,
        "loss_commission": COMMISSION_LOSS_ROUNDTRIP,
        "description": "Win pays open only ($4.92); loss pays roundtrip ($9.84)",
    },
    "always_close": {
        "win_commission": COMMISSION_ALWAYS_CLOSE,
        "loss_commission": COMMISSION_ALWAYS_CLOSE,
        "description": "Win/loss both pay roundtrip ($9.84)",
    },
}

BUCKET_COLORS = {
    "0-15": "#4C78A8",
    "15-20": "#59A14F",
    "20-25": "#F28E2B",
    "25-30": "#E15759",
    "30+": "#B07AA1",
}


@dataclass
class CommissionModel:
    name: str
    win_commission: float
    loss_commission: float


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

    def table(self, df: pd.DataFrame, index: bool = True, float_fmt: str = "{:.4f}") -> None:
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
            "figure.figsize": (10, 6),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.frameon": True,
        }
    )


def _safe_div(num: float, den: float) -> float:
    if den == 0 or not np.isfinite(den):
        return float("nan")
    return float(num) / float(den)


def _write_csv(df: pd.DataFrame, path: Path, index: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def _fmt_pct(x: float) -> str:
    return "NA" if not np.isfinite(x) else f"{x:.1%}"


def _fmt_usd(x: float) -> str:
    return "NA" if not np.isfinite(x) else f"${x:,.2f}"


def _fmt_num(x: float) -> str:
    return "NA" if not np.isfinite(x) else f"{x:.2f}"


def _bucketize_vix(vix: pd.Series) -> pd.Categorical:
    return pd.cut(vix, bins=VIX_BUCKETS, labels=VIX_BUCKET_LABELS, right=False, include_lowest=True)


def _load_config() -> Dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config YAML: {CONFIG_PATH}")
    return cfg


def _cfg_with_wing(base_cfg: Dict[str, Any], wing_width: float) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("default", {})
    if not isinstance(cfg["default"], dict):
        raise ValueError("config.default must be a mapping")
    cfg["default"]["wing_width"] = float(wing_width)
    # Keep scenarios available in backtester; no changes needed for all 4 outcomes.
    return cfg


def _load_cached_backtest_data(cache_file: Path, start: str, end: str) -> pd.DataFrame:
    data = pd.read_csv(cache_file, parse_dates=["date"])
    data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)
    data = data[(data["date"] >= pd.to_datetime(start)) & (data["date"] <= pd.to_datetime(end))].copy()
    data = data.sort_values("date").reset_index(drop=True)
    for col in [c for c in ["vix_open", "vix_high", "vix_low", "vix_close"] if c in data.columns]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    vix_cols = [c for c in ["vix_open", "vix_high", "vix_low", "vix_close"] if c in data.columns]
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

    report.section("Backtester Data Load")
    report.line(f"Date range: {start_date} -> {end_date}")

    candidates = [
        VIX_EDA_DIR / "_module_b_source_data_snapshot.csv",
        cache_file,
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            data = _load_cached_backtest_data(candidate, start_date, end_date)
            if _data_has_usable_vix(data):
                report.line(f"Using source data cache: {candidate}")
                summary = sb.validate_data(data)
                report.line(
                    "Source rows: {rows} | Date range: {d0} -> {d1}".format(
                        rows=len(data), d0=summary["date_range"][0], d1=summary["date_range"][1]
                    )
                )
                return data
            report.line(f"Cache present but unusable VIX data: {candidate}")
        except Exception as exc:
            report.line(f"WARNING reading cache {candidate}: {exc.__class__.__name__}: {exc}")

    report.line("No reusable source cache found. Downloading via synthetic_backtest._download_spx_vix()")
    data = sb._download_spx_vix(start_date=start_date, end_date=end_date)
    data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)
    data = data[(data["date"] >= pd.to_datetime(start_date)) & (data["date"] <= pd.to_datetime(end_date))].copy()
    data = data.sort_values("date").reset_index(drop=True)
    if not _data_has_usable_vix(data):
        raise ValueError("Downloaded backtester source data has no usable VIX values")
    snapshot = OUTPUT_DIR / "_source_data_snapshot.csv"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(snapshot, index=False)
    report.line(f"Saved source snapshot: {snapshot}")
    summary = sb.validate_data(data)
    report.line(
        "Source rows: {rows} | Date range: {d0} -> {d1}".format(
            rows=len(data), d0=summary["date_range"][0], d1=summary["date_range"][1]
        )
    )
    return data


def _required_result_columns() -> List[str]:
    cols = ["date", "vix", "entry_credit", "entry_credit_usd", "short_put", "short_call", "spx_open", "spx_high", "spx_low", "skipped"]
    cols += [f"outcome_{s}" for s in SCENARIO_ORDER]
    cols += [
        "hold_to_expiry_pre_commission",
        "tp50_or_expiry_pre_commission",
        "tp50_sl_capped_pre_commission",
        "worst_case_pre_commission",
        "put_breached",
        "call_breached",
        "put_wing_breached",
        "call_wing_breached",
    ]
    return cols


def _load_results_snapshot(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df


def _results_snapshot_is_usable(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    req = set(_required_result_columns())
    return req.issubset(set(df.columns))


def _run_backtest_or_reuse(cfg: Dict[str, Any], source_data: pd.DataFrame, wing_width: float, report: Reporter) -> pd.DataFrame:
    out_cache = OUTPUT_DIR / f"backtest_results_wing{int(wing_width)}.csv"
    legacy_candidates = [VIX_EDA_DIR / f"_module_b_results_wing{int(wing_width)}.csv"]

    for candidate in [out_cache, *legacy_candidates]:
        if not candidate.exists():
            continue
        try:
            df = _load_results_snapshot(candidate)
            if _results_snapshot_is_usable(df):
                report.line(f"Using backtest results snapshot (wing={wing_width:g}): {candidate}")
                return df
            report.line(f"Result snapshot missing required columns, re-running: {candidate}")
        except Exception as exc:
            report.line(f"WARNING reading results snapshot {candidate}: {exc.__class__.__name__}: {exc}")

    cfg_wing = _cfg_with_wing(cfg, wing_width=wing_width)
    params = sb._backtest_params(cfg_wing)
    report.line(
        "Running backtest wing={w:.1f} | delta={d:.2f} | entry_hour={eh:.2f}".format(
            w=float(params["wing_width"]),
            d=float(params["target_delta"]),
            eh=float(params["entry_hour"]),
        )
    )
    results = sb.run_backtest(cfg_wing, data=source_data)
    out_cache.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_cache, index=False)
    report.line(f"Saved results snapshot: {out_cache}")
    return results


def _scenario_pre_col(scenario: str) -> str:
    mapping = {
        "hold_to_expiry": "hold_to_expiry_pre_commission",
        "tp50_or_expiry": "tp50_or_expiry_pre_commission",
        "tp50_sl_capped": "tp50_sl_capped_pre_commission",
        "worst_case": "worst_case_pre_commission",
    }
    if scenario not in mapping:
        raise KeyError(f"Unsupported scenario: {scenario}")
    return mapping[scenario]


def _prepare_trade_frame(results: pd.DataFrame, report: Optional[Reporter] = None) -> pd.DataFrame:
    if report is not None:
        report.section("Normalize Backtest Output")
    df = results.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    num_cols = [
        "vix",
        "entry_credit",
        "entry_credit_usd",
        "short_put",
        "short_call",
        "spx_open",
        "spx_high",
        "spx_low",
        "outcome_hold_to_expiry",
        "outcome_tp50_or_expiry",
        "outcome_tp50_sl_capped",
        "outcome_worst_case",
        "hold_to_expiry_pre_commission",
        "tp50_or_expiry_pre_commission",
        "tp50_sl_capped_pre_commission",
        "worst_case_pre_commission",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    bool_cols = ["put_breached", "call_breached", "put_wing_breached", "call_wing_breached", "skipped"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    mask = (
        (~df["skipped"])
        & np.isfinite(df["vix"])
        & np.isfinite(df["entry_credit"])
        & np.isfinite(df["spx_open"])
    )
    df = df.loc[mask].copy()
    if df.empty:
        raise ValueError("No tradable rows after filtering skipped/invalid rows")

    df["entry_credit_usd_eff"] = np.where(
        np.isfinite(df.get("entry_credit_usd", np.nan)),
        pd.to_numeric(df.get("entry_credit_usd"), errors="coerce"),
        pd.to_numeric(df["entry_credit"], errors="coerce") * 100.0,
    )
    # Fallback if column exists but has NaNs
    missing_credit_usd = ~np.isfinite(df["entry_credit_usd_eff"])
    if missing_credit_usd.any():
        df.loc[missing_credit_usd, "entry_credit_usd_eff"] = pd.to_numeric(df.loc[missing_credit_usd, "entry_credit"], errors="coerce") * 100.0

    df["vix_bucket"] = _bucketize_vix(df["vix"]).astype(VIX_BUCKET_DTYPE)
    df = df[~df["vix_bucket"].isna()].copy()

    df["any_short_breach"] = df["put_breached"] | df["call_breached"]
    df["any_wing_breach"] = df["put_wing_breached"] | df["call_wing_breached"]

    spot = df["spx_open"].astype(float)
    df["mae_put_pct"] = np.maximum(0.0, df["short_put"].astype(float) - df["spx_low"].astype(float)) / spot * 100.0
    df["mae_call_pct"] = np.maximum(0.0, df["spx_high"].astype(float) - df["short_call"].astype(float)) / spot * 100.0
    df["mae_max_pct"] = np.maximum(df["mae_put_pct"], df["mae_call_pct"])

    if report is not None:
        report.line(f"Tradable rows after filters: {len(df)}")
        bucket_counts = df.groupby("vix_bucket", observed=True).size().reindex(VIX_BUCKET_LABELS, fill_value=0).rename("n_trades").to_frame()
        report.table(bucket_counts, index=True, float_fmt="{:.0f}")
    return df.sort_values("date").reset_index(drop=True)


def _commission_model(name: str) -> CommissionModel:
    spec = COMMISSION_MODELS[name]
    return CommissionModel(
        name=name,
        win_commission=float(spec["win_commission"]),
        loss_commission=float(spec["loss_commission"]),
    )


def _apply_commissions_from_gross(gross_pnl: pd.Series, model: CommissionModel) -> pd.Series:
    gross = pd.to_numeric(gross_pnl, errors="coerce")
    is_win = gross > 0
    comm = np.where(is_win, model.win_commission, model.loss_commission)
    return gross - comm


def _expectancy_metrics_for_series(gross_pnl: pd.Series, entry_credit_usd: pd.Series, total_years: float) -> Dict[str, float]:
    pnl = pd.to_numeric(gross_pnl, errors="coerce").dropna()
    if pnl.empty:
        return {
            "n_trades": 0.0,
            "win_rate": np.nan,
            "avg_win_gross": np.nan,
            "avg_loss_gross": np.nan,
            "avg_credit_usd": np.nan,
            "e_gross": np.nan,
            "trades_per_year": np.nan,
            "e_annual_gross": np.nan,
        }
    credit = pd.to_numeric(entry_credit_usd.loc[pnl.index], errors="coerce")
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    wr = float((pnl > 0).mean())
    avg_win = float(wins.mean()) if not wins.empty else np.nan
    avg_loss = float(losses.mean()) if not losses.empty else np.nan
    # Use unconditional mean directly (equivalent to WR-weighted formula, but more stable with empty-side edge cases).
    e_gross = float(pnl.mean())
    n_trades = float(len(pnl))
    trades_per_year = _safe_div(n_trades, total_years)
    return {
        "n_trades": n_trades,
        "win_rate": wr,
        "avg_win_gross": avg_win,
        "avg_loss_gross": avg_loss,
        "avg_credit_usd": float(credit.mean()) if credit.notna().any() else np.nan,
        "e_gross": e_gross,
        "trades_per_year": trades_per_year,
        "e_annual_gross": e_gross * trades_per_year if np.isfinite(e_gross) and np.isfinite(trades_per_year) else np.nan,
    }


def _breakeven_win_rate(avg_win_gross: float, avg_loss_gross: float, model: CommissionModel) -> float:
    if not (np.isfinite(avg_win_gross) and np.isfinite(avg_loss_gross)):
        return float("nan")
    win_net = avg_win_gross - model.win_commission
    loss_net = avg_loss_gross - model.loss_commission
    denom = win_net - loss_net
    if not np.isfinite(denom) or denom == 0:
        return float("nan")
    wr = -loss_net / denom
    return float(wr)


def _compute_expectancy_table(df: pd.DataFrame, total_years: float, report: Reporter) -> pd.DataFrame:
    report.section("Analysis 1: Expectancy by VIX Bucket (Per Scenario)")
    hold_model = _commission_model("hold_to_expiry_like")
    close_model = _commission_model("always_close")

    rows: List[Dict[str, Any]] = []
    for scenario in SCENARIO_ORDER:
        pre_col = _scenario_pre_col(scenario)
        for bucket in VIX_BUCKET_LABELS:
            sub = df[df["vix_bucket"] == bucket]
            metrics = _expectancy_metrics_for_series(sub[pre_col], sub["entry_credit_usd_eff"], total_years=total_years)
            gross_series = pd.to_numeric(sub[pre_col], errors="coerce")
            net_hold = _apply_commissions_from_gross(gross_series, hold_model)
            net_close = _apply_commissions_from_gross(gross_series, close_model)
            e_net_hold = float(net_hold.mean()) if net_hold.notna().any() else np.nan
            e_net_close = float(net_close.mean()) if net_close.notna().any() else np.nan
            trades_per_year = metrics["trades_per_year"]
            wr_be_hold = _breakeven_win_rate(metrics["avg_win_gross"], metrics["avg_loss_gross"], hold_model)
            wr_be_close = _breakeven_win_rate(metrics["avg_win_gross"], metrics["avg_loss_gross"], close_model)
            margin_hold = metrics["win_rate"] - wr_be_hold if np.isfinite(metrics["win_rate"]) and np.isfinite(wr_be_hold) else np.nan
            margin_close = metrics["win_rate"] - wr_be_close if np.isfinite(metrics["win_rate"]) and np.isfinite(wr_be_close) else np.nan

            rows.append(
                {
                    "scenario": scenario,
                    "vix_bucket": bucket,
                    "n_trades": int(metrics["n_trades"]),
                    "win_rate": metrics["win_rate"],
                    "avg_win_gross": metrics["avg_win_gross"],
                    "avg_loss_gross": metrics["avg_loss_gross"],
                    "avg_credit_usd": metrics["avg_credit_usd"],
                    "breach_rate_any_short": float(sub["any_short_breach"].mean()) if len(sub) else np.nan,
                    "breach_rate_any_wing": float(sub["any_wing_breach"].mean()) if len(sub) else np.nan,
                    "e_gross": metrics["e_gross"],
                    "e_net_hold_model": e_net_hold,
                    "e_net_always_close": e_net_close,
                    "trades_per_year": trades_per_year,
                    "e_annual_gross": metrics["e_annual_gross"],
                    "e_annual_net_hold_model": e_net_hold * trades_per_year if np.isfinite(e_net_hold) and np.isfinite(trades_per_year) else np.nan,
                    "e_annual_net_always_close": e_net_close * trades_per_year if np.isfinite(e_net_close) and np.isfinite(trades_per_year) else np.nan,
                    "wr_breakeven_hold_model": wr_be_hold,
                    "wr_breakeven_always_close": wr_be_close,
                    "margin_hold_model": margin_hold,
                    "margin_always_close": margin_close,
                    "commission_model_hold_win": hold_model.win_commission,
                    "commission_model_hold_loss": hold_model.loss_commission,
                    "commission_model_close_win": close_model.win_commission,
                    "commission_model_close_loss": close_model.loss_commission,
                }
            )

    out = pd.DataFrame(rows)
    out["scenario"] = pd.Categorical(out["scenario"], categories=SCENARIO_ORDER, ordered=True)
    out["vix_bucket"] = pd.Categorical(out["vix_bucket"], categories=VIX_BUCKET_LABELS, ordered=True)
    out = out.sort_values(["scenario", "vix_bucket"]).reset_index(drop=True)
    _write_csv(out, OUTPUT_DIR / "expectancy_table.csv", index=False)

    preview_cols = ["scenario", "vix_bucket", "n_trades", "win_rate", "avg_credit_usd", "e_gross", "e_net_hold_model", "e_net_always_close"]
    report.table(out[preview_cols], index=False, float_fmt="{:.4f}")
    return out


def _breakeven_status(margin: float) -> str:
    if not np.isfinite(margin):
        return "NA"
    if margin > 0.10:
        return "SAFE"
    if margin > 0.03:
        return "THIN"
    if margin >= 0:
        return "VERY_THIN"
    return "NEGATIVE"


def _analysis_breakeven(expectancy_table: pd.DataFrame, report: Reporter) -> pd.DataFrame:
    report.section("Analysis 2: Breakeven Win Rate by VIX Bucket")
    sub = expectancy_table[expectancy_table["scenario"] == TARGET_SCENARIO].copy()
    if sub.empty:
        raise ValueError(f"No rows found for target scenario {TARGET_SCENARIO}")

    rows: List[Dict[str, Any]] = []
    for _, r in sub.iterrows():
        for model_name, wr_col, margin_col in [
            ("hold_to_expiry_like", "wr_breakeven_hold_model", "margin_hold_model"),
            ("always_close", "wr_breakeven_always_close", "margin_always_close"),
        ]:
            margin = float(r[margin_col]) if pd.notna(r[margin_col]) else np.nan
            rows.append(
                {
                    "scenario": TARGET_SCENARIO,
                    "commission_model": model_name,
                    "vix_bucket": r["vix_bucket"],
                    "n_trades": int(r["n_trades"]),
                    "actual_wr": float(r["win_rate"]) if pd.notna(r["win_rate"]) else np.nan,
                    "breakeven_wr": float(r[wr_col]) if pd.notna(r[wr_col]) else np.nan,
                    "margin_of_safety": margin,
                    "status": _breakeven_status(margin),
                    "avg_win_gross": float(r["avg_win_gross"]) if pd.notna(r["avg_win_gross"]) else np.nan,
                    "avg_loss_gross": float(r["avg_loss_gross"]) if pd.notna(r["avg_loss_gross"]) else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out["vix_bucket"] = pd.Categorical(out["vix_bucket"], categories=VIX_BUCKET_LABELS, ordered=True)
    out["commission_model"] = pd.Categorical(out["commission_model"], categories=["hold_to_expiry_like", "always_close"], ordered=True)
    out = out.sort_values(["commission_model", "vix_bucket"]).reset_index(drop=True)
    _write_csv(out, OUTPUT_DIR / "breakeven_analysis.csv", index=False)
    report.table(out, index=False, float_fmt="{:.4f}")

    chart_df = out[out["commission_model"] == "hold_to_expiry_like"].copy().sort_values("vix_bucket")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(chart_df))
    width = 0.36
    ax.bar(x - width / 2, chart_df["actual_wr"].fillna(0).to_numpy(dtype=float), width, label="Actual WR", color="#4C78A8")
    ax.bar(x + width / 2, chart_df["breakeven_wr"].fillna(0).to_numpy(dtype=float), width, label="Breakeven WR", color="#E15759")
    ax.set_xticks(x)
    ax.set_xticklabels(chart_df["vix_bucket"].astype(str).tolist())
    ax.set_ylim(0, 1)
    ax.set_ylabel("Win rate")
    ax.set_title("Breakeven Win Rate Margin by VIX Bucket (tp50_sl_capped)")
    ax.legend(loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(x, chart_df["margin_of_safety"].to_numpy(dtype=float), marker="o", color="#59A14F", linewidth=2, label="Margin of safety")
    y2 = chart_df["margin_of_safety"].replace([np.inf, -np.inf], np.nan).dropna()
    if not y2.empty:
        lim = max(0.05, float(np.nanmax(np.abs(y2))) * 1.25)
        ax2.set_ylim(-lim, lim)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylabel("Margin (Actual WR - Breakeven WR)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart_breakeven_margin.png", dpi=180)
    plt.close(fig)

    return out


def _daily_series_with_gate(df: pd.DataFrame, pnl_col: str, l3_threshold: float) -> pd.Series:
    work = df[["date", "vix", pnl_col]].copy()
    work[pnl_col] = pd.to_numeric(work[pnl_col], errors="coerce")
    work = work.dropna(subset=[pnl_col])
    work = work.sort_values("date")
    selected = work["vix"] < float(l3_threshold)
    work["pnl_selected"] = np.where(selected, work[pnl_col], 0.0)
    daily = work.groupby("date", as_index=True)["pnl_selected"].sum().sort_index()
    return daily


def _max_drawdown(cum_pnl: pd.Series) -> float:
    if cum_pnl is None or cum_pnl.empty:
        return float("nan")
    running_max = cum_pnl.cummax()
    dd = cum_pnl - running_max
    return float(dd.min()) if not dd.empty else float("nan")


def _max_consecutive_losses(values: pd.Series) -> int:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    best = 0
    cur = 0
    for x in arr:
        if x < 0:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def _annualized_sharpe_from_daily(daily_pnl: pd.Series) -> float:
    if daily_pnl is None or daily_pnl.empty:
        return float("nan")
    x = pd.to_numeric(daily_pnl, errors="coerce").dropna()
    if x.empty:
        return float("nan")
    std = float(x.std(ddof=1))
    mean = float(x.mean())
    if not np.isfinite(std) or std == 0:
        return float("nan")
    return float((mean / std) * math.sqrt(252.0))


def _net_pnl_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hold_model = _commission_model("hold_to_expiry_like")
    close_model = _commission_model("always_close")
    out["pnl_hold_net_hold_model"] = _apply_commissions_from_gross(out["hold_to_expiry_pre_commission"], hold_model)
    out["pnl_tp50sl_net_hold_model"] = _apply_commissions_from_gross(out["tp50_sl_capped_pre_commission"], hold_model)
    out["pnl_tp50sl_net_always_close"] = _apply_commissions_from_gross(out["tp50_sl_capped_pre_commission"], close_model)
    out["pnl_tporexp_net_always_close"] = _apply_commissions_from_gross(out["tp50_or_expiry_pre_commission"], close_model)
    return out


def _analysis_l3_thresholds(df: pd.DataFrame, total_years: float, report: Reporter) -> Tuple[pd.DataFrame, Dict[int, pd.Series]]:
    report.section("Analysis 3: Cumulative PnL Simulation by L3 Threshold")
    work = _net_pnl_columns(df)

    rows: List[Dict[str, Any]] = []
    tp50_curves: Dict[int, pd.Series] = {}
    for l3 in L3_VALUES:
        selected = work["vix"] < float(l3)
        selected_trade_pnls_hold = work.loc[selected, "pnl_hold_net_hold_model"]
        selected_trade_pnls_tp50 = work.loc[selected, "pnl_tp50sl_net_always_close"]

        daily_hold = _daily_series_with_gate(work, "pnl_hold_net_hold_model", l3)
        daily_tp50 = _daily_series_with_gate(work, "pnl_tp50sl_net_always_close", l3)
        cum_tp50 = daily_tp50.cumsum()
        tp50_curves[int(l3)] = cum_tp50

        total_hold = float(selected_trade_pnls_hold.sum()) if selected_trade_pnls_hold.notna().any() else np.nan
        total_tp50 = float(selected_trade_pnls_tp50.sum()) if selected_trade_pnls_tp50.notna().any() else np.nan

        rows.append(
            {
                "l3": int(l3),
                "l3_label": "no_gate" if int(l3) >= 999 else str(int(l3)),
                "n_trades": int(selected.sum()),
                "total_pnl_hold_model": total_hold,
                "annual_pnl_hold_model": total_hold / total_years if np.isfinite(total_hold) and total_years > 0 else np.nan,
                "sharpe_hold_model": _annualized_sharpe_from_daily(daily_hold),
                "max_dd_hold_model": _max_drawdown(daily_hold.cumsum()),
                "max_consec_losses_hold_model": _max_consecutive_losses(selected_trade_pnls_hold),
                "total_pnl_tp50sl_always_close": total_tp50,
                "annual_pnl_tp50sl_always_close": total_tp50 / total_years if np.isfinite(total_tp50) and total_years > 0 else np.nan,
                "sharpe_tp50sl_always_close": _annualized_sharpe_from_daily(daily_tp50),
                "max_dd_tp50sl_always_close": _max_drawdown(cum_tp50),
                "max_consec_losses_tp50sl_always_close": _max_consecutive_losses(selected_trade_pnls_tp50),
            }
        )

    out = pd.DataFrame(rows).sort_values("l3").reset_index(drop=True)
    _write_csv(out, OUTPUT_DIR / "l3_comparison.csv", index=False)
    report.table(out, index=False, float_fmt="{:.4f}")

    # Overlay cumulative curves (tp50_sl_capped, always-close model)
    fig, ax = plt.subplots(figsize=(12, 7))
    palette = sns.color_palette("viridis", len(L3_VALUES)) if sns is not None else None
    for i, l3 in enumerate(L3_VALUES):
        curve = tp50_curves[int(l3)]
        if curve.empty:
            continue
        label = "L3=NoGate" if int(l3) >= 999 else f"L3={int(l3)}"
        color = palette[i] if palette is not None else None
        ax.plot(curve.index, curve.values, label=label, linewidth=1.8, color=color)
    ax.set_title("Cumulative PnL by L3 Threshold (tp50_sl_capped, always-close commissions)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL (USD)")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart_l3_cumulative_pnl.png", dpi=180)
    plt.close(fig)

    # Dual-axis tradeoff chart: Sharpe vs N trades (tp50 scenario)
    trade_df = out.copy()
    fig, ax1 = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(trade_df))
    ax1.plot(x, trade_df["sharpe_tp50sl_always_close"].to_numpy(dtype=float), marker="o", linewidth=2, color="#4C78A8", label="Sharpe (tp50_sl_capped)")
    ax1.set_ylabel("Sharpe (annualized, daily PnL)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(trade_df["l3_label"].tolist())
    ax1.set_xlabel("L3 threshold")
    ax1.set_title("L3 Tradeoff: Sharpe vs Trade Count (tp50_sl_capped)")

    ax2 = ax1.twinx()
    ax2.bar(x, trade_df["n_trades"].to_numpy(dtype=float), alpha=0.25, color="#59A14F", label="N trades")
    ax2.set_ylabel("N trades")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart_l3_tradeoff.png", dpi=180)
    plt.close(fig)
    return out, tp50_curves


def _analysis_credit_risk_tradeoff(expectancy_table: pd.DataFrame, report: Reporter) -> pd.DataFrame:
    report.section("Analysis 4: Credit vs Risk Tradeoff")
    sub = expectancy_table[expectancy_table["scenario"] == TARGET_SCENARIO].copy().sort_values("vix_bucket")
    sub["credit_per_risk"] = sub["avg_credit_usd"] / sub["avg_loss_gross"].abs()
    out = sub[
        [
            "scenario",
            "vix_bucket",
            "n_trades",
            "avg_credit_usd",
            "avg_loss_gross",
            "breach_rate_any_short",
            "breach_rate_any_wing",
            "e_net_hold_model",
            "e_net_always_close",
            "credit_per_risk",
        ]
    ].copy()
    _write_csv(out, OUTPUT_DIR / "credit_risk_tradeoff.csv", index=False)
    report.table(out, index=False, float_fmt="{:.4f}")

    fig, ax1 = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(out))
    ax1.bar(x, out["avg_credit_usd"].to_numpy(dtype=float), color=[BUCKET_COLORS[str(b)] for b in out["vix_bucket"].astype(str)], alpha=0.8, label="Avg credit ($)")
    ax1.set_ylabel("Average credit (USD)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(out["vix_bucket"].astype(str).tolist())
    ax1.set_xlabel("VIX bucket")
    ax1.set_title("Credit vs Breach Tradeoff by VIX Bucket (tp50_sl_capped)")

    ax2 = ax1.twinx()
    ax2.plot(x, out["breach_rate_any_short"].to_numpy(dtype=float), marker="o", color="#7A1E1E", linewidth=2, label="Any short breach rate")
    ax2.set_ylabel("Breach probability")
    ax2.set_ylim(0, 1)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart_credit_vs_breach.png", dpi=180)
    plt.close(fig)
    return out


def _projection_rows_for_wing(df: pd.DataFrame, wing: int, total_years: float) -> pd.DataFrame:
    hold_model = _commission_model("hold_to_expiry_like")
    rows: List[Dict[str, Any]] = []
    for bucket in VIX_BUCKET_LABELS:
        sub = df[df["vix_bucket"] == bucket]
        gross = pd.to_numeric(sub["tp50_sl_capped_pre_commission"], errors="coerce")
        net_hold = _apply_commissions_from_gross(gross, hold_model)
        wins = gross[gross > 0]
        losses = gross[gross <= 0]
        wr = float((gross > 0).mean()) if gross.notna().any() else np.nan
        avg_win = float(wins.mean()) if not wins.empty else np.nan
        avg_loss = float(losses.mean()) if not losses.empty else np.nan
        wr_be = _breakeven_win_rate(avg_win, avg_loss, hold_model)
        rows.append(
            {
                "wing_width": int(wing),
                "vix_bucket": bucket,
                "n_trades": int(len(sub)),
                "win_rate": wr,
                "avg_credit_usd": float(pd.to_numeric(sub["entry_credit_usd_eff"], errors="coerce").mean()) if len(sub) else np.nan,
                "avg_win_gross": avg_win,
                "avg_loss_gross": avg_loss,
                "e_gross": float(gross.mean()) if gross.notna().any() else np.nan,
                "e_net_hold_model": float(net_hold.mean()) if net_hold.notna().any() else np.nan,
                "e_annual_net_hold_model": float(net_hold.mean()) * (len(sub) / total_years) if net_hold.notna().any() and total_years > 0 else np.nan,
                "wr_breakeven_hold_model": wr_be,
                "margin_hold_model": wr - wr_be if np.isfinite(wr) and np.isfinite(wr_be) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _analysis_wing_width_projection(base_cfg: Dict[str, Any], source_data: pd.DataFrame, total_years: float, report: Reporter) -> pd.DataFrame:
    report.section("Analysis 5: Sensitivity to Wing Width (Projection)")
    rows: List[pd.DataFrame] = []
    failures: List[str] = []
    for wing in WING_PROJECTION_VALUES:
        try:
            results_w = _run_backtest_or_reuse(base_cfg, source_data, wing_width=float(wing), report=report)
            df_w = _prepare_trade_frame(results_w, report=None)
            rows.append(_projection_rows_for_wing(df_w, wing=wing, total_years=total_years))
        except Exception as exc:
            report.line(f"Wing width {wing} failed: {exc.__class__.__name__}: {exc}")
            failures.append(f"wing={wing}: {exc.__class__.__name__}: {exc}")
            continue

    if not rows:
        msg = "Wing width projection requires backtester re-run with different wing_width parameter. Current analysis limited to wing=2."
        report.line(msg)
        empty = pd.DataFrame(columns=["wing_width", "vix_bucket", "e_net_hold_model"])
        _write_csv(empty, OUTPUT_DIR / "wing_width_projection.csv", index=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, msg, ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "chart_wing_width_expectancy.png", dpi=180)
        plt.close(fig)
        return empty

    out = pd.concat(rows, ignore_index=True)
    out["vix_bucket"] = pd.Categorical(out["vix_bucket"], categories=VIX_BUCKET_LABELS, ordered=True)
    out = out.sort_values(["wing_width", "vix_bucket"]).reset_index(drop=True)
    _write_csv(out, OUTPUT_DIR / "wing_width_projection.csv", index=False)
    report.table(out, index=False, float_fmt="{:.4f}")
    if failures:
        report.line("Partial projection completed. Failed wings:")
        for item in failures:
            report.line(f"  - {item}")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for bucket in VIX_BUCKET_LABELS:
        sub = out[out["vix_bucket"] == bucket].sort_values("wing_width")
        if sub.empty:
            continue
        ax.plot(
            sub["wing_width"].to_numpy(dtype=float),
            sub["e_net_hold_model"].to_numpy(dtype=float),
            marker="o",
            linewidth=2,
            label=bucket,
            color=BUCKET_COLORS.get(str(bucket), None),
        )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(WING_PROJECTION_VALUES)
    ax.set_xlabel("Wing width")
    ax.set_ylabel("E_net per trade (USD, hold-to-expiry-like commissions)")
    ax.set_title("Wing Width Projection: Expectancy by VIX Bucket (tp50_sl_capped)")
    ax.legend(title="VIX bucket", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart_wing_width_expectancy.png", dpi=180)
    plt.close(fig)

    # Key output: where VIX 20-25 becomes clearly profitable.
    sub_2025 = out[out["vix_bucket"] == "20-25"].copy().sort_values("wing_width")
    if not sub_2025.empty:
        clear = sub_2025[(sub_2025["e_net_hold_model"] > 0) & (sub_2025["margin_hold_model"] > 0.05)]
        if not clear.empty:
            first_w = int(clear.iloc[0]["wing_width"])
            report.line(f"VIX 20-25 becomes clearly profitable (E_net>0 and margin>5%) at wing_width={first_w}.")
        else:
            report.line("VIX 20-25 does not become clearly profitable under the chosen criterion (E_net>0 and margin>5%) in tested wings.")
    return out


def _load_trade_day_count(start_date: pd.Timestamp, end_date: pd.Timestamp, source_data: pd.DataFrame, report: Reporter) -> Tuple[int, float, str]:
    """Return (trading_days, total_years, source_label). Prefer vix_breach_eda module A cache if available."""
    module_a_cache = VIX_EDA_DIR / "_module_a_spx_vix_cache.csv"
    if module_a_cache.exists():
        try:
            d = pd.read_csv(module_a_cache, parse_dates=["date"])
            d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None)
            d = d[(d["date"] >= start_date) & (d["date"] <= end_date)].copy()
            if not d.empty:
                n_days = int(d["date"].nunique())
                return n_days, float(n_days / 252.0), str(module_a_cache)
        except Exception as exc:
            report.line(f"WARNING reading module A cache for trade-day count: {exc.__class__.__name__}: {exc}")

    n_days = int(pd.to_datetime(source_data["date"]).dt.normalize().nunique())
    return n_days, float(n_days / 252.0), "backtester_source_data"


def _commission_drag_pct(e_gross: float, e_net: float) -> float:
    if not (np.isfinite(e_gross) and np.isfinite(e_net)) or e_gross <= 0:
        return float("nan")
    return float((e_gross - e_net) / e_gross)


def _weighted_metric_by_buckets(df: pd.DataFrame, value_col: str, weight_col: str, buckets: Iterable[str]) -> float:
    sub = df[df["vix_bucket"].astype(str).isin(list(buckets))].copy()
    if sub.empty:
        return float("nan")
    vals = pd.to_numeric(sub[value_col], errors="coerce")
    w = pd.to_numeric(sub[weight_col], errors="coerce")
    mask = np.isfinite(vals) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(vals[mask], weights=w[mask]))


def _pick_optimal_l3(l3_df: pd.DataFrame) -> Tuple[int, pd.Series]:
    # Pragmatic criterion: maximize Sharpe on tp50_sl_capped (always-close) among rows with >=100 trades and finite Sharpe.
    candidates = l3_df.copy()
    candidates = candidates[np.isfinite(candidates["sharpe_tp50sl_always_close"])]
    primary = candidates[candidates["n_trades"] >= 100]
    chosen = primary if not primary.empty else candidates
    if chosen.empty:
        return int(999), l3_df.iloc[-1]
    # Tie-breakers: higher annual pnl, lower max drawdown (less negative), more trades.
    chosen = chosen.sort_values(
        ["sharpe_tp50sl_always_close", "annual_pnl_tp50sl_always_close", "max_dd_tp50sl_always_close", "n_trades"],
        ascending=[False, False, False, False],
    )
    row = chosen.iloc[0]
    return int(row["l3"]), row


def _summary_table_for_target(expectancy_table: pd.DataFrame) -> pd.DataFrame:
    sub = expectancy_table[expectancy_table["scenario"] == TARGET_SCENARIO].copy().sort_values("vix_bucket")
    out = sub[
        [
            "vix_bucket",
            "n_trades",
            "win_rate",
            "avg_win_gross",
            "avg_loss_gross",
            "avg_credit_usd",
            "e_net_hold_model",
            "margin_hold_model",
        ]
    ].copy()
    out = out.rename(
        columns={
            "vix_bucket": "VIX Bucket",
            "n_trades": "N",
            "win_rate": "WR%",
            "avg_win_gross": "Avg Win",
            "avg_loss_gross": "Avg Loss",
            "avg_credit_usd": "Avg Credit",
            "e_net_hold_model": "E_net",
            "margin_hold_model": "Margin",
        }
    )
    return out


def _write_consolidated_report(
    report: Reporter,
    expectancy_table: pd.DataFrame,
    breakeven_df: pd.DataFrame,
    l3_df: pd.DataFrame,
    wing_proj_df: pd.DataFrame,
    total_years: float,
) -> None:
    target_rows = expectancy_table[expectancy_table["scenario"] == TARGET_SCENARIO].copy().sort_values("vix_bucket")
    table = _summary_table_for_target(expectancy_table)

    report.section("CONSOLIDATED SUMMARY")
    report.line("============================================================")
    report.line("EXPECTANCY BY VIX - STRATEGY ECONOMICS")
    report.line("============================================================")
    report.line()
    report.line(f"SCENARIO: {TARGET_SCENARIO} (live config equivalent approximation)")
    report.line(f"Commission model: Hold-to-expiry-like (${COMMISSION_WIN_OPEN_ONLY:.2f} win / ${COMMISSION_LOSS_ROUNDTRIP:.2f} loss)")
    report.line()

    # Pretty print compact table
    header = "VIX Bucket | N   | WR%   | Avg Win | Avg Loss | Avg Credit | E_net  | Margin"
    rule = "-----------+-----+-------+---------+----------+------------+--------+--------"
    report.line(header)
    report.line(rule)
    for _, r in table.iterrows():
        report.line(
            f"{str(r['VIX Bucket']):<9} | "
            f"{int(r['N']):>3} | "
            f"{_fmt_pct(float(r['WR%'])):>5} | "
            f"{_fmt_usd(float(r['Avg Win'])):>7} | "
            f"{_fmt_usd(float(r['Avg Loss'])):>8} | "
            f"{_fmt_usd(float(r['Avg Credit'])):>10} | "
            f"{_fmt_usd(float(r['E_net'])):>6} | "
            f"{_fmt_pct(float(r['Margin'])):>6}"
        )

    report.line()
    report.line("OPTIMAL L3 DETERMINATION")
    report.line("============================================================")
    row30 = l3_df[l3_df["l3"] == 30].iloc[0] if (l3_df["l3"] == 30).any() else None
    row25 = l3_df[l3_df["l3"] == 25].iloc[0] if (l3_df["l3"] == 25).any() else None
    row20 = l3_df[l3_df["l3"] == 20].iloc[0] if (l3_df["l3"] == 20).any() else None
    best_l3, best_row = _pick_optimal_l3(l3_df)

    def _l3_line(label: str, row: Optional[pd.Series]) -> None:
        if row is None:
            report.line(f"{label}: n/a")
            return
        report.line(
            f"{label:<18} E_annual={_fmt_usd(float(row['annual_pnl_tp50sl_always_close']))}, "
            f"Sharpe={_fmt_num(float(row['sharpe_tp50sl_always_close']))}, "
            f"N_trades={int(row['n_trades'])}, "
            f"MaxDD={_fmt_usd(float(row['max_dd_tp50sl_always_close']))}"
        )

    _l3_line("L3=30 (current):", row30)
    _l3_line("L3=25:", row25)
    _l3_line("L3=20:", row20)
    _l3_line("L3=optimal:", best_row)
    report.line()

    reco = f"Set L3 = {best_l3} based on highest Sharpe (tp50_sl_capped, always-close commissions) with minimum trade-count guard"
    report.line(f"RECOMMENDATION: {reco}")

    row_2025 = target_rows[target_rows["vix_bucket"] == "20-25"]
    if not row_2025.empty:
        e_2025 = float(row_2025.iloc[0]["e_net_hold_model"])
        credit_2025 = float(row_2025.iloc[0]["avg_credit_usd"])
        breach_2025 = float(row_2025.iloc[0]["breach_rate_any_short"])
        yes_no = "YES" if (np.isfinite(e_2025) and e_2025 > 0) else "NO"
        report.line(
            f"Key insight: Does extra credit in VIX 20-25 compensate for breach rate? {yes_no} "
            f"(AvgCredit={_fmt_usd(credit_2025)}, ShortBreach={_fmt_pct(breach_2025)}, E_net={_fmt_usd(e_2025)})"
        )

    report.line()
    report.line("BREAKEVEN ANALYSIS")
    report.line("============================================================")
    breakeven_target = breakeven_df[
        (breakeven_df["scenario"] == TARGET_SCENARIO) & (breakeven_df["commission_model"] == "hold_to_expiry_like")
    ].copy()
    breakeven_target["vix_bucket"] = pd.Categorical(breakeven_target["vix_bucket"], categories=VIX_BUCKET_LABELS, ordered=True)
    breakeven_target = breakeven_target.sort_values("vix_bucket")
    for _, r in breakeven_target.iterrows():
        report.line(
            f"VIX {r['vix_bucket']}: Actual WR={_fmt_pct(float(r['actual_wr']))} vs "
            f"Breakeven={_fmt_pct(float(r['breakeven_wr']))} -> Margin: {_fmt_pct(float(r['margin_of_safety']))} "
            f"[{r['status']}]"
        )

    report.line()
    report.line("COMMISSION IMPACT")
    report.line("============================================================")
    # Weighted by trade count within the target scenario
    target_rows = target_rows.copy()
    drag_vix_lt20 = _commission_drag_pct(
        _weighted_metric_by_buckets(target_rows, "e_gross", "n_trades", ["0-15", "15-20"]),
        _weighted_metric_by_buckets(target_rows, "e_net_hold_model", "n_trades", ["0-15", "15-20"]),
    )
    drag_vix_2025 = _commission_drag_pct(
        _weighted_metric_by_buckets(target_rows, "e_gross", "n_trades", ["20-25"]),
        _weighted_metric_by_buckets(target_rows, "e_net_hold_model", "n_trades", ["20-25"]),
    )
    report.line(f"At wing=2, commissions consume {_fmt_pct(drag_vix_lt20)} of gross expectancy in VIX<20")
    report.line(f"At wing=2, commissions consume {_fmt_pct(drag_vix_2025)} of gross expectancy in VIX 20-25")
    report.line("[Reminder: larger wings typically reduce commission drag as % of credit]")

    report.line()
    report.line("WING WIDTH PROJECTION")
    report.line("============================================================")
    if wing_proj_df.empty:
        report.line("Wing width projection unavailable (re-run failed or no data).")
    else:
        sub = wing_proj_df[wing_proj_df["vix_bucket"] == "20-25"].copy().sort_values("wing_width")
        if sub.empty:
            report.line("No VIX 20-25 projection rows available.")
        else:
            report.line("VIX 20-25 (tp50_sl_capped, hold-to-expiry-like commissions):")
            for _, r in sub.iterrows():
                report.line(
                    f"  wing={int(r['wing_width'])}: E_net={_fmt_usd(float(r['e_net_hold_model']))}, "
                    f"WR={_fmt_pct(float(r['win_rate']))}, Margin={_fmt_pct(float(r['margin_hold_model']))}"
                )
    report.line("============================================================")


def _collect_outputs_manifest(report: Reporter) -> None:
    report.line()
    report.line("Artifacts")
    report.line("---------")
    for name in [
        "expectancy_table.csv",
        "breakeven_analysis.csv",
        "chart_breakeven_margin.png",
        "l3_comparison.csv",
        "chart_l3_cumulative_pnl.png",
        "chart_l3_tradeoff.png",
        "credit_risk_tradeoff.csv",
        "chart_credit_vs_breach.png",
        "wing_width_projection.csv",
        "chart_wing_width_expectancy.png",
        "expectancy_report.txt",
    ]:
        report.line(f"- {OUTPUT_DIR / name}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _set_plot_style()
    report = Reporter()

    report.line("Expectancy by VIX Bucket - Strategy Economics")
    report.line(f"Repo root: {REPO_ROOT}")
    report.line(f"Output dir: {OUTPUT_DIR}")

    cfg = _load_config()
    cfg_wing2 = _cfg_with_wing(cfg, wing_width=2.0)
    source_data = _load_backtester_source_data(cfg_wing2, report)

    start_date = pd.to_datetime(source_data["date"]).min()
    end_date = pd.to_datetime(source_data["date"]).max()
    trade_days, total_years, day_count_source = _load_trade_day_count(start_date, end_date, source_data, report)
    report.section("Trade-Day Counting")
    report.line(f"Trading days in backtest period: {trade_days}")
    report.line(f"Total years (trading_days/252): {total_years:.4f}")
    report.line(f"Trade-day source: {day_count_source}")

    results_wing2 = _run_backtest_or_reuse(cfg_wing2, source_data, wing_width=2.0, report=report)
    df = _prepare_trade_frame(results_wing2, report=report)

    expectancy_table = _compute_expectancy_table(df, total_years=total_years, report=report)
    breakeven_df = _analysis_breakeven(expectancy_table, report=report)
    l3_df, _ = _analysis_l3_thresholds(df, total_years=total_years, report=report)
    _ = _analysis_credit_risk_tradeoff(expectancy_table, report=report)
    wing_proj_df = _analysis_wing_width_projection(cfg, source_data, total_years=total_years, report=report)

    _write_consolidated_report(
        report=report,
        expectancy_table=expectancy_table,
        breakeven_df=breakeven_df,
        l3_df=l3_df,
        wing_proj_df=wing_proj_df,
        total_years=total_years,
    )
    _collect_outputs_manifest(report)

    report_path = OUTPUT_DIR / "expectancy_report.txt"
    report.save(report_path)
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
