"""Synthetic 0DTE Iron Condor backtester using SPX OHLC + VIX daily data."""

from __future__ import annotations

import copy
import itertools
import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

from .bs_pricer import (
    XSP_MULTIPLIER,
    bs_delta,
    bs_price,
    estimate_ic_credit,
    find_strike_by_delta,
)

LOGGER = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252.0
TRADING_HOURS_PER_DAY = 6.5
MARKET_CLOSE_HOUR = 16.0
DEFAULT_RANDOM_SEED = 42

SCENARIO_COLUMNS = {
    "hold_to_expiry": "outcome_hold_to_expiry",
    "worst_case": "outcome_worst_case",
    "tp50_or_expiry": "outcome_tp50_or_expiry",
    "tp50_sl_capped": "outcome_tp50_sl_capped",
}

PRE_COMMISSION_COLUMNS = {
    "hold_to_expiry": "hold_to_expiry_pre_commission",
    "worst_case": "worst_case_pre_commission",
    "tp50_or_expiry": "tp50_or_expiry_pre_commission",
    "tp50_sl_capped": "tp50_sl_capped_pre_commission",
}

COMMISSION_COLUMNS = {
    "hold_to_expiry": "commission_hold_to_expiry",
    "worst_case": "commission_worst_case",
    "tp50_or_expiry": "commission_tp50_or_expiry",
    "tp50_sl_capped": "commission_tp50_sl_capped",
}


@dataclass
class TradeResult:
    """Single-day synthetic trade output."""

    date: pd.Timestamp
    spx_open: float
    spx_high: float
    spx_low: float
    spx_close: float
    vix: float
    short_put: float
    short_call: float
    long_put: float
    long_call: float
    wing_width: float
    entry_credit: float
    entry_credit_usd: float
    max_loss_usd: float
    settlement_pnl_usd: float
    put_breached: bool
    call_breached: bool
    put_wing_breached: bool
    call_wing_breached: bool
    outcome_hold_to_expiry: float
    outcome_worst_case: float
    outcome_tp50_or_expiry: float
    outcome_tp50_sl_capped: float
    risk_reward_ratio: float
    selection_method: str
    commission_usd: float
    skipped: bool
    skip_reason: str


def _cfg(config: Dict[str, Any], *keys: str, default: Any) -> Any:
    """Read nested config keys with a default fallback."""
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _extract_ohlc(frame: pd.DataFrame, symbol: str, prefix: str) -> pd.DataFrame:
    """Normalize Yahoo OHLC frame into prefixed flat columns."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=[f"{prefix}_open", f"{prefix}_high", f"{prefix}_low", f"{prefix}_close"])

    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        level0 = list(data.columns.get_level_values(0))
        level1 = list(data.columns.get_level_values(1))
        if symbol in level0:
            data = data[symbol]
        elif symbol in level1:
            data = data.xs(symbol, axis=1, level=1)
        else:
            data.columns = data.columns.get_level_values(-1)

    lower_cols = {c: str(c).lower() for c in data.columns}
    data = data.rename(columns=lower_cols)
    keep = ["open", "high", "low", "close"]
    for col in keep:
        if col not in data.columns:
            raise ValueError(f"Missing expected column '{col}' for {symbol}")

    out = data[keep].rename(columns={c: f"{prefix}_{c}" for c in keep})
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out


def _download_spx_vix(start_date: str, end_date: str) -> pd.DataFrame:
    """Download raw SPX/VIX daily OHLC from Yahoo Finance."""
    # Yahoo end date is exclusive. Add 1 day so requested end date is included.
    end_plus = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    spx_raw = yf.download(
        "^GSPC",
        start=start_date,
        end=end_plus,
        auto_adjust=False,
        progress=False,
        interval="1d",
        threads=False,
    )
    vix_raw = yf.download(
        "^VIX",
        start=start_date,
        end=end_plus,
        auto_adjust=False,
        progress=False,
        interval="1d",
        threads=False,
    )

    spx = _extract_ohlc(spx_raw, "^GSPC", "spx")
    vix = _extract_ohlc(vix_raw, "^VIX", "vix")

    merged = spx.join(vix, how="left")
    merged = merged.sort_index()
    merged["date"] = merged.index
    merged = merged.reset_index(drop=True)
    merged = merged[merged["date"].dt.dayofweek < 5].copy()
    return merged


def download_data(
    start_date: str = "2020-01-01",
    end_date: str = "2026-02-13",
    cache_file: str = "data/spx_vix_daily.csv",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download SPX (^GSPC) and VIX (^VIX) daily OHLC and cache to CSV.
    """
    cache_path = Path(cache_file)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    use_cache = False
    cached = pd.DataFrame()
    if cache_path.exists() and not force_refresh:
        cached = pd.read_csv(cache_path, parse_dates=["date"])
        if not cached.empty:
            min_cached = cached["date"].min().date()
            max_cached = cached["date"].max().date()
            start_dt = pd.to_datetime(start_date).date()
            end_dt = pd.to_datetime(end_date).date()
            required = {
                "date",
                "spx_open",
                "spx_high",
                "spx_low",
                "spx_close",
                "vix_open",
                "vix_high",
                "vix_low",
                "vix_close",
            }
            if required.issubset(set(cached.columns)) and min_cached <= start_dt <= max_cached and min_cached <= end_dt <= max_cached:
                use_cache = True

    if use_cache:
        data = cached.copy()
    else:
        data = _download_spx_vix(start_date, end_date)
        data.to_csv(cache_path, index=False)

    data = data.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)
    data = data[(data["date"] >= pd.to_datetime(start_date)) & (data["date"] <= pd.to_datetime(end_date))]
    data = data.sort_values("date").reset_index(drop=True)

    vix_cols = ["vix_open", "vix_high", "vix_low", "vix_close"]
    data[vix_cols] = data[vix_cols].ffill()

    invalid_hilo = data["spx_high"] < data["spx_low"]
    if invalid_hilo.any():
        data = data[~invalid_hilo].copy()

    data = data.dropna(subset=["spx_open", "spx_close"])
    missing_critical = data[["spx_open", "spx_close"]].isna().sum().sum()
    if missing_critical > 0:
        raise ValueError("Critical NaN values found in SPX open/close after cleaning")

    start_loaded = data["date"].min().date() if not data.empty else "N/A"
    end_loaded = data["date"].max().date() if not data.empty else "N/A"
    print(f"{len(data)} trading days loaded from {start_loaded} to {end_loaded}")
    return data.reset_index(drop=True)


def validate_data(data: pd.DataFrame) -> Dict[str, Any]:
    """Compute and print mandatory EDA validation summary."""
    if data.empty:
        summary = {
            "total_trading_days": 0,
            "date_range": ("N/A", "N/A"),
            "spx_range": (np.nan, np.nan),
            "vix_range": (np.nan, np.nan),
            "remaining_nans": 0,
            "high_lt_low_count": 0,
            "missing_vix_count": 0,
        }
        print("No data available after loading.")
        return summary

    high_lt_low_count = int((data["spx_high"] < data["spx_low"]).sum())
    missing_vix_count = int(data["vix_close"].isna().sum())
    summary = {
        "total_trading_days": int(len(data)),
        "date_range": (data["date"].min().date(), data["date"].max().date()),
        "spx_range": (float(data["spx_low"].min()), float(data["spx_high"].max())),
        "vix_range": (float(data["vix_low"].min(skipna=True)), float(data["vix_high"].max(skipna=True))),
        "remaining_nans": int(data.isna().sum().sum()),
        "high_lt_low_count": high_lt_low_count,
        "missing_vix_count": missing_vix_count,
    }

    print("\n📊 DATA SUMMARY")
    print(f"   Trading Days: {summary['total_trading_days']:,}")
    print(f"   Date Range: {summary['date_range'][0]} to {summary['date_range'][1]}")
    print(f"   SPX Range: {summary['spx_range'][0]:.2f} - {summary['spx_range'][1]:.2f}")
    print(f"   VIX Range: {summary['vix_range'][0]:.2f} - {summary['vix_range'][1]:.2f}")
    print(f"   Remaining NaNs: {summary['remaining_nans']:,}")
    print(f"   High < Low errors: {summary['high_lt_low_count']:,}")
    print(f"   Missing VIX rows: {summary['missing_vix_count']:,}")
    return summary


def compute_time_to_expiry(entry_hour: float) -> float:
    """
    Convert entry hour into fractional years to expiry.

    Formula:
        hours_remaining = max(0, 16 - entry_hour)
        T = hours_remaining / (252 * 6.5)
    """
    hours_remaining = max(0.0, MARKET_CLOSE_HOUR - float(entry_hour))
    return max(hours_remaining / (TRADING_DAYS_PER_YEAR * TRADING_HOURS_PER_DAY), 1e-8)


def _bs_price_vectorized(
    S: np.ndarray,
    K: np.ndarray,
    T: float,
    r: float,
    sigma: np.ndarray,
    option_type: str,
) -> np.ndarray:
    """Vectorized Black-Scholes pricing for arrays of strikes."""
    opt = option_type.lower().strip()
    if opt not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if opt == "call":
        intrinsic = np.maximum(0.0, S - K)
    else:
        intrinsic = np.maximum(0.0, K - S)

    prices = intrinsic.copy()
    if T <= 0:
        return prices

    valid = (
        np.isfinite(S)
        & np.isfinite(K)
        & np.isfinite(sigma)
        & (S > 0)
        & (K > 0)
        & (sigma > 0)
    )
    if not np.any(valid):
        return prices

    sqrt_t = np.sqrt(T)
    vol_term = sigma[valid] * sqrt_t
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S[valid] / K[valid]) + (r + 0.5 * sigma[valid] ** 2) * T) / vol_term
    d2 = d1 - vol_term

    if opt == "call":
        px = S[valid] * norm.cdf(d1) - K[valid] * np.exp(-r * T) * norm.cdf(d2)
    else:
        px = K[valid] * np.exp(-r * T) * norm.cdf(-d2) - S[valid] * norm.cdf(-d1)
    prices[valid] = np.maximum(px, 0.0)
    return prices


def _delta_to_strike_vectorized(
    S: np.ndarray,
    T: float,
    r: float,
    sigma: np.ndarray,
    target_delta: float,
    option_type: str,
    strike_step: float,
) -> np.ndarray:
    """Vectorized analytic strike inversion for a target Black-Scholes delta."""
    opt = option_type.lower().strip()
    if strike_step <= 0:
        raise ValueError("strike_step must be positive")
    if opt not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    S = np.asarray(S, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    strikes = np.full_like(S, np.nan, dtype=float)
    valid = np.isfinite(S) & np.isfinite(sigma) & (S > 0) & (sigma > 0) & np.isfinite(T) & (T > 0)
    if not np.any(valid):
        return strikes

    sqrt_t = np.sqrt(T)
    if opt == "call":
        q = np.clip(target_delta, 1e-6, 1 - 1e-6)
    else:
        q = np.clip(1.0 + target_delta, 1e-6, 1 - 1e-6)
    d1_target = norm.ppf(q)

    sig = sigma[valid]
    log_term = d1_target * sig * sqrt_t - (r + 0.5 * sig**2) * T
    k = S[valid] / np.exp(log_term)
    k = np.round(k / strike_step) * strike_step

    if opt == "call":
        k = np.where(k <= S[valid], np.ceil((S[valid] + strike_step) / strike_step) * strike_step, k)
    else:
        k = np.where(k >= S[valid], np.floor((S[valid] - strike_step) / strike_step) * strike_step, k)

    strikes[valid] = np.maximum(strike_step, k)
    return strikes


def _settlement_pnl_per_share(
    close: np.ndarray,
    short_put: np.ndarray,
    long_put: np.ndarray,
    short_call: np.ndarray,
    long_call: np.ndarray,
    put_credit: np.ndarray,
    call_credit: np.ndarray,
) -> np.ndarray:
    """
    Settlement PnL per-share using spread payoff decomposition.

    put_spread_value = max(0, short_put-close) - max(0, long_put-close)
    call_spread_value = max(0, close-short_call) - max(0, close-long_call)
    pnl_share = put_credit - put_spread_value + call_credit - call_spread_value
    """
    put_spread_value = np.maximum(0.0, short_put - close) - np.maximum(0.0, long_put - close)
    call_spread_value = np.maximum(0.0, close - short_call) - np.maximum(0.0, close - long_call)
    return (put_credit - put_spread_value) + (call_credit - call_spread_value)


def _commission_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve commission model settings from v2 or legacy config layouts."""
    comm_cfg = _cfg(config, "commissions", default={})
    if isinstance(comm_cfg, dict) and comm_cfg:
        pricing_plan = str(comm_cfg.get("pricing_plan", "fixed")).lower()
        plan_cfg = comm_cfg.get(pricing_plan, {})
        if not isinstance(plan_cfg, dict):
            plan_cfg = {}

        per_contract = float(plan_cfg.get("per_contract", comm_cfg.get("per_contract", 0.65)))
        legs_per_ic = int(comm_cfg.get("legs_per_ic", 4))
        model = str(comm_cfg.get("model", "conditional")).lower()
        if model not in {"conditional", "flat"}:
            model = "conditional"
        flat_amount = float(comm_cfg.get("flat_amount", 5.00))

        open_commission = per_contract * legs_per_ic
        round_trip_commission = open_commission * 2.0
        commission_per_trade = flat_amount if model == "flat" else round_trip_commission
        return {
            "commission_model": model,
            "pricing_plan": pricing_plan,
            "per_contract_commission": per_contract,
            "legs_per_ic": legs_per_ic,
            "open_commission": open_commission,
            "round_trip_commission": round_trip_commission,
            "flat_commission_amount": flat_amount,
            # Backward-compatible aggregate value for existing callers.
            "commission_per_trade": commission_per_trade,
        }

    # Legacy fallback path.
    legacy_commission = float(_cfg(config, "costs", "commission_per_trade", default=5.0))
    return {
        "commission_model": "flat",
        "pricing_plan": "legacy",
        "per_contract_commission": legacy_commission / 8.0,
        "legs_per_ic": 4,
        "open_commission": legacy_commission / 2.0,
        "round_trip_commission": legacy_commission,
        "flat_commission_amount": legacy_commission,
        "commission_per_trade": legacy_commission,
    }


def _backtest_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested backtest config into a runtime parameter map."""
    commission = _commission_params(config)

    def finite_or_nan(value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return np.nan
        return parsed if np.isfinite(parsed) else np.nan

    start_date = _cfg(config, "dates", "start", default=_cfg(config, "data", "start_date", default="2020-01-01"))
    end_date = _cfg(config, "dates", "end", default=_cfg(config, "data", "end_date", default="2026-02-13"))
    target_delta = _cfg(config, "default", "delta_target", default=_cfg(config, "strike_selection", "target_delta", default=0.10))
    wing_width = _cfg(config, "default", "wing_width", default=_cfg(config, "structure", "wing_width", default=2.0))
    min_credit = _cfg(config, "default", "min_credit", default=_cfg(config, "credit_filters", "min_credit", default=0.20))
    bid_ask_haircut = _cfg(config, "default", "bid_ask_haircut", default=_cfg(config, "costs", "bid_ask_haircut", default=0.25))
    take_profit_pct = _cfg(config, "default", "tp_pct", default=_cfg(config, "exit_rules", "take_profit_pct", default=0.50))
    stop_loss_mult = _cfg(config, "default", "sl_mult", default=_cfg(config, "exit_rules", "stop_loss_mult", default=3.0))
    entry_hour = _cfg(config, "default", "entry_hour", default=_cfg(config, "timing", "entry_hour", default=10.0))

    max_credit_to_wing_ratio = _cfg(config, "sweep_filters", "max_credit_to_wing_ratio", default=np.nan)
    max_margin_per_contract = _cfg(config, "sweep_filters", "max_margin_per_contract", default=np.nan)

    return {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "cache_file": _cfg(config, "data", "cache_file", default="data/spx_vix_daily.csv"),
        "min_vix": float(_cfg(config, "entry_filters", "min_vix", default=14.0)),
        "max_vix": float(_cfg(config, "entry_filters", "max_vix", default=35.0)),
        "target_delta": float(target_delta),
        "strike_step": float(_cfg(config, "strike_selection", "strike_step", default=1.0)),
        "wing_width": float(wing_width),
        "min_credit": float(min_credit),
        "max_risk_reward": float(_cfg(config, "credit_filters", "max_risk_reward", default=6.0)),
        "bid_ask_haircut": float(bid_ask_haircut),
        "entry_hour": float(entry_hour),
        "risk_free_rate": float(_cfg(config, "timing", "risk_free_rate", default=0.05)),
        "vix_source": str(_cfg(config, "timing", "vix_source", default="previous_close")).lower(),
        "take_profit_pct": float(take_profit_pct),
        "stop_loss_mult": float(stop_loss_mult),
        "starting_capital": float(_cfg(config, "capital", "starting_capital", default=1580.0)),
        "xsp_multiplier": float(_cfg(config, "default", "multiplier", default=XSP_MULTIPLIER)),
        "max_credit_to_wing_ratio": finite_or_nan(max_credit_to_wing_ratio),
        "max_margin_per_contract": finite_or_nan(max_margin_per_contract),
        **commission,
    }


def _simulate_dataframe(data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Run the core vectorized daily simulation engine."""
    params = _backtest_params(config)
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    frame = frame.sort_values("date").reset_index(drop=True)

    S = frame["spx_open"].to_numpy(dtype=float)
    high = frame["spx_high"].to_numpy(dtype=float)
    low = frame["spx_low"].to_numpy(dtype=float)
    close = frame["spx_close"].to_numpy(dtype=float)

    if params["vix_source"] == "vix_open":
        entry_vix = frame["vix_open"].to_numpy(dtype=float)
    else:
        # Default: previous day close to avoid same-day look-ahead.
        entry_vix = frame["vix_close"].shift(1).to_numpy(dtype=float)

    sigma = entry_vix / 100.0
    T = compute_time_to_expiry(params["entry_hour"])

    target = abs(params["target_delta"])
    short_put = _delta_to_strike_vectorized(S, T, params["risk_free_rate"], sigma, -target, "put", params["strike_step"])
    short_call = _delta_to_strike_vectorized(S, T, params["risk_free_rate"], sigma, target, "call", params["strike_step"])
    long_put = short_put - params["wing_width"]
    long_call = short_call + params["wing_width"]

    short_put_price = _bs_price_vectorized(S, short_put, T, params["risk_free_rate"], sigma, "put")
    long_put_price = _bs_price_vectorized(S, long_put, T, params["risk_free_rate"], sigma, "put")
    short_call_price = _bs_price_vectorized(S, short_call, T, params["risk_free_rate"], sigma, "call")
    long_call_price = _bs_price_vectorized(S, long_call, T, params["risk_free_rate"], sigma, "call")

    put_credit_theoretical = short_put_price - long_put_price
    call_credit_theoretical = short_call_price - long_call_price
    theoretical_credit = put_credit_theoretical + call_credit_theoretical

    haircut_scale = 1.0 - params["bid_ask_haircut"]
    put_credit = put_credit_theoretical * haircut_scale
    call_credit = call_credit_theoretical * haircut_scale
    entry_credit = theoretical_credit * haircut_scale
    entry_credit_usd = entry_credit * XSP_MULTIPLIER
    max_loss_usd = (params["wing_width"] - entry_credit) * XSP_MULTIPLIER
    max_loss_usd = np.maximum(max_loss_usd, 0.0)

    risk_reward_ratio = np.divide(
        max_loss_usd,
        entry_credit_usd,
        out=np.full_like(max_loss_usd, np.inf, dtype=float),
        where=entry_credit_usd > 0,
    )
    breakeven_winrate = np.divide(
        max_loss_usd,
        max_loss_usd + entry_credit_usd,
        out=np.full_like(max_loss_usd, np.nan, dtype=float),
        where=(max_loss_usd + entry_credit_usd) > 0,
    )

    put_breached = np.isfinite(short_put) & (low <= short_put)
    call_breached = np.isfinite(short_call) & (high >= short_call)
    put_wing_breached = np.isfinite(long_put) & (low <= long_put)
    call_wing_breached = np.isfinite(long_call) & (high >= long_call)

    settlement_share = _settlement_pnl_per_share(close, short_put, long_put, short_call, long_call, put_credit, call_credit)
    settlement_pre_commission = settlement_share * XSP_MULTIPLIER

    touched_short = put_breached | call_breached
    worst_pre_commission = np.where(touched_short, -max_loss_usd, entry_credit_usd)

    untouched = (~touched_short) & np.isfinite(settlement_pre_commission)
    tp_cap_usd = params["take_profit_pct"] * entry_credit_usd
    tp_pre_commission = np.where(
        untouched & (settlement_pre_commission > 0),
        np.minimum(settlement_pre_commission, tp_cap_usd),
        settlement_pre_commission,
    )
    sl_cap_usd = params["stop_loss_mult"] * entry_credit_usd
    tp50_sl_capped_pre_commission = np.where(
        touched_short,
        np.maximum(settlement_pre_commission, -sl_cap_usd),
        np.where(
            settlement_pre_commission > 0,
            np.minimum(settlement_pre_commission, tp_cap_usd),
            settlement_pre_commission,
        ),
    )

    reason = np.full(len(frame), "", dtype=object)

    def set_reason(mask: np.ndarray, text: str) -> None:
        reason[(reason == "") & mask] = text

    set_reason(~np.isfinite(entry_vix), "vix_unavailable")
    set_reason((entry_vix < params["min_vix"]) | (entry_vix > params["max_vix"]), "vix_out_of_range")
    set_reason(~np.isfinite(short_put) | ~np.isfinite(short_call), "invalid_strikes")
    set_reason(~np.isfinite(entry_credit) | (entry_credit <= 0) | (entry_credit >= params["wing_width"]), "invalid_credit")
    set_reason(entry_credit < params["min_credit"], "credit_below_min")
    set_reason(~np.isfinite(risk_reward_ratio) | (risk_reward_ratio > params["max_risk_reward"]), "risk_reward_too_high")
    set_reason(high < low, "invalid_ohlc")

    skipped = reason != ""
    open_commission = float(params["open_commission"])
    round_trip_commission = float(params["round_trip_commission"])

    if params["commission_model"] == "flat":
        flat_commission = float(params["flat_commission_amount"])
        commission_hold = np.full(len(frame), flat_commission, dtype=float)
        commission_worst = np.full(len(frame), flat_commission, dtype=float)
        commission_tp = np.full(len(frame), flat_commission, dtype=float)
        commission_tp_sl_capped = np.full(len(frame), flat_commission, dtype=float)
    else:
        # Hold-to-expiry wins (both shorts OTM at settlement) can expire worthless:
        # only opening commission is paid. All other outcomes are modeled as active
        # closes and pay round-trip commission.
        hold_expired_worthless = np.isfinite(short_put) & np.isfinite(short_call) & (close >= short_put) & (close <= short_call)
        commission_hold = np.where(hold_expired_worthless, open_commission, round_trip_commission)
        commission_worst = np.full(len(frame), round_trip_commission, dtype=float)
        commission_tp = np.full(len(frame), round_trip_commission, dtype=float)
        commission_tp_sl_capped = np.full(len(frame), round_trip_commission, dtype=float)

    commission_hold = np.where(skipped, 0.0, commission_hold)
    commission_worst = np.where(skipped, 0.0, commission_worst)
    commission_tp = np.where(skipped, 0.0, commission_tp)
    commission_tp_sl_capped = np.where(skipped, 0.0, commission_tp_sl_capped)

    outcome_hold = np.where(skipped, np.nan, settlement_pre_commission - commission_hold)
    outcome_worst = np.where(skipped, np.nan, worst_pre_commission - commission_worst)
    outcome_tp = np.where(skipped, np.nan, tp_pre_commission - commission_tp)
    outcome_tp_sl_capped = np.where(skipped, np.nan, tp50_sl_capped_pre_commission - commission_tp_sl_capped)

    result = pd.DataFrame(
        {
            "date": frame["date"],
            "spx_open": S,
            "spx_high": high,
            "spx_low": low,
            "spx_close": close,
            "vix": entry_vix,
            "short_put": short_put,
            "short_call": short_call,
            "long_put": long_put,
            "long_call": long_call,
            "wing_width": params["wing_width"],
            "entry_credit": entry_credit,
            "entry_credit_usd": entry_credit_usd,
            "theoretical_credit": theoretical_credit,
            "max_loss_usd": max_loss_usd,
            "risk_reward_ratio": risk_reward_ratio,
            "trade_breakeven_winrate": breakeven_winrate,
            "short_put_price": short_put_price,
            "short_call_price": short_call_price,
            "long_put_price": long_put_price,
            "long_call_price": long_call_price,
            "put_breached": put_breached,
            "call_breached": call_breached,
            "put_wing_breached": put_wing_breached,
            "call_wing_breached": call_wing_breached,
            "hold_to_expiry_pre_commission": settlement_pre_commission,
            "worst_case_pre_commission": worst_pre_commission,
            "tp50_or_expiry_pre_commission": tp_pre_commission,
            "tp50_sl_capped_pre_commission": tp50_sl_capped_pre_commission,
            "settlement_pnl_usd": outcome_hold,
            "outcome_hold_to_expiry": outcome_hold,
            "outcome_worst_case": outcome_worst,
            "outcome_tp50_or_expiry": outcome_tp,
            "outcome_tp50_sl_capped": outcome_tp_sl_capped,
            "commission_open_usd": np.where(skipped, 0.0, open_commission),
            "commission_round_trip_usd": np.where(skipped, 0.0, round_trip_commission),
            "commission_hold_to_expiry": commission_hold,
            "commission_worst_case": commission_worst,
            "commission_tp50_or_expiry": commission_tp,
            "commission_tp50_sl_capped": commission_tp_sl_capped,
            "selection_method": np.where(skipped, "", "DELTA"),
            # Legacy commission field kept for compatibility with existing consumers.
            "commission_usd": commission_hold,
            "skipped": skipped,
            "skip_reason": reason,
        }
    )
    return result


def simulate_day(row: pd.Series, config: Dict[str, Any]) -> TradeResult:
    """Simulate one day and return a TradeResult dataclass."""
    row_df = pd.DataFrame([row])
    if "date" not in row_df.columns:
        if isinstance(row.name, (pd.Timestamp, str)):
            row_df["date"] = pd.to_datetime(row.name)
        else:
            raise ValueError("Row must include a 'date' column or timestamp index")
    result = _simulate_dataframe(row_df, config).iloc[0].to_dict()
    allowed = {item.name for item in fields(TradeResult)}
    filtered = {key: result[key] for key in allowed if key in result}
    return TradeResult(**filtered)


def run_backtest(config: Dict[str, Any], data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Run synthetic backtest over all available trading days."""
    params = _backtest_params(config)
    if data is None:
        data = download_data(
            start_date=params["start_date"],
            end_date=params["end_date"],
            cache_file=params["cache_file"],
        )
    else:
        data = data.copy()
        data["date"] = pd.to_datetime(data["date"])
        data = data[
            (data["date"] >= pd.to_datetime(params["start_date"]))
            & (data["date"] <= pd.to_datetime(params["end_date"]))
        ].copy()
    return _simulate_dataframe(data, config)


def breakeven_winrate(
    credit: float,
    wing_width: float,
    commission: float,
    tp_pct: float = 0.50,
    sl_mult: float = 3.0,
) -> Dict[str, float]:
    """
    Compute break-even win rate under TP/SL and hold-to-expiry assumptions.
    """
    if not np.isfinite(credit) or credit <= 0:
        return {"with_tp_sl": 1.0, "hold_to_expiry": 1.0}

    # TP/SL scenario from requested formula.
    avg_win_tp = credit * tp_pct * XSP_MULTIPLIER - commission
    avg_loss_tp = credit * sl_mult * XSP_MULTIPLIER + commission
    if avg_win_tp <= 0:
        be_tp = 1.0
    else:
        be_tp = avg_loss_tp / (avg_win_tp + avg_loss_tp)

    # Hold-to-expiry bounded loss/profit formulation.
    avg_win_hold = credit * XSP_MULTIPLIER - commission
    avg_loss_hold = max(0.0, (wing_width - credit) * XSP_MULTIPLIER + commission)
    if avg_win_hold <= 0:
        be_hold = 1.0
    else:
        be_hold = avg_loss_hold / (avg_win_hold + avg_loss_hold)

    return {"with_tp_sl": float(be_tp), "hold_to_expiry": float(be_hold)}


def _max_consecutive_losses(pnl: pd.Series) -> int:
    """Return longest consecutive negative-PnL streak."""
    max_streak = 0
    current = 0
    for value in pnl:
        if value < 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return int(max_streak)


def _compute_drawdown(daily_pnl: pd.Series, starting_capital: float) -> pd.Series:
    """Compute drawdown series from daily PnL and starting capital."""
    equity = starting_capital + daily_pnl.cumsum()
    running_peak = equity.cummax()
    return equity - running_peak


def _estimate_risk_of_ruin(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    capital: float,
    avg_bet: float,
) -> float:
    """
    Simplified risk-of-ruin estimate based on requested Kelly-adjacent heuristic.
    """
    if capital <= 0 or avg_bet <= 0 or avg_win <= 0 or avg_loss <= 0:
        return 1.0

    # Kelly fraction proxy.
    kelly_f = (win_rate * avg_win - (1.0 - win_rate) * avg_loss) / avg_win
    if kelly_f <= 0:
        return 1.0

    breakeven = avg_loss / (avg_loss + avg_win)
    if win_rate <= breakeven:
        return 1.0

    if win_rate <= 0:
        return 1.0

    ratio = (1.0 - win_rate) / win_rate
    if ratio <= 0:
        return 0.0
    ruin = ratio ** (capital / avg_bet)
    return float(np.clip(ruin, 0.0, 1.0))


def compute_metrics(results: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Compute complete metrics by scenario."""
    params = _backtest_params(config)
    starting_capital = params["starting_capital"]
    commission = params["commission_per_trade"]
    wing_width = params["wing_width"]
    tp_pct = params["take_profit_pct"]
    sl_mult = params["stop_loss_mult"]

    total_days = len(results)
    total_skipped = int(results["skipped"].sum()) if "skipped" in results else 0
    scenario_metrics: Dict[str, Dict[str, float]] = {}

    for scenario_name, outcome_col in SCENARIO_COLUMNS.items():
        pre_col = PRE_COMMISSION_COLUMNS[scenario_name]
        tradable = results[(~results["skipped"]) & results[outcome_col].notna()].copy()
        pnl = tradable[outcome_col].astype(float) if not tradable.empty else pd.Series(dtype=float)
        pnl_pre = tradable[pre_col].astype(float) if not tradable.empty else pd.Series(dtype=float)
        daily_pnl = results[outcome_col].fillna(0.0).astype(float)
        commission_col = COMMISSION_COLUMNS.get(scenario_name, "")
        if commission_col and commission_col in tradable.columns:
            scenario_commission = tradable[commission_col].astype(float)
        else:
            scenario_commission = pd.Series(np.full(len(tradable), commission, dtype=float), index=tradable.index)

        total_trades = int(len(tradable))
        win_rate = float((pnl > 0).mean()) if total_trades else 0.0
        avg_win = float(pnl[pnl > 0].mean()) if (pnl > 0).any() else 0.0
        avg_loss = float(pnl[pnl < 0].mean()) if (pnl < 0).any() else 0.0

        gross_wins = float(pnl[pnl > 0].sum()) if total_trades else 0.0
        gross_losses = float(-pnl[pnl < 0].sum()) if total_trades else 0.0
        if gross_losses > 0:
            profit_factor = gross_wins / gross_losses
        elif gross_wins > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        total_pnl = float(pnl.sum()) if total_trades else 0.0
        avg_pnl = float(pnl.mean()) if total_trades else 0.0
        median_pnl = float(pnl.median()) if total_trades else 0.0
        std_pnl = float(pnl.std(ddof=0)) if total_trades else 0.0

        drawdown = _compute_drawdown(daily_pnl, starting_capital)
        max_dd_usd = float(drawdown.min()) if not drawdown.empty else 0.0
        max_dd_pct = (max_dd_usd / starting_capital * 100.0) if starting_capital > 0 else np.nan

        max_consec_losses = _max_consecutive_losses(pnl) if total_trades else 0
        worst_day = float(pnl.min()) if total_trades else 0.0
        if total_trades:
            var_95 = float(pnl.quantile(0.05))
            cvar_95 = float(pnl[pnl <= var_95].mean()) if (pnl <= var_95).any() else 0.0
        else:
            cvar_95 = 0.0

        std_daily = float(daily_pnl.std(ddof=0))
        if std_daily > 0:
            sharpe_daily = float((daily_pnl.mean() / std_daily) * np.sqrt(TRADING_DAYS_PER_YEAR))
        else:
            sharpe_daily = 0.0

        years = max(total_days / TRADING_DAYS_PER_YEAR, 1.0 / TRADING_DAYS_PER_YEAR)
        ending_capital = float(starting_capital + daily_pnl.sum())
        total_return_pct = ((ending_capital / starting_capital) - 1.0) * 100.0 if starting_capital > 0 else np.nan
        if starting_capital > 0 and ending_capital > 0:
            annualized_return_pct = ((ending_capital / starting_capital) ** (1.0 / years) - 1.0) * 100.0
        else:
            annualized_return_pct = -100.0

        if max_dd_usd < 0:
            calmar = annualized_return_pct / abs(max_dd_pct) if abs(max_dd_pct) > 0 else np.nan
        else:
            calmar = np.nan

        avg_max_loss = float(tradable["max_loss_usd"].mean()) if total_trades else np.nan
        edge_per_dollar = (avg_pnl / avg_max_loss) if total_trades and avg_max_loss and avg_max_loss > 0 else np.nan

        avg_credit = float(tradable["entry_credit"].mean()) if total_trades else np.nan
        be = breakeven_winrate(avg_credit, wing_width, commission, tp_pct, sl_mult)
        scenario_be = be["hold_to_expiry"] if scenario_name == "hold_to_expiry" else be["with_tp_sl"]
        winrate_margin = win_rate - scenario_be if np.isfinite(scenario_be) else np.nan

        total_commissions = float(scenario_commission.sum()) if total_trades else 0.0
        if params["commission_model"] == "conditional" and total_trades:
            total_open_commissions = float(total_trades * params["open_commission"])
            close_count = int((scenario_commission > (params["open_commission"] + 1e-9)).sum())
            total_close_commissions = float(close_count * params["open_commission"])
        else:
            total_open_commissions = 0.0
            total_close_commissions = 0.0
            close_count = 0

        gross_pnl = float(pnl_pre.sum()) if total_trades else 0.0
        commission_pct = (total_commissions / abs(gross_pnl)) if gross_pnl != 0 else np.inf
        avg_commission_per_trade = (total_commissions / total_trades) if total_trades else 0.0

        risk_of_ruin = _estimate_risk_of_ruin(
            win_rate=win_rate,
            avg_win=max(avg_win, 0.0),
            avg_loss=abs(min(avg_loss, 0.0)),
            capital=starting_capital,
            avg_bet=float(avg_max_loss) if np.isfinite(avg_max_loss) else 0.0,
        )

        scenario_metrics[scenario_name] = {
            "total_trades": total_trades,
            "total_skipped": total_skipped,
            "skip_rate": (total_skipped / total_days) if total_days else np.nan,
            "win_rate": win_rate,
            "avg_win_usd": avg_win,
            "avg_loss_usd": avg_loss,
            "profit_factor": profit_factor,
            "total_pnl_usd": total_pnl,
            "avg_pnl_per_trade_usd": avg_pnl,
            "median_pnl_per_trade_usd": median_pnl,
            "std_pnl_per_trade_usd": std_pnl,
            "max_drawdown_usd": max_dd_usd,
            "max_drawdown_pct": max_dd_pct,
            "max_consecutive_losses": max_consec_losses,
            "worst_day_usd": worst_day,
            "cvar_95_usd": cvar_95,
            "sharpe_daily": sharpe_daily,
            "calmar": calmar,
            "edge_per_dollar_risked": edge_per_dollar,
            "breakeven_winrate": scenario_be,
            "actual_winrate": win_rate,
            "winrate_margin": winrate_margin,
            "total_commissions_usd": total_commissions,
            "total_open_commissions_usd": total_open_commissions,
            "total_close_commissions_usd": total_close_commissions,
            "active_closes": close_count,
            "avg_commission_per_trade_usd": avg_commission_per_trade,
            "commission_pct_of_gross": commission_pct,
            "pnl_before_commissions": gross_pnl,
            "pnl_after_commissions": total_pnl,
            "starting_capital": starting_capital,
            "ending_capital": ending_capital,
            "total_return_pct": total_return_pct,
            "annualized_return_pct": annualized_return_pct,
            "risk_of_ruin_estimate": risk_of_ruin,
        }

    return scenario_metrics


def metrics_to_dataframe(metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Convert nested metrics dict to a display-friendly dataframe."""
    return pd.DataFrame(metrics).T


def run_parameter_sweep(
    base_config: Dict[str, Any],
    param_grid: Dict[str, List[Any]],
    data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Run parameter sweep across all valid combinations.
    """
    params_base = _backtest_params(base_config)
    if data is None:
        data = download_data(
            start_date=params_base["start_date"],
            end_date=params_base["end_date"],
            cache_file=params_base["cache_file"],
        )

    if not param_grid:
        return pd.DataFrame()

    param_names = list(param_grid.keys())
    total_combos = int(np.prod([len(param_grid[name]) for name in param_names]))
    print(f"Sweep combinations (raw grid): {total_combos:,}")
    rows: List[Dict[str, Any]] = []

    for combo in itertools.product(*(param_grid[name] for name in param_names)):
        combo_params = dict(zip(param_names, combo))

        wing = float(combo_params.get("wing_width", params_base["wing_width"]))
        min_credit = float(combo_params.get("min_credit", params_base["min_credit"]))
        tp_pct = float(combo_params.get("tp_pct", combo_params.get("take_profit_pct", params_base["take_profit_pct"])))
        sl_mult = float(combo_params.get("sl_mult", combo_params.get("stop_loss_mult", params_base["stop_loss_mult"])))
        commission = params_base["commission_per_trade"]

        row_base = dict(combo_params)

        # Invalid combo 1: impossible min credit relative to wing.
        if min_credit >= wing:
            row_base["combo_skipped"] = True
            row_base["combo_skip_reason"] = "min_credit_exceeds_wing_width"
            rows.append(row_base)
            continue

        # Invalid combo 2: user-defined max credit/wing feasibility filter.
        ratio_limit = params_base.get("max_credit_to_wing_ratio", np.nan)
        if np.isfinite(ratio_limit) and (min_credit > (wing * ratio_limit)):
            row_base["combo_skipped"] = True
            row_base["combo_skip_reason"] = "min_credit_exceeds_wing_ratio_limit"
            rows.append(row_base)
            continue

        # Invalid combo 3: account-level margin cap filter.
        margin_limit = params_base.get("max_margin_per_contract", np.nan)
        if np.isfinite(margin_limit) and (wing * XSP_MULTIPLIER > margin_limit):
            row_base["combo_skipped"] = True
            row_base["combo_skip_reason"] = "margin_above_limit"
            rows.append(row_base)
            continue

        # Invalid combo 4: structurally impossible required win rate.
        be_tp = breakeven_winrate(min_credit, wing, commission, tp_pct, sl_mult)["with_tp_sl"]
        if be_tp > 0.95:
            row_base["combo_skipped"] = True
            row_base["combo_skip_reason"] = "breakeven_winrate_gt_95pct"
            rows.append(row_base)
            continue

        cfg = copy.deepcopy(base_config)
        if "wing_width" in combo_params:
            cfg.setdefault("default", {})["wing_width"] = float(combo_params["wing_width"])
        if "delta_target" in combo_params:
            cfg.setdefault("default", {})["delta_target"] = float(combo_params["delta_target"])
        if "target_delta" in combo_params:
            cfg.setdefault("default", {})["delta_target"] = float(combo_params["target_delta"])
        if "min_credit" in combo_params:
            cfg.setdefault("default", {})["min_credit"] = float(combo_params["min_credit"])
        if "bid_ask_haircut" in combo_params:
            cfg.setdefault("default", {})["bid_ask_haircut"] = float(combo_params["bid_ask_haircut"])
        if "tp_pct" in combo_params:
            cfg.setdefault("default", {})["tp_pct"] = float(combo_params["tp_pct"])
        if "take_profit_pct" in combo_params:
            cfg.setdefault("default", {})["tp_pct"] = float(combo_params["take_profit_pct"])
        if "sl_mult" in combo_params:
            cfg.setdefault("default", {})["sl_mult"] = float(combo_params["sl_mult"])
        if "stop_loss_mult" in combo_params:
            cfg.setdefault("default", {})["sl_mult"] = float(combo_params["stop_loss_mult"])

        results = run_backtest(cfg, data=data)
        metrics = compute_metrics(results, cfg)

        out = dict(combo_params)
        out["combo_skipped"] = False
        out["combo_skip_reason"] = ""
        for scenario_name, scenario_values in metrics.items():
            for key, value in scenario_values.items():
                out[f"{scenario_name}_{key}"] = value
        rows.append(out)

    sweep_df = pd.DataFrame(rows)
    sweep_metric_col = "tp50_sl_capped_total_pnl_usd"
    if sweep_metric_col not in sweep_df.columns:
        sweep_metric_col = "tp50_or_expiry_total_pnl_usd"
    if not sweep_df.empty and sweep_metric_col in sweep_df.columns:
        sweep_df = sweep_df.sort_values(sweep_metric_col, ascending=False).reset_index(drop=True)

    if not sweep_df.empty and "combo_skipped" in sweep_df.columns:
        valid = sweep_df[sweep_df["combo_skipped"] == False]  # noqa: E712
        if not valid.empty and sweep_metric_col in valid.columns:
            print("\nTop 10 configurations:")
            print(valid.head(10).to_string(index=False))
            print("\nBottom 10 configurations:")
            print(valid.tail(10).to_string(index=False))

    return sweep_df


def test_bs_pricer() -> bool:
    """Self-check suite for pricing primitives."""
    S = 100.0
    K = 100.0
    T = 0.00366
    r = 0.0
    sigma = 0.20

    # 1) ATM call approx (Brenner-Subrahmanyam): C ≈ S*sigma*sqrt(T)/sqrt(2π)
    call_px = bs_price(S, K, T, r, sigma, "call")
    approx = S * sigma * np.sqrt(T) / np.sqrt(2.0 * np.pi)
    assert abs(call_px - approx) / approx < 0.20

    # 2) Put-call parity: C - P = S - K*exp(-rT)
    put_px = bs_price(S, K, T, r, sigma, "put")
    lhs = call_px - put_px
    rhs = S - K * np.exp(-r * T)
    assert abs(lhs - rhs) < 1e-6

    # 3) ATM call delta around 0.5
    atm_delta = bs_delta(S, K, T, r, sigma, "call")
    assert 0.45 <= atm_delta <= 0.55

    # 4) Deep OTM put delta near zero
    deep_otm_put_delta = bs_delta(S, 70.0, T, r, sigma, "put")
    assert -0.05 <= deep_otm_put_delta <= 0.0

    # 5) OTM IC should have positive credit
    ic_small_t = estimate_ic_credit(
        S=500.0,
        short_put=490.0,
        short_call=510.0,
        wing_width=2.0,
        T=0.00366,
        r=0.05,
        sigma=0.20,
        bid_ask_haircut=0.25,
    )
    assert ic_small_t["net_credit"] > 0

    # 6) Credit for tiny 0DTE T should be below full-day T
    ic_full_day = estimate_ic_credit(
        S=500.0,
        short_put=490.0,
        short_call=510.0,
        wing_width=2.0,
        T=1.0 / 252.0,
        r=0.05,
        sigma=0.20,
        bid_ask_haircut=0.25,
    )
    assert ic_small_t["net_credit"] < ic_full_day["net_credit"]

    # Explicit strike finder smoke check.
    k_put = find_strike_by_delta(500.0, T, r, sigma, -0.10, "put", strike_step=1.0)
    k_call = find_strike_by_delta(500.0, T, r, sigma, 0.10, "call", strike_step=1.0)
    assert k_put < 500.0 < k_call
    return True


def test_settlement_pnl() -> bool:
    """Validate settlement payoff boundaries."""
    short_put = 99.0
    long_put = 97.0
    short_call = 101.0
    long_call = 103.0
    wing = 2.0

    put_credit = 0.15
    call_credit = 0.15
    total_credit = put_credit + call_credit

    # 1) Close between shorts -> max profit = full credit
    close = np.array([100.0])
    pnl = _settlement_pnl_per_share(
        close,
        np.array([short_put]),
        np.array([long_put]),
        np.array([short_call]),
        np.array([long_call]),
        np.array([put_credit]),
        np.array([call_credit]),
    )[0]
    assert abs(pnl - total_credit) < 1e-9

    # 2) Close beyond long put -> max loss on put side
    close = np.array([95.0])
    pnl = _settlement_pnl_per_share(
        close,
        np.array([short_put]),
        np.array([long_put]),
        np.array([short_call]),
        np.array([long_call]),
        np.array([put_credit]),
        np.array([call_credit]),
    )[0]
    assert abs(pnl - (-(wing - total_credit))) < 1e-9

    # 3) Close beyond long call -> max loss on call side
    close = np.array([105.0])
    pnl = _settlement_pnl_per_share(
        close,
        np.array([short_put]),
        np.array([long_put]),
        np.array([short_call]),
        np.array([long_call]),
        np.array([put_credit]),
        np.array([call_credit]),
    )[0]
    assert abs(pnl - (-(wing - total_credit))) < 1e-9

    # 4) Close exactly at short put boundary -> still max profit.
    close = np.array([short_put])
    pnl = _settlement_pnl_per_share(
        close,
        np.array([short_put]),
        np.array([long_put]),
        np.array([short_call]),
        np.array([long_call]),
        np.array([put_credit]),
        np.array([call_credit]),
    )[0]
    assert abs(pnl - total_credit) < 1e-9
    return True


if __name__ == "__main__":
    np.random.seed(DEFAULT_RANDOM_SEED)
    print("Running synthetic backtester self-tests...")
    test_bs_pricer()
    test_settlement_pnl()
    print("All synthetic backtester tests passed.")
