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
MIN_BS_REPRICE_MINUTES = 5.0
MIN_BS_REPRICE_TTE_YEARS = MIN_BS_REPRICE_MINUTES / (TRADING_DAYS_PER_YEAR * 390.0)

SCENARIO_COLUMNS = {
    "hold_to_expiry": "outcome_hold_to_expiry",
    "worst_case": "outcome_worst_case",
    "tp50_or_expiry": "outcome_tp50_or_expiry",
    "tp50_sl_capped": "outcome_tp50_sl_capped",
    "dynamic_sl": "outcome_dynamic_sl",
    "ev_based_exit": "outcome_ev_based_exit",
}

PRE_COMMISSION_COLUMNS = {
    "hold_to_expiry": "hold_to_expiry_pre_commission",
    "worst_case": "worst_case_pre_commission",
    "tp50_or_expiry": "tp50_or_expiry_pre_commission",
    "tp50_sl_capped": "tp50_sl_capped_pre_commission",
    "dynamic_sl": "dynamic_sl_pre_commission",
    "ev_based_exit": "ev_based_exit_pre_commission",
}

COMMISSION_COLUMNS = {
    "hold_to_expiry": "commission_hold_to_expiry",
    "worst_case": "commission_worst_case",
    "tp50_or_expiry": "commission_tp50_or_expiry",
    "tp50_sl_capped": "commission_tp50_sl_capped",
    "dynamic_sl": "commission_dynamic_sl",
    "ev_based_exit": "commission_ev_based_exit",
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


def reprice_ic_spread(
    S_t: float,
    K_short_put: float,
    K_long_put: float,
    K_short_call: float,
    K_long_call: float,
    TTE: float,
    sigma: float,
    r: float,
) -> float:
    """
    Reprice a 4-leg iron condor using Black-Scholes and return cost-to-close per share.

    This reprices the short and long verticals separately:
      spread_cost = (short_put - long_put) + (short_call - long_call)
    """
    put_short = bs_price(S_t, K_short_put, TTE, r, sigma, "put")
    put_long = bs_price(S_t, K_long_put, TTE, r, sigma, "put")
    call_short = bs_price(S_t, K_short_call, TTE, r, sigma, "call")
    call_long = bs_price(S_t, K_long_call, TTE, r, sigma, "call")
    spread_cost = (put_short - put_long) + (call_short - call_long)
    return float(max(spread_cost, 0.0))


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


def _reprice_ic_spread_vectorized(
    S_t: np.ndarray,
    short_put: np.ndarray,
    long_put: np.ndarray,
    short_call: np.ndarray,
    long_call: np.ndarray,
    TTE: float,
    sigma: np.ndarray,
    r: float,
) -> np.ndarray:
    """Vectorized iron condor cost-to-close repricing (per share) using Black-Scholes."""
    put_short = _bs_price_vectorized(S_t, short_put, TTE, r, sigma, "put")
    put_long = _bs_price_vectorized(S_t, long_put, TTE, r, sigma, "put")
    call_short = _bs_price_vectorized(S_t, short_call, TTE, r, sigma, "call")
    call_long = _bs_price_vectorized(S_t, long_call, TTE, r, sigma, "call")
    return np.maximum((put_short - put_long) + (call_short - call_long), 0.0)


def _linear_spread_cost_proxy_vectorized(
    S_t: np.ndarray,
    short_put: np.ndarray,
    long_put: np.ndarray,
    short_call: np.ndarray,
    long_call: np.ndarray,
) -> np.ndarray:
    """
    Legacy linear proxy for spread cost (per share) based on strike intrusion.

    Proxy = min(max(intrusion past short put, intrusion past short call), wing_width).
    """
    S_t = np.asarray(S_t, dtype=float)
    put_intrusion = np.maximum(short_put - S_t, 0.0)
    call_intrusion = np.maximum(S_t - short_call, 0.0)
    intrusion = np.maximum(put_intrusion, call_intrusion)
    wing_width = np.maximum(short_put - long_put, long_call - short_call)
    return np.minimum(intrusion, np.maximum(wing_width, 0.0))


def _spread_cost_with_fallback_vectorized(
    S_t: np.ndarray,
    short_put: np.ndarray,
    long_put: np.ndarray,
    short_call: np.ndarray,
    long_call: np.ndarray,
    TTE: float,
    sigma: np.ndarray,
    r: float,
    use_bs_repricing: bool,
) -> np.ndarray:
    """
    Cost-to-close proxy with BS repricing and short-T fallback to legacy linear proxy.
    """
    if (not use_bs_repricing) or (TTE <= MIN_BS_REPRICE_TTE_YEARS):
        return _linear_spread_cost_proxy_vectorized(S_t, short_put, long_put, short_call, long_call)
    return _reprice_ic_spread_vectorized(S_t, short_put, long_put, short_call, long_call, TTE, sigma, r)


def _checkpoint_ttes_from_entry_T(T_entry: float) -> tuple[float, float]:
    """
    Coarse OHLC checkpoint model for daily bars:
    - checkpoint 1 at ~1/3 elapsed (2/3 T remaining)
    - checkpoint 2 at ~2/3 elapsed (1/3 T remaining)
    """
    t1 = max(float(T_entry) * (2.0 / 3.0), 0.0)
    t2 = max(float(T_entry) * (1.0 / 3.0), 0.0)
    return t1, t2


def _build_legacy_exit_annotations(
    skipped: np.ndarray,
    touched_short: np.ndarray,
    untouched: np.ndarray,
    settlement_pre_commission: np.ndarray,
    tp_cap_usd: np.ndarray,
    sl_cap_usd: np.ndarray,
    S_open: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    short_put: np.ndarray,
    short_call: np.ndarray,
    T_entry: float,
) -> Dict[str, np.ndarray]:
    """
    Legacy path annotations (reasons + trigger metadata) for comparison reporting.

    This preserves existing scenario PnL behavior and only adds metadata.
    """
    n = len(S_open)
    tp_reason = np.full(n, "", dtype=object)
    tp_sl_reason = np.full(n, "", dtype=object)
    tp_spot = np.full(n, np.nan, dtype=float)
    tp_sl_spot = np.full(n, np.nan, dtype=float)
    tp_tte = np.full(n, np.nan, dtype=float)
    tp_sl_tte = np.full(n, np.nan, dtype=float)

    t1_tte, t2_tte = _checkpoint_ttes_from_entry_T(T_entry)
    first_is_low = close >= S_open
    first_spot = np.where(first_is_low, low, high)
    second_spot = np.where(first_is_low, high, low)
    first_is_touch = np.where(first_is_low, low <= short_put, high >= short_call)
    second_is_touch = np.where(first_is_low, high >= short_call, low <= short_put)
    legacy_touch_trigger = touched_short.astype(bool)

    tp_hit_legacy = untouched & (settlement_pre_commission > tp_cap_usd)
    tp_reason[:] = np.where(skipped, "", np.where(tp_hit_legacy, "TP", "EXPIRY")).astype(object)
    tp_spot[:] = np.where(tp_hit_legacy, close, np.nan)
    tp_tte[:] = np.where(tp_hit_legacy, 0.0, np.nan)

    sl_hit_legacy = touched_short & (settlement_pre_commission < -sl_cap_usd)
    tp_hit_legacy_sl_scenario = untouched & (settlement_pre_commission > tp_cap_usd)
    tp_sl_reason[:] = np.where(
        skipped,
        "",
        np.where(sl_hit_legacy, "SL", np.where(tp_hit_legacy_sl_scenario, "TP", "EXPIRY")),
    ).astype(object)

    # Approximate trigger metadata for legacy SL as first touched extreme in our coarse path model.
    use_first_sl = sl_hit_legacy & first_is_touch
    use_second_sl = sl_hit_legacy & (~first_is_touch) & second_is_touch
    tp_sl_spot[:] = np.where(use_first_sl, first_spot, tp_sl_spot)
    tp_sl_spot[:] = np.where(use_second_sl, second_spot, tp_sl_spot)
    tp_sl_tte[:] = np.where(use_first_sl, t1_tte, tp_sl_tte)
    tp_sl_tte[:] = np.where(use_second_sl, t2_tte, tp_sl_tte)

    # Legacy TP trigger metadata is only inferable from settlement under the current model.
    tp_sl_spot[:] = np.where(tp_hit_legacy_sl_scenario, close, tp_sl_spot)
    tp_sl_tte[:] = np.where(tp_hit_legacy_sl_scenario, 0.0, tp_sl_tte)

    tp_dist = np.minimum(np.abs(tp_spot - short_put), np.abs(short_call - tp_spot))
    tp_sl_dist = np.minimum(np.abs(tp_sl_spot - short_put), np.abs(short_call - tp_sl_spot))

    return {
        "tp50_or_expiry_exit_type": tp_reason,
        "tp50_or_expiry_trigger_spot": tp_spot,
        "tp50_or_expiry_tte_at_trigger": tp_tte,
        "tp50_or_expiry_distance_to_short_strike": tp_dist,
        "tp50_sl_capped_exit_type": tp_sl_reason,
        "tp50_sl_capped_trigger_spot": tp_sl_spot,
        "tp50_sl_capped_tte_at_trigger": tp_sl_tte,
        "tp50_sl_capped_distance_to_short_strike": tp_sl_dist,
        "tp50_sl_capped_sl_trigger": sl_hit_legacy.astype(bool),
        "legacy_sl_touch_trigger": legacy_touch_trigger.astype(bool),
    }


def _build_bs_exit_annotations_and_pnl(
    skipped: np.ndarray,
    S_open: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    sigma: np.ndarray,
    short_put: np.ndarray,
    long_put: np.ndarray,
    short_call: np.ndarray,
    long_call: np.ndarray,
    entry_credit: np.ndarray,
    settlement_pre_commission: np.ndarray,
    tp_cap_usd: np.ndarray,
    sl_cap_usd: np.ndarray,
    T_entry: float,
    r: float,
    tp_pct: float,
    use_bs_repricing: bool,
) -> Dict[str, np.ndarray]:
    """
    Coarse intraday OHLC checkpoint engine for TP/SL using BS spread repricing.

    Path order assumption (deterministic, due daily OHLC limitations):
      - Up-close day:   Open -> Low -> High -> Close
      - Down-close day: Open -> High -> Low -> Close
    """
    n = len(S_open)
    tp_pre = settlement_pre_commission.copy()
    tp_sl_pre = settlement_pre_commission.copy()

    tp_reason = np.full(n, "", dtype=object)
    tp_sl_reason = np.full(n, "", dtype=object)
    tp_spot = np.full(n, np.nan, dtype=float)
    tp_sl_spot = np.full(n, np.nan, dtype=float)
    tp_tte = np.full(n, np.nan, dtype=float)
    tp_sl_tte = np.full(n, np.nan, dtype=float)

    t1_tte, t2_tte = _checkpoint_ttes_from_entry_T(T_entry)
    first_is_low = close >= S_open
    s1 = np.where(first_is_low, low, high)
    s2 = np.where(first_is_low, high, low)

    c1 = _spread_cost_with_fallback_vectorized(
        s1, short_put, long_put, short_call, long_call, t1_tte, sigma, r, use_bs_repricing
    )
    c2 = _spread_cost_with_fallback_vectorized(
        s2, short_put, long_put, short_call, long_call, t2_tte, sigma, r, use_bs_repricing
    )
    # Settlement spread cost = entry_credit - settlement PnL/share.
    c3 = np.maximum(entry_credit - (settlement_pre_commission / XSP_MULTIPLIER), 0.0)

    tp_enabled = bool(tp_pct < 1.0 - 1e-12)
    tp_target_cost = np.maximum(entry_credit * (1.0 - tp_pct), 0.0)
    sl_target_cost = np.maximum(entry_credit * 0.0 + (sl_cap_usd / XSP_MULTIPLIER), 0.0)

    valid = (~skipped) & np.isfinite(entry_credit) & np.isfinite(settlement_pre_commission)

    # Scenario: TP50 or expiry (no SL)
    if tp_enabled:
        tp1 = valid & (c1 <= tp_target_cost)
        tp2 = valid & (~tp1) & (c2 <= tp_target_cost)
        tp3 = valid & (~tp1) & (~tp2) & (c3 <= tp_target_cost)
        tp_hit = tp1 | tp2 | tp3
        tp_pre = np.where(tp_hit, tp_cap_usd, settlement_pre_commission)
        tp_reason[:] = np.where(skipped, "", np.where(tp_hit, "TP", "EXPIRY")).astype(object)
        tp_spot[:] = np.where(tp1, s1, tp_spot)
        tp_spot[:] = np.where(tp2, s2, tp_spot)
        tp_spot[:] = np.where(tp3, close, tp_spot)
        tp_tte[:] = np.where(tp1, t1_tte, tp_tte)
        tp_tte[:] = np.where(tp2, t2_tte, tp_tte)
        tp_tte[:] = np.where(tp3, 0.0, tp_tte)
    else:
        tp_reason[:] = np.where(skipped, "", "EXPIRY").astype(object)

    # Scenario: TP50 + SL capped (SL priority at each checkpoint)
    sl_hit = np.zeros(n, dtype=bool)
    tp_hit_sl_scenario = np.zeros(n, dtype=bool)

    undecided = valid.copy()
    sl1 = undecided & (c1 >= sl_target_cost)
    tp1s = undecided & (~sl1) & tp_enabled & (c1 <= tp_target_cost)
    sl_hit |= sl1
    tp_hit_sl_scenario |= tp1s
    tp_sl_spot[:] = np.where(sl1 | tp1s, s1, tp_sl_spot)
    tp_sl_tte[:] = np.where(sl1 | tp1s, t1_tte, tp_sl_tte)
    undecided &= ~(sl1 | tp1s)

    sl2 = undecided & (c2 >= sl_target_cost)
    tp2s = undecided & (~sl2) & tp_enabled & (c2 <= tp_target_cost)
    sl_hit |= sl2
    tp_hit_sl_scenario |= tp2s
    tp_sl_spot[:] = np.where(sl2 | tp2s, s2, tp_sl_spot)
    tp_sl_tte[:] = np.where(sl2 | tp2s, t2_tte, tp_sl_tte)
    undecided &= ~(sl2 | tp2s)

    # No active close at settlement for this scenario if still undecided -> expire.
    tp_sl_pre = np.where(sl_hit, -sl_cap_usd, tp_sl_pre)
    tp_sl_pre = np.where(tp_hit_sl_scenario, tp_cap_usd, tp_sl_pre)
    tp_sl_reason[:] = np.where(
        skipped,
        "",
        np.where(sl_hit, "SL", np.where(tp_hit_sl_scenario, "TP", "EXPIRY")),
    ).astype(object)

    tp_dist = np.minimum(np.abs(tp_spot - short_put), np.abs(short_call - tp_spot))
    tp_sl_dist = np.minimum(np.abs(tp_sl_spot - short_put), np.abs(short_call - tp_sl_spot))

    return {
        "tp50_or_expiry_pre_commission": tp_pre,
        "tp50_or_expiry_exit_type": tp_reason,
        "tp50_or_expiry_trigger_spot": tp_spot,
        "tp50_or_expiry_tte_at_trigger": tp_tte,
        "tp50_or_expiry_distance_to_short_strike": tp_dist,
        "tp50_sl_capped_pre_commission": tp_sl_pre,
        "tp50_sl_capped_exit_type": tp_sl_reason,
        "tp50_sl_capped_trigger_spot": tp_sl_spot,
        "tp50_sl_capped_tte_at_trigger": tp_sl_tte,
        "tp50_sl_capped_distance_to_short_strike": tp_sl_dist,
        "tp50_sl_capped_sl_trigger": sl_hit.astype(bool),
    }


def calc_continuation_value(
    S_t: float,
    K_short_put: float,
    K_short_call: float,
    TTE: float,
    sigma: float,
    r: float,
    credit: float,
    wing: float,
    qty: float = 1.0,
    multiplier: float = XSP_MULTIPLIER,
) -> tuple[float, float, float, float]:
    """
    Continuation value for an iron condor using conservative min-side P(OTM).

    Returns: (V_continue, P_OTM_min, P_OTM_put, P_OTM_call)
    """
    if not (
        np.isfinite(S_t)
        and np.isfinite(K_short_put)
        and np.isfinite(K_short_call)
        and np.isfinite(TTE)
        and np.isfinite(sigma)
        and np.isfinite(credit)
        and np.isfinite(wing)
        and S_t > 0
        and K_short_put > 0
        and K_short_call > 0
        and TTE > 0
        and sigma > 0
    ):
        return (float("nan"), float("nan"), float("nan"), float("nan"))

    sqrt_t = float(np.sqrt(TTE))
    d2_put = (np.log(S_t / K_short_put) + (r - 0.5 * sigma**2) * TTE) / (sigma * sqrt_t)
    d2_call = (np.log(S_t / K_short_call) + (r - 0.5 * sigma**2) * TTE) / (sigma * sqrt_t)
    p_otm_put = float(norm.cdf(d2_put))       # P(S_T > K_short_put)
    p_otm_call = float(norm.cdf(-d2_call))    # P(S_T < K_short_call)
    p_otm = float(min(p_otm_put, p_otm_call))
    p_itm = 1.0 - p_otm

    e_profit = float(credit) * float(multiplier) * float(qty)
    e_loss_itm = max(float(wing) - float(credit), 0.0) * float(multiplier) * float(qty)
    v_continue = p_otm * e_profit - p_itm * e_loss_itm
    return (float(v_continue), p_otm, p_otm_put, p_otm_call)


def _calc_continuation_value_vectorized(
    S_t: np.ndarray,
    short_put: np.ndarray,
    short_call: np.ndarray,
    TTE: float,
    sigma: np.ndarray,
    r: float,
    credit: np.ndarray,
    wing_width: float,
    qty: float = 1.0,
    multiplier: float = XSP_MULTIPLIER,
) -> Dict[str, np.ndarray]:
    """Vectorized continuation-value calculation for EV-based exits."""
    n = len(S_t)
    v = np.full(n, np.nan, dtype=float)
    p_min = np.full(n, np.nan, dtype=float)
    p_put = np.full(n, np.nan, dtype=float)
    p_call = np.full(n, np.nan, dtype=float)

    if (not np.isfinite(TTE)) or TTE <= 0.0:
        return {"V_continue": v, "P_OTM_min": p_min, "P_OTM_put": p_put, "P_OTM_call": p_call}

    S_t = np.asarray(S_t, dtype=float)
    short_put = np.asarray(short_put, dtype=float)
    short_call = np.asarray(short_call, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    credit = np.asarray(credit, dtype=float)

    valid = (
        np.isfinite(S_t)
        & np.isfinite(short_put)
        & np.isfinite(short_call)
        & np.isfinite(sigma)
        & np.isfinite(credit)
        & (S_t > 0)
        & (short_put > 0)
        & (short_call > 0)
        & (sigma > 0)
    )
    if not np.any(valid):
        return {"V_continue": v, "P_OTM_min": p_min, "P_OTM_put": p_put, "P_OTM_call": p_call}

    sqrt_t = float(np.sqrt(TTE))
    with np.errstate(divide="ignore", invalid="ignore"):
        d2_put = (np.log(S_t[valid] / short_put[valid]) + (r - 0.5 * sigma[valid] ** 2) * TTE) / (sigma[valid] * sqrt_t)
        d2_call = (np.log(S_t[valid] / short_call[valid]) + (r - 0.5 * sigma[valid] ** 2) * TTE) / (sigma[valid] * sqrt_t)

    p_put_valid = norm.cdf(d2_put)
    p_call_valid = norm.cdf(-d2_call)
    p_min_valid = np.minimum(p_put_valid, p_call_valid)
    p_itm_valid = 1.0 - p_min_valid

    e_profit = credit[valid] * float(multiplier) * float(qty)
    e_loss_itm = np.maximum(float(wing_width) - credit[valid], 0.0) * float(multiplier) * float(qty)
    v_valid = p_min_valid * e_profit - p_itm_valid * e_loss_itm

    p_put[valid] = p_put_valid
    p_call[valid] = p_call_valid
    p_min[valid] = p_min_valid
    v[valid] = v_valid
    return {"V_continue": v, "P_OTM_min": p_min, "P_OTM_put": p_put, "P_OTM_call": p_call}


def calc_dynamic_sl(
    P_OTM: np.ndarray,
    tte_ratio: np.ndarray,
    sl_floor: float,
    sl_ceiling: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """
    Compute dynamic SL multiplier as a function of P(OTM) and time remaining.
    """
    p = np.clip(np.asarray(P_OTM, dtype=float), 0.0, 1.0)
    t = np.clip(np.asarray(tte_ratio, dtype=float), 0.0, 1.0)
    base = float(sl_floor)
    span = float(sl_ceiling) - float(sl_floor)
    return base + span * np.power(p, float(alpha)) * np.power(t, float(beta))


def calc_P_OTM_vectorized(
    S_t: np.ndarray,
    K_short_put: np.ndarray,
    K_short_call: np.ndarray,
    TTE: float,
    sigma: np.ndarray,
    r: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute P(OTM) for both sides of the condor using Black-Scholes d2.

    Returns (P_OTM_min, P_OTM_put, P_OTM_call); returns NaNs when TTE is too small.
    """
    S_t = np.asarray(S_t, dtype=float)
    K_short_put = np.asarray(K_short_put, dtype=float)
    K_short_call = np.asarray(K_short_call, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = len(S_t)

    p_put = np.full(n, np.nan, dtype=float)
    p_call = np.full(n, np.nan, dtype=float)
    p_min = np.full(n, np.nan, dtype=float)

    if (not np.isfinite(TTE)) or (TTE <= MIN_BS_REPRICE_TTE_YEARS):
        return p_min, p_put, p_call

    valid = (
        np.isfinite(S_t)
        & np.isfinite(K_short_put)
        & np.isfinite(K_short_call)
        & np.isfinite(sigma)
        & (S_t > 0)
        & (K_short_put > 0)
        & (K_short_call > 0)
        & (sigma > 0)
    )
    if not np.any(valid):
        return p_min, p_put, p_call

    sqrt_T = float(np.sqrt(TTE))
    with np.errstate(divide="ignore", invalid="ignore"):
        d2_put = (np.log(S_t[valid] / K_short_put[valid]) + (r - 0.5 * sigma[valid] ** 2) * TTE) / (sigma[valid] * sqrt_T)
        d2_call = (np.log(S_t[valid] / K_short_call[valid]) + (r - 0.5 * sigma[valid] ** 2) * TTE) / (sigma[valid] * sqrt_T)
    p_put[valid] = norm.cdf(d2_put)      # P(S_T > K_short_put)
    p_call[valid] = norm.cdf(-d2_call)   # P(S_T < K_short_call)
    p_min = np.minimum(p_put, p_call)
    return p_min, p_put, p_call


def _build_dynamic_sl_annotations_and_pnl(
    skipped: np.ndarray,
    S_open: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    sigma: np.ndarray,
    short_put: np.ndarray,
    long_put: np.ndarray,
    short_call: np.ndarray,
    long_call: np.ndarray,
    entry_credit: np.ndarray,
    settlement_pre_commission: np.ndarray,
    T_entry: float,
    r: float,
    xsp_multiplier: float,
    use_bs_repricing: bool,
    sl_floor: float,
    sl_ceiling: float,
    alpha: float,
    beta: float,
) -> Dict[str, np.ndarray]:
    """
    Dynamic SL scenario: state-dependent SL multiplier based on P(OTM) and TTE.

    No TP; trade holds to expiry unless DYNAMIC_SL triggers at a checkpoint.
    """
    n = len(S_open)
    result_pre = settlement_pre_commission.copy()
    exit_reason = np.full(n, "", dtype=object)
    exit_spot = np.full(n, np.nan, dtype=float)
    exit_tte = np.full(n, np.nan, dtype=float)
    exit_P_OTM = np.full(n, np.nan, dtype=float)
    exit_sl_mult = np.full(n, np.nan, dtype=float)
    exit_dist = np.full(n, np.nan, dtype=float)
    exit_spread_cost = np.full(n, np.nan, dtype=float)
    sl_triggered = np.zeros(n, dtype=bool)

    t1_tte, t2_tte = _checkpoint_ttes_from_entry_T(T_entry)
    first_is_low = close >= S_open
    s1 = np.where(first_is_low, low, high)
    s2 = np.where(first_is_low, high, low)

    valid = (~skipped) & np.isfinite(entry_credit) & np.isfinite(settlement_pre_commission)
    if T_entry > 0:
        tte_ratio_1 = float(np.clip(t1_tte / T_entry, 0.0, 1.0))
        tte_ratio_2 = float(np.clip(t2_tte / T_entry, 0.0, 1.0))
    else:
        tte_ratio_1 = 0.0
        tte_ratio_2 = 0.0

    # Checkpoint 1
    c1 = _spread_cost_with_fallback_vectorized(s1, short_put, long_put, short_call, long_call, t1_tte, sigma, r, use_bs_repricing)
    pnl1 = (entry_credit - c1) * float(xsp_multiplier)
    p_min_1, p_put_1, p_call_1 = calc_P_OTM_vectorized(s1, short_put, short_call, t1_tte, sigma, r)
    sl_mult_1 = calc_dynamic_sl(p_min_1, np.full(n, tte_ratio_1, dtype=float), sl_floor, sl_ceiling, alpha, beta)
    sl_cost_1 = entry_credit * sl_mult_1
    sl1 = valid & np.isfinite(p_min_1) & np.isfinite(c1) & np.isfinite(sl_cost_1) & (c1 >= sl_cost_1) & (pnl1 < 0)

    sl_triggered |= sl1
    result_pre = np.where(sl1, -(c1 - entry_credit) * float(xsp_multiplier), result_pre)
    exit_spot = np.where(sl1, s1, exit_spot)
    exit_tte = np.where(sl1, t1_tte, exit_tte)
    exit_P_OTM = np.where(sl1, p_min_1, exit_P_OTM)
    exit_sl_mult = np.where(sl1, sl_mult_1, exit_sl_mult)
    exit_spread_cost = np.where(sl1, c1, exit_spread_cost)

    undecided = valid & (~sl_triggered)

    # Checkpoint 2
    c2 = _spread_cost_with_fallback_vectorized(s2, short_put, long_put, short_call, long_call, t2_tte, sigma, r, use_bs_repricing)
    pnl2 = (entry_credit - c2) * float(xsp_multiplier)
    p_min_2, p_put_2, p_call_2 = calc_P_OTM_vectorized(s2, short_put, short_call, t2_tte, sigma, r)
    sl_mult_2 = calc_dynamic_sl(p_min_2, np.full(n, tte_ratio_2, dtype=float), sl_floor, sl_ceiling, alpha, beta)
    sl_cost_2 = entry_credit * sl_mult_2
    sl2 = undecided & np.isfinite(p_min_2) & np.isfinite(c2) & np.isfinite(sl_cost_2) & (c2 >= sl_cost_2) & (pnl2 < 0)

    sl_triggered |= sl2
    result_pre = np.where(sl2, -(c2 - entry_credit) * float(xsp_multiplier), result_pre)
    exit_spot = np.where(sl2, s2, exit_spot)
    exit_tte = np.where(sl2, t2_tte, exit_tte)
    exit_P_OTM = np.where(sl2, p_min_2, exit_P_OTM)
    exit_sl_mult = np.where(sl2, sl_mult_2, exit_sl_mult)
    exit_spread_cost = np.where(sl2, c2, exit_spread_cost)

    exit_reason[:] = np.where(
        skipped,
        "",
        np.where(
            sl_triggered,
            "DYNAMIC_SL",
            np.where((settlement_pre_commission >= 0), "EXPIRED_OTM", "EXPIRED_ITM"),
        ),
    ).astype(object)

    # Settlement rows use settlement spot and no active SL metadata.
    expired = valid & (~sl_triggered)
    exit_spot = np.where(expired, close, exit_spot)
    exit_tte = np.where(expired, 0.0, exit_tte)
    exit_dist = np.minimum(np.abs(exit_spot - short_put), np.abs(short_call - exit_spot))

    # Keep optional diagnostic arrays for both sides if needed later.
    # For triggered rows, pick checkpoint-side values; expiry rows remain NaN.
    exit_p_put = np.full(n, np.nan, dtype=float)
    exit_p_call = np.full(n, np.nan, dtype=float)
    exit_p_put = np.where(sl1, p_put_1, exit_p_put)
    exit_p_put = np.where(sl2, p_put_2, exit_p_put)
    exit_p_call = np.where(sl1, p_call_1, exit_p_call)
    exit_p_call = np.where(sl2, p_call_2, exit_p_call)

    return {
        "dynamic_sl_pre_commission": result_pre,
        "dynamic_sl_exit_type": exit_reason,
        "dynamic_sl_trigger_spot": exit_spot,
        "dynamic_sl_tte_at_trigger": exit_tte,
        "dynamic_sl_P_OTM_at_trigger": exit_P_OTM,
        "dynamic_sl_P_OTM_put_at_trigger": exit_p_put,
        "dynamic_sl_P_OTM_call_at_trigger": exit_p_call,
        "dynamic_sl_mult_at_trigger": exit_sl_mult,
        "dynamic_sl_distance_to_short_strike": exit_dist,
        "dynamic_sl_spread_cost_at_trigger": exit_spread_cost,
        "dynamic_sl_triggered": sl_triggered.astype(bool),
    }


def _build_ev_exit_annotations_and_pnl(
    skipped: np.ndarray,
    S_open: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    sigma: np.ndarray,
    short_put: np.ndarray,
    long_put: np.ndarray,
    short_call: np.ndarray,
    long_call: np.ndarray,
    entry_credit: np.ndarray,
    settlement_pre_commission: np.ndarray,
    T_entry: float,
    r: float,
    wing_width: float,
    xsp_multiplier: float,
    use_bs_repricing: bool,
) -> Dict[str, np.ndarray]:
    """
    EV-based early-exit scenario using continuation value vs current close cost.

    Evaluates at the same coarse OHLC checkpoints used by TP/SL scenarios.
    """
    n = len(S_open)
    ev_pre = settlement_pre_commission.copy()
    exit_reason = np.full(n, "", dtype=object)
    trigger_spot = np.full(n, np.nan, dtype=float)
    trigger_tte = np.full(n, np.nan, dtype=float)
    dist_to_short = np.full(n, np.nan, dtype=float)
    p_otm_min = np.full(n, np.nan, dtype=float)
    p_otm_put = np.full(n, np.nan, dtype=float)
    p_otm_call = np.full(n, np.nan, dtype=float)
    v_continue_out = np.full(n, np.nan, dtype=float)
    early_exit = np.zeros(n, dtype=bool)

    t1_tte, t2_tte = _checkpoint_ttes_from_entry_T(T_entry)
    first_is_low = close >= S_open
    s1 = np.where(first_is_low, low, high)
    s2 = np.where(first_is_low, high, low)

    c1 = _spread_cost_with_fallback_vectorized(
        s1, short_put, long_put, short_call, long_call, t1_tte, sigma, r, use_bs_repricing
    )
    c2 = _spread_cost_with_fallback_vectorized(
        s2, short_put, long_put, short_call, long_call, t2_tte, sigma, r, use_bs_repricing
    )
    close1_pre = (entry_credit - c1) * float(xsp_multiplier)
    close2_pre = (entry_credit - c2) * float(xsp_multiplier)

    cv1 = _calc_continuation_value_vectorized(
        s1, short_put, short_call, t1_tte, sigma, r, entry_credit, wing_width, qty=1.0, multiplier=xsp_multiplier
    )
    cv2 = _calc_continuation_value_vectorized(
        s2, short_put, short_call, t2_tte, sigma, r, entry_credit, wing_width, qty=1.0, multiplier=xsp_multiplier
    )

    valid = (~skipped) & np.isfinite(entry_credit) & np.isfinite(settlement_pre_commission)
    can_eval_1 = valid & (t1_tte > MIN_BS_REPRICE_TTE_YEARS) & np.isfinite(cv1["V_continue"]) & np.isfinite(close1_pre)
    can_eval_2 = valid & (t2_tte > MIN_BS_REPRICE_TTE_YEARS) & np.isfinite(cv2["V_continue"]) & np.isfinite(close2_pre)

    exit1 = can_eval_1 & (cv1["V_continue"] < close1_pre)
    exit2 = (~exit1) & can_eval_2 & (cv2["V_continue"] < close2_pre)

    early_exit |= exit1 | exit2
    ev_pre = np.where(exit1, close1_pre, ev_pre)
    ev_pre = np.where(exit2, close2_pre, ev_pre)

    trigger_spot = np.where(exit1, s1, trigger_spot)
    trigger_spot = np.where(exit2, s2, trigger_spot)
    trigger_tte = np.where(exit1, t1_tte, trigger_tte)
    trigger_tte = np.where(exit2, t2_tte, trigger_tte)

    p_otm_min = np.where(exit1, cv1["P_OTM_min"], p_otm_min)
    p_otm_min = np.where(exit2, cv2["P_OTM_min"], p_otm_min)
    p_otm_put = np.where(exit1, cv1["P_OTM_put"], p_otm_put)
    p_otm_put = np.where(exit2, cv2["P_OTM_put"], p_otm_put)
    p_otm_call = np.where(exit1, cv1["P_OTM_call"], p_otm_call)
    p_otm_call = np.where(exit2, cv2["P_OTM_call"], p_otm_call)
    v_continue_out = np.where(exit1, cv1["V_continue"], v_continue_out)
    v_continue_out = np.where(exit2, cv2["V_continue"], v_continue_out)

    expired_mask = valid & (~early_exit)
    expired_otm = expired_mask & (close >= short_put) & (close <= short_call)
    expired_itm = expired_mask & (~expired_otm)
    exit_reason[:] = np.where(skipped, "", np.where(early_exit, "EV_NEGATIVE", np.where(expired_otm, "EXPIRED_OTM", "EXPIRED_ITM"))).astype(object)

    # For expiry rows, expose last evaluated continuation metrics (checkpoint 2, then checkpoint 1) for diagnostics.
    last_eval_2 = expired_mask & can_eval_2
    last_eval_1 = expired_mask & (~can_eval_2) & can_eval_1
    p_otm_min = np.where(last_eval_2, cv2["P_OTM_min"], p_otm_min)
    p_otm_min = np.where(last_eval_1, cv1["P_OTM_min"], p_otm_min)
    p_otm_put = np.where(last_eval_2, cv2["P_OTM_put"], p_otm_put)
    p_otm_put = np.where(last_eval_1, cv1["P_OTM_put"], p_otm_put)
    p_otm_call = np.where(last_eval_2, cv2["P_OTM_call"], p_otm_call)
    p_otm_call = np.where(last_eval_1, cv1["P_OTM_call"], p_otm_call)
    v_continue_out = np.where(last_eval_2, cv2["V_continue"], v_continue_out)
    v_continue_out = np.where(last_eval_1, cv1["V_continue"], v_continue_out)
    # "At exit" for expiry is settlement; use 0.0. Add a separate eval TTE for diagnostics.
    eval_tte = np.full(n, np.nan, dtype=float)
    eval_tte = np.where(exit1, t1_tte, eval_tte)
    eval_tte = np.where(exit2, t2_tte, eval_tte)
    eval_tte = np.where(last_eval_2, t2_tte, eval_tte)
    eval_tte = np.where(last_eval_1, t1_tte, eval_tte)

    trigger_spot = np.where(expired_mask, close, trigger_spot)
    trigger_tte = np.where(expired_mask, 0.0, trigger_tte)

    dist_to_short = np.minimum(np.abs(trigger_spot - short_put), np.abs(short_call - trigger_spot))

    return {
        "ev_based_exit_pre_commission": ev_pre,
        "ev_based_exit_exit_reason": exit_reason,
        "ev_based_exit_trigger_spot": trigger_spot,
        "ev_based_exit_tte_at_exit": trigger_tte,
        "ev_based_exit_tte_at_eval": eval_tte,
        "ev_based_exit_distance_to_short_strike": dist_to_short,
        "ev_based_exit_P_OTM_min": p_otm_min,
        "ev_based_exit_P_OTM_put": p_otm_put,
        "ev_based_exit_P_OTM_call": p_otm_call,
        "ev_based_exit_V_continue": v_continue_out,
        "ev_based_exit_early_exit": early_exit.astype(bool),
    }


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
    iv_scaling_factor = float(_cfg(config, "iv_scaling_factor", default=_cfg(config, "timing", "iv_scaling_factor", default=1.0)))
    if (not np.isfinite(iv_scaling_factor)) or iv_scaling_factor <= 0.0:
        raise ValueError("iv_scaling_factor must be a finite value > 0")

    max_credit_to_wing_ratio = _cfg(config, "sweep_filters", "max_credit_to_wing_ratio", default=np.nan)
    max_margin_per_contract = _cfg(config, "sweep_filters", "max_margin_per_contract", default=np.nan)
    dyn_sl_enabled = bool(_cfg(config, "dynamic_sl", "enabled", default=True))
    dyn_sl_floor = float(_cfg(config, "dynamic_sl", "sl_floor", default=1.5))
    dyn_sl_ceiling = float(_cfg(config, "dynamic_sl", "sl_ceiling", default=4.0))
    dyn_sl_alpha = float(_cfg(config, "dynamic_sl", "alpha", default=1.0))
    dyn_sl_beta = float(_cfg(config, "dynamic_sl", "beta", default=0.5))

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
        "use_bs_repricing": bool(_cfg(config, "pricing", "use_bs_repricing", default=True)),
        "iv_scaling_factor": float(iv_scaling_factor),
        "take_profit_pct": float(take_profit_pct),
        "stop_loss_mult": float(stop_loss_mult),
        "starting_capital": float(_cfg(config, "capital", "starting_capital", default=1580.0)),
        "xsp_multiplier": float(_cfg(config, "default", "multiplier", default=XSP_MULTIPLIER)),
        "max_credit_to_wing_ratio": finite_or_nan(max_credit_to_wing_ratio),
        "max_margin_per_contract": finite_or_nan(max_margin_per_contract),
        "dynamic_sl_enabled": dyn_sl_enabled,
        "dynamic_sl_floor": dyn_sl_floor,
        "dynamic_sl_ceiling": dyn_sl_ceiling,
        "dynamic_sl_alpha": dyn_sl_alpha,
        "dynamic_sl_beta": dyn_sl_beta,
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

    # VIX is annualized implied vol in percent; scale to approximate 0DTE IV regime.
    sigma = (entry_vix / 100.0) * params["iv_scaling_factor"]
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
    use_bs_repricing = bool(params.get("use_bs_repricing", True))

    legacy_exit_annotations = _build_legacy_exit_annotations(
        skipped=skipped,
        touched_short=touched_short,
        untouched=untouched,
        settlement_pre_commission=settlement_pre_commission,
        tp_cap_usd=tp_cap_usd,
        sl_cap_usd=sl_cap_usd,
        S_open=S,
        high=high,
        low=low,
        close=close,
        short_put=short_put,
        short_call=short_call,
        T_entry=T,
    )

    if use_bs_repricing:
        bs_exit_annotations = _build_bs_exit_annotations_and_pnl(
            skipped=skipped,
            S_open=S,
            high=high,
            low=low,
            close=close,
            sigma=sigma,
            short_put=short_put,
            long_put=long_put,
            short_call=short_call,
            long_call=long_call,
            entry_credit=entry_credit,
            settlement_pre_commission=settlement_pre_commission,
            tp_cap_usd=tp_cap_usd,
            sl_cap_usd=sl_cap_usd,
            T_entry=T,
            r=params["risk_free_rate"],
            tp_pct=params["take_profit_pct"],
            use_bs_repricing=use_bs_repricing,
        )
        tp_pre_commission = bs_exit_annotations["tp50_or_expiry_pre_commission"]
        tp50_sl_capped_pre_commission = bs_exit_annotations["tp50_sl_capped_pre_commission"]
        exit_annotations = {**legacy_exit_annotations, **bs_exit_annotations}
        pricing_mode = "bs"
    else:
        exit_annotations = legacy_exit_annotations
        pricing_mode = "linear"

    ev_exit_annotations = _build_ev_exit_annotations_and_pnl(
        skipped=skipped,
        S_open=S,
        high=high,
        low=low,
        close=close,
        sigma=sigma,
        short_put=short_put,
        long_put=long_put,
        short_call=short_call,
        long_call=long_call,
        entry_credit=entry_credit,
        settlement_pre_commission=settlement_pre_commission,
        T_entry=T,
        r=params["risk_free_rate"],
        wing_width=params["wing_width"],
        xsp_multiplier=params["xsp_multiplier"],
        use_bs_repricing=use_bs_repricing,
    )
    dynamic_sl_annotations = _build_dynamic_sl_annotations_and_pnl(
        skipped=skipped,
        S_open=S,
        high=high,
        low=low,
        close=close,
        sigma=sigma,
        short_put=short_put,
        long_put=long_put,
        short_call=short_call,
        long_call=long_call,
        entry_credit=entry_credit,
        settlement_pre_commission=settlement_pre_commission,
        T_entry=T,
        r=params["risk_free_rate"],
        xsp_multiplier=params["xsp_multiplier"],
        use_bs_repricing=use_bs_repricing,
        sl_floor=params["dynamic_sl_floor"],
        sl_ceiling=params["dynamic_sl_ceiling"],
        alpha=params["dynamic_sl_alpha"],
        beta=params["dynamic_sl_beta"],
    )

    open_commission = float(params["open_commission"])
    round_trip_commission = float(params["round_trip_commission"])

    if params["commission_model"] == "flat":
        flat_commission = float(params["flat_commission_amount"])
        commission_hold = np.full(len(frame), flat_commission, dtype=float)
        commission_worst = np.full(len(frame), flat_commission, dtype=float)
        commission_tp = np.full(len(frame), flat_commission, dtype=float)
        commission_tp_sl_capped = np.full(len(frame), flat_commission, dtype=float)
        commission_dynamic = np.full(len(frame), flat_commission, dtype=float)
        commission_ev = np.full(len(frame), flat_commission, dtype=float)
    else:
        # Hold-to-expiry wins (both shorts OTM at settlement) can expire worthless:
        # only opening commission is paid. All other outcomes are modeled as active
        # closes and pay round-trip commission.
        hold_expired_worthless = np.isfinite(short_put) & np.isfinite(short_call) & (close >= short_put) & (close <= short_call)
        commission_hold = np.where(hold_expired_worthless, open_commission, round_trip_commission)
        commission_worst = np.full(len(frame), round_trip_commission, dtype=float)
        commission_tp = np.full(len(frame), round_trip_commission, dtype=float)
        commission_tp_sl_capped = np.full(len(frame), round_trip_commission, dtype=float)
        commission_dynamic = np.where(dynamic_sl_annotations["dynamic_sl_triggered"], round_trip_commission, commission_hold)
        commission_ev = np.where(ev_exit_annotations["ev_based_exit_early_exit"], round_trip_commission, commission_hold)

    commission_hold = np.where(skipped, 0.0, commission_hold)
    commission_worst = np.where(skipped, 0.0, commission_worst)
    commission_tp = np.where(skipped, 0.0, commission_tp)
    commission_tp_sl_capped = np.where(skipped, 0.0, commission_tp_sl_capped)
    commission_dynamic = np.where(skipped, 0.0, commission_dynamic)
    commission_ev = np.where(skipped, 0.0, commission_ev)

    outcome_hold = np.where(skipped, np.nan, settlement_pre_commission - commission_hold)
    outcome_worst = np.where(skipped, np.nan, worst_pre_commission - commission_worst)
    outcome_tp = np.where(skipped, np.nan, tp_pre_commission - commission_tp)
    outcome_tp_sl_capped = np.where(skipped, np.nan, tp50_sl_capped_pre_commission - commission_tp_sl_capped)
    outcome_dynamic = np.where(skipped, np.nan, dynamic_sl_annotations["dynamic_sl_pre_commission"] - commission_dynamic)
    outcome_ev = np.where(skipped, np.nan, ev_exit_annotations["ev_based_exit_pre_commission"] - commission_ev)

    result = pd.DataFrame(
        {
            "date": frame["date"],
            "spx_open": S,
            "spx_high": high,
            "spx_low": low,
            "spx_close": close,
            "vix": entry_vix,
            "iv_scaling_factor": params["iv_scaling_factor"],
            "sigma": sigma,
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
            "dynamic_sl_pre_commission": dynamic_sl_annotations["dynamic_sl_pre_commission"],
            "ev_based_exit_pre_commission": ev_exit_annotations["ev_based_exit_pre_commission"],
            "pricing_mode": pricing_mode,
            "use_bs_repricing": np.full(len(frame), use_bs_repricing, dtype=bool),
            "settlement_pnl_usd": outcome_hold,
            "outcome_hold_to_expiry": outcome_hold,
            "outcome_worst_case": outcome_worst,
            "outcome_tp50_or_expiry": outcome_tp,
            "outcome_tp50_sl_capped": outcome_tp_sl_capped,
            "outcome_dynamic_sl": outcome_dynamic,
            "outcome_ev_based_exit": outcome_ev,
            "commission_open_usd": np.where(skipped, 0.0, open_commission),
            "commission_round_trip_usd": np.where(skipped, 0.0, round_trip_commission),
            "commission_hold_to_expiry": commission_hold,
            "commission_worst_case": commission_worst,
            "commission_tp50_or_expiry": commission_tp,
            "commission_tp50_sl_capped": commission_tp_sl_capped,
            "commission_dynamic_sl": commission_dynamic,
            "commission_ev_based_exit": commission_ev,
            "selection_method": np.where(skipped, "", "DELTA"),
            # Legacy commission field kept for compatibility with existing consumers.
            "commission_usd": commission_hold,
            "skipped": skipped,
            "skip_reason": reason,
            # Generic aliases for EV scenario diagnostics (requested by PR2 spec).
            "P_OTM_min": ev_exit_annotations["ev_based_exit_P_OTM_min"],
            "V_continue": ev_exit_annotations["ev_based_exit_V_continue"],
            "exit_reason": ev_exit_annotations["ev_based_exit_exit_reason"],
        }
    )
    for col_name, values in {**exit_annotations, **dynamic_sl_annotations, **ev_exit_annotations}.items():
        result[col_name] = values
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


def _with_pricing_mode(config: Dict[str, Any], use_bs_repricing: bool) -> Dict[str, Any]:
    """Return a config copy with pricing.use_bs_repricing forced to requested mode."""
    cfg = copy.deepcopy(config)
    cfg.setdefault("pricing", {})["use_bs_repricing"] = bool(use_bs_repricing)
    return cfg


def _scenario_sl_trigger_count(results: pd.DataFrame, scenario_name: str) -> int:
    """Count SL triggers for a scenario where applicable."""
    if scenario_name == "tp50_sl_capped" and "tp50_sl_capped_sl_trigger" in results.columns:
        return int(results["tp50_sl_capped_sl_trigger"].fillna(False).astype(bool).sum())
    if scenario_name == "dynamic_sl" and "dynamic_sl_triggered" in results.columns:
        return int(results["dynamic_sl_triggered"].fillna(False).astype(bool).sum())
    return 0


def _scenario_exit_meta_columns(scenario_name: str) -> Dict[str, str]:
    """Return per-scenario metadata columns used for pricing-mode diffs."""
    if scenario_name == "tp50_or_expiry":
        return {
            "exit_type": "tp50_or_expiry_exit_type",
            "spot": "tp50_or_expiry_trigger_spot",
            "dist": "tp50_or_expiry_distance_to_short_strike",
            "tte": "tp50_or_expiry_tte_at_trigger",
        }
    if scenario_name == "tp50_sl_capped":
        return {
            "exit_type": "tp50_sl_capped_exit_type",
            "spot": "tp50_sl_capped_trigger_spot",
            "dist": "tp50_sl_capped_distance_to_short_strike",
            "tte": "tp50_sl_capped_tte_at_trigger",
        }
    if scenario_name == "dynamic_sl":
        return {
            "exit_type": "dynamic_sl_exit_type",
            "spot": "dynamic_sl_trigger_spot",
            "dist": "dynamic_sl_distance_to_short_strike",
            "tte": "dynamic_sl_tte_at_trigger",
        }
    if scenario_name == "ev_based_exit":
        return {
            "exit_type": "ev_based_exit_exit_reason",
            "spot": "ev_based_exit_trigger_spot",
            "dist": "ev_based_exit_distance_to_short_strike",
            "tte": "ev_based_exit_tte_at_eval",
        }
    return {}


def _default_outcome_label_for_scenario(scenario_name: str, frame: pd.DataFrame, suffix: str = "") -> np.ndarray:
    """Fallback outcome labels for scenarios without explicit exit-type metadata."""
    n = len(frame)
    if scenario_name == "hold_to_expiry":
        labels = np.full(n, "EXPIRY", dtype=object)
    elif scenario_name == "worst_case":
        touched = (frame.get(f"put_breached{suffix}", pd.Series(False, index=frame.index)).fillna(False).astype(bool)) | (
            frame.get(f"call_breached{suffix}", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
        )
        labels = np.where(touched.to_numpy(), "WORST_TOUCH", "NO_TOUCH_MAXPROFIT").astype(object)
    else:
        labels = np.full(n, "", dtype=object)

    skipped_col = f"skipped{suffix}"
    if skipped_col in frame.columns:
        skipped = frame[skipped_col].fillna(False).astype(bool).to_numpy()
        labels = np.where(skipped, "", labels).astype(object)
    return labels


def build_bs_repricing_comparison_outputs(
    config: Dict[str, Any],
    data: Optional[pd.DataFrame] = None,
    output_dir: str | Path = "outputs",
) -> Dict[str, Any]:
    """
    Run linear-vs-BS repricing comparison and write requested CSV outputs.

    Returns dict with dataframes and output paths for downstream callers.
    """
    base_params = _backtest_params(config)
    if data is None:
        data = download_data(
            start_date=base_params["start_date"],
            end_date=base_params["end_date"],
            cache_file=base_params["cache_file"],
        )
    else:
        data = data.copy()

    cfg_linear = _with_pricing_mode(config, False)
    cfg_bs = _with_pricing_mode(config, True)

    linear_results = run_backtest(cfg_linear, data=data)
    bs_results = run_backtest(cfg_bs, data=data)

    linear_metrics = compute_metrics(linear_results, cfg_linear)
    bs_metrics = compute_metrics(bs_results, cfg_bs)

    sl_delta_by_scenario: Dict[str, int] = {}
    for scenario_name in SCENARIO_COLUMNS:
        sl_delta_by_scenario[scenario_name] = _scenario_sl_trigger_count(bs_results, scenario_name) - _scenario_sl_trigger_count(
            linear_results, scenario_name
        )

    comparison_rows: List[Dict[str, Any]] = []
    for pricing_mode, metrics in (("linear", linear_metrics), ("bs", bs_metrics)):
        results_frame = linear_results if pricing_mode == "linear" else bs_results
        for scenario_name in SCENARIO_COLUMNS:
            metric = metrics.get(scenario_name, {})
            comparison_rows.append(
                {
                    "scenario": scenario_name,
                    "pricing_mode": pricing_mode,
                    "total_pnl": float(metric.get("total_pnl_usd", 0.0)),
                    "win_rate": float(metric.get("win_rate", 0.0)),
                    "max_drawdown": float(metric.get("max_drawdown_usd", 0.0)),
                    "avg_loss": float(metric.get("avg_loss_usd", 0.0)),
                    "sharpe": float(metric.get("sharpe_daily", 0.0)),
                    "total_trades": int(metric.get("total_trades", 0)),
                    "sl_triggers": _scenario_sl_trigger_count(results_frame, scenario_name),
                    "sl_triggers_delta": int(sl_delta_by_scenario.get(scenario_name, 0)),
                }
            )

    comparison_df = pd.DataFrame(comparison_rows)

    linear_renamed = linear_results.copy()
    bs_renamed = bs_results.copy()
    linear_renamed = linear_renamed.add_suffix("_linear")
    bs_renamed = bs_renamed.add_suffix("_bs")
    merged = pd.merge(
        linear_renamed,
        bs_renamed,
        left_on="date_linear",
        right_on="date_bs",
        how="outer",
        validate="one_to_one",
    )
    if "date_linear" in merged.columns:
        merged["date"] = merged["date_linear"].combine_first(merged.get("date_bs"))

    diff_rows: List[Dict[str, Any]] = []
    pnl_tol = 1e-6

    for scenario_name, outcome_col in SCENARIO_COLUMNS.items():
        meta = _scenario_exit_meta_columns(scenario_name)
        lin_col = f"{outcome_col}_linear"
        bs_col = f"{outcome_col}_bs"
        if lin_col not in merged.columns or bs_col not in merged.columns:
            continue

        linear_pnl = pd.to_numeric(merged[lin_col], errors="coerce")
        bs_pnl = pd.to_numeric(merged[bs_col], errors="coerce")
        pnl_changed = ~np.isclose(linear_pnl.to_numpy(), bs_pnl.to_numpy(), atol=pnl_tol, rtol=0.0, equal_nan=True)

        if meta:
            lin_exit = merged.get(f"{meta['exit_type']}_linear", pd.Series("", index=merged.index)).fillna("").astype(str)
            bs_exit = merged.get(f"{meta['exit_type']}_bs", pd.Series("", index=merged.index)).fillna("").astype(str)
        else:
            lin_exit = pd.Series(_default_outcome_label_for_scenario(scenario_name, merged, suffix="_linear"), index=merged.index)
            bs_exit = pd.Series(_default_outcome_label_for_scenario(scenario_name, merged, suffix="_bs"), index=merged.index)

        exit_changed = lin_exit.to_numpy(dtype=object) != bs_exit.to_numpy(dtype=object)
        changed_mask = pnl_changed | exit_changed
        if not changed_mask.any():
            continue

        if meta:
            spot_lin = pd.to_numeric(merged.get(f"{meta['spot']}_linear"), errors="coerce")
            spot_bs = pd.to_numeric(merged.get(f"{meta['spot']}_bs"), errors="coerce")
            dist_lin = pd.to_numeric(merged.get(f"{meta['dist']}_linear"), errors="coerce")
            dist_bs = pd.to_numeric(merged.get(f"{meta['dist']}_bs"), errors="coerce")
            tte_lin = pd.to_numeric(merged.get(f"{meta['tte']}_linear"), errors="coerce")
            tte_bs = pd.to_numeric(merged.get(f"{meta['tte']}_bs"), errors="coerce")
            spot_out = spot_bs.combine_first(spot_lin)
            dist_out = dist_bs.combine_first(dist_lin)
            tte_out = tte_bs.combine_first(tte_lin)
        else:
            spot_out = pd.Series(np.nan, index=merged.index, dtype=float)
            dist_out = pd.Series(np.nan, index=merged.index, dtype=float)
            tte_out = pd.Series(np.nan, index=merged.index, dtype=float)

        vix_out = pd.to_numeric(merged.get("vix_bs"), errors="coerce").combine_first(pd.to_numeric(merged.get("vix_linear"), errors="coerce"))

        changed_idx = np.flatnonzero(changed_mask)
        for idx in changed_idx:
            diff_rows.append(
                {
                    "date": pd.to_datetime(merged.iloc[idx]["date"]).date() if pd.notna(merged.iloc[idx]["date"]) else pd.NaT,
                    "scenario": scenario_name,
                    "linear_outcome": lin_exit.iloc[idx],
                    "bs_outcome": bs_exit.iloc[idx],
                    "linear_pnl": float(linear_pnl.iloc[idx]) if pd.notna(linear_pnl.iloc[idx]) else np.nan,
                    "bs_pnl": float(bs_pnl.iloc[idx]) if pd.notna(bs_pnl.iloc[idx]) else np.nan,
                    "spot_at_trigger": float(spot_out.iloc[idx]) if pd.notna(spot_out.iloc[idx]) else np.nan,
                    "distance_to_short_strike": float(dist_out.iloc[idx]) if pd.notna(dist_out.iloc[idx]) else np.nan,
                    "VIX": float(vix_out.iloc[idx]) if pd.notna(vix_out.iloc[idx]) else np.nan,
                    "TTE_at_trigger": float(tte_out.iloc[idx]) if pd.notna(tte_out.iloc[idx]) else np.nan,
                }
            )

    trade_diff_df = pd.DataFrame(diff_rows)
    if not trade_diff_df.empty:
        trade_diff_df = trade_diff_df.sort_values(["date", "scenario"]).reset_index(drop=True)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = out_dir / "bs_repricing_comparison.csv"
    trade_diff_path = out_dir / "bs_repricing_trade_diff.csv"
    comparison_df.to_csv(comparison_path, index=False)
    trade_diff_df.to_csv(trade_diff_path, index=False)

    return {
        "comparison": comparison_df,
        "trade_diff": trade_diff_df,
        "linear_results": linear_results,
        "bs_results": bs_results,
        "comparison_path": comparison_path,
        "trade_diff_path": trade_diff_path,
    }


def build_ev_exit_outputs(
    config: Dict[str, Any],
    data: Optional[pd.DataFrame] = None,
    output_dir: str | Path = "outputs",
) -> Dict[str, Any]:
    """
    Run backtest (BS repricing default) and emit EV-exit comparison/detail CSVs.
    """
    params = _backtest_params(config)
    if data is None:
        data = download_data(
            start_date=params["start_date"],
            end_date=params["end_date"],
            cache_file=params["cache_file"],
        )
    else:
        data = data.copy()

    results = run_backtest(config, data=data)
    metrics = compute_metrics(results, config)

    comparison_rows: List[Dict[str, Any]] = []
    for scenario_name in SCENARIO_COLUMNS:
        m = metrics.get(scenario_name, {})
        early_exits = 0
        if scenario_name == "ev_based_exit" and "ev_based_exit_early_exit" in results.columns:
            early_exits = int(results["ev_based_exit_early_exit"].fillna(False).astype(bool).sum())
        comparison_rows.append(
            {
                "scenario": scenario_name,
                "total_pnl": float(m.get("total_pnl_usd", 0.0)),
                "win_rate": float(m.get("win_rate", 0.0)),
                "max_drawdown": float(m.get("max_drawdown_usd", 0.0)),
                "avg_win": float(m.get("avg_win_usd", 0.0)),
                "avg_loss": float(m.get("avg_loss_usd", 0.0)),
                "sharpe": float(m.get("sharpe_daily", 0.0)),
                "total_trades": int(m.get("total_trades", 0)),
                "early_exits": int(early_exits),
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)

    ev_rows = results[(~results["skipped"]) & results["outcome_ev_based_exit"].notna()].copy() if "outcome_ev_based_exit" in results.columns else pd.DataFrame()
    if not ev_rows.empty:
        ev_analysis_df = pd.DataFrame(
            {
                "date": pd.to_datetime(ev_rows["date"]).dt.date,
                "exit_reason": ev_rows["ev_based_exit_exit_reason"],
                "P_OTM_at_exit": ev_rows["ev_based_exit_P_OTM_min"],
                "V_continue_at_exit": ev_rows["ev_based_exit_V_continue"],
                "TTE_at_exit": ev_rows["ev_based_exit_tte_at_eval"],
                "pnl": ev_rows["outcome_ev_based_exit"],
                "VIX": ev_rows["vix"],
                "spot_at_exit": ev_rows["ev_based_exit_trigger_spot"],
                "distance_to_short": ev_rows["ev_based_exit_distance_to_short_strike"],
            }
        ).sort_values("date").reset_index(drop=True)
    else:
        ev_analysis_df = pd.DataFrame(
            columns=[
                "date",
                "exit_reason",
                "P_OTM_at_exit",
                "V_continue_at_exit",
                "TTE_at_exit",
                "pnl",
                "VIX",
                "spot_at_exit",
                "distance_to_short",
            ]
        )

    potm_hist_df = ev_analysis_df[ev_analysis_df["exit_reason"] == "EV_NEGATIVE"][["date", "P_OTM_at_exit"]].copy()
    potm_hist_df = potm_hist_df.rename(columns={"P_OTM_at_exit": "P_OTM_min"})

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = out_dir / "ev_exit_comparison.csv"
    analysis_path = out_dir / "ev_exit_analysis.csv"
    potm_hist_path = out_dir / "ev_exit_P_OTM_distribution.csv"

    comparison_df.to_csv(comparison_path, index=False)
    ev_analysis_df.to_csv(analysis_path, index=False)
    potm_hist_df.to_csv(potm_hist_path, index=False)

    return {
        "results": results,
        "metrics": metrics,
        "comparison": comparison_df,
        "ev_analysis": ev_analysis_df,
        "potm_distribution": potm_hist_df,
        "comparison_path": comparison_path,
        "analysis_path": analysis_path,
        "potm_hist_path": potm_hist_path,
    }


def build_dynamic_sl_outputs(
    config: Dict[str, Any],
    data: Optional[pd.DataFrame] = None,
    output_dir: str | Path = "outputs",
    run_sweep: bool = True,
) -> Dict[str, Any]:
    """
    Produce PR2b dynamic SL comparison and diagnostics CSV outputs.
    """
    params = _backtest_params(config)
    if data is None:
        data = download_data(
            start_date=params["start_date"],
            end_date=params["end_date"],
            cache_file=params["cache_file"],
        )
    else:
        data = data.copy()

    # Base run (should use BS repricing by default from config)
    base_results = run_backtest(config, data=data)
    base_metrics = compute_metrics(base_results, config)

    # Direct fixed 2x comparison = TP=100%, SL=2x under same BS pricing.
    fixed_cfg = copy.deepcopy(config)
    fixed_cfg.setdefault("default", {})["tp_pct"] = 1.0
    fixed_cfg.setdefault("default", {})["sl_mult"] = 2.0
    fixed_cfg.setdefault("pricing", {})["use_bs_repricing"] = True
    fixed_results = run_backtest(fixed_cfg, data=data)
    fixed_metrics = compute_metrics(fixed_results, fixed_cfg)

    comparison_rows: List[Dict[str, Any]] = []

    def _append_row(label: str, results_df: pd.DataFrame, metrics_dict: Dict[str, Dict[str, float]], scenario_key: str) -> None:
        m = metrics_dict.get(scenario_key, {})
        sl_triggers = _scenario_sl_trigger_count(results_df, scenario_key)
        total_trades = int(m.get("total_trades", 0))
        comparison_rows.append(
            {
                "scenario": label,
                "total_pnl": float(m.get("total_pnl_usd", 0.0)),
                "win_rate": float(m.get("win_rate", 0.0)),
                "max_drawdown": float(m.get("max_drawdown_usd", 0.0)),
                "avg_win": float(m.get("avg_win_usd", 0.0)),
                "avg_loss": float(m.get("avg_loss_usd", 0.0)),
                "sharpe": float(m.get("sharpe_daily", 0.0)),
                "total_trades": total_trades,
                "sl_triggers": sl_triggers,
                "sl_trigger_rate": (sl_triggers / total_trades) if total_trades else np.nan,
            }
        )

    for scen in ("hold_to_expiry", "tp50_sl_capped", "dynamic_sl"):
        if scen in base_metrics:
            _append_row(scen, base_results, base_metrics, scen)
    if "tp50_sl_capped" in fixed_metrics:
        _append_row("fixed_sl_2x", fixed_results, fixed_metrics, "tp50_sl_capped")

    comparison_df = pd.DataFrame(comparison_rows)

    dyn_rows = base_results[(~base_results["skipped"]) & base_results["outcome_dynamic_sl"].notna()].copy()
    if not dyn_rows.empty:
        trigger_analysis_df = pd.DataFrame(
            {
                "date": pd.to_datetime(dyn_rows["date"]).dt.date,
                "exit_type": dyn_rows["dynamic_sl_exit_type"],
                "P_OTM_at_exit": dyn_rows["dynamic_sl_P_OTM_at_trigger"],
                "sl_mult_at_exit": dyn_rows["dynamic_sl_mult_at_trigger"],
                "TTE_at_exit_hours": pd.to_numeric(dyn_rows["dynamic_sl_tte_at_trigger"], errors="coerce") * (TRADING_DAYS_PER_YEAR * TRADING_HOURS_PER_DAY),
                "pnl": dyn_rows["outcome_dynamic_sl"],
                "VIX": dyn_rows["vix"],
                "spot_at_exit": dyn_rows["dynamic_sl_trigger_spot"],
                "distance_to_short": dyn_rows["dynamic_sl_distance_to_short_strike"],
                "spread_cost_at_exit": dyn_rows["dynamic_sl_spread_cost_at_trigger"],
            }
        ).sort_values("date").reset_index(drop=True)
    else:
        trigger_analysis_df = pd.DataFrame(
            columns=[
                "date",
                "exit_type",
                "P_OTM_at_exit",
                "sl_mult_at_exit",
                "TTE_at_exit_hours",
                "pnl",
                "VIX",
                "spot_at_exit",
                "distance_to_short",
                "spread_cost_at_exit",
            ]
        )

    dyn_only = trigger_analysis_df[trigger_analysis_df["exit_type"] == "DYNAMIC_SL"]["P_OTM_at_exit"].dropna()
    stats_rows = []
    if len(dyn_only) > 0:
        stats_rows = [
            {"stat": "n", "value": float(len(dyn_only))},
            {"stat": "mean", "value": float(dyn_only.mean())},
            {"stat": "median", "value": float(dyn_only.median())},
            {"stat": "std", "value": float(dyn_only.std(ddof=0))},
            {"stat": "p10", "value": float(dyn_only.quantile(0.10))},
            {"stat": "p25", "value": float(dyn_only.quantile(0.25))},
            {"stat": "p50", "value": float(dyn_only.quantile(0.50))},
            {"stat": "p75", "value": float(dyn_only.quantile(0.75))},
            {"stat": "p90", "value": float(dyn_only.quantile(0.90))},
            {"stat": "min", "value": float(dyn_only.min())},
            {"stat": "max", "value": float(dyn_only.max())},
        ]
    potm_dist_df = pd.DataFrame(stats_rows, columns=["stat", "value"])

    sweep_df = pd.DataFrame()
    if run_sweep:
        floors = np.arange(1.0, 2.5 + 1e-9, 0.5)
        ceilings = np.arange(2.5, 6.0 + 1e-9, 0.5)
        alphas = np.arange(0.5, 2.0 + 1e-9, 0.5)
        betas = np.arange(0.0, 1.5 + 1e-9, 0.5)
        sweep_rows: List[Dict[str, Any]] = []
        for sl_floor in floors:
            for sl_ceiling in ceilings:
                if sl_ceiling <= sl_floor:
                    continue
                for alpha in alphas:
                    for beta in betas:
                        cfg = copy.deepcopy(config)
                        dyn = cfg.setdefault("dynamic_sl", {})
                        dyn["enabled"] = True
                        dyn["sl_floor"] = float(sl_floor)
                        dyn["sl_ceiling"] = float(sl_ceiling)
                        dyn["alpha"] = float(alpha)
                        dyn["beta"] = float(beta)
                        res = run_backtest(cfg, data=data)
                        met = compute_metrics(res, cfg).get("dynamic_sl", {})
                        sl_trig = int(res.get("dynamic_sl_triggered", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if "dynamic_sl_triggered" in res.columns else 0
                        trig_mask = res.get("dynamic_sl_triggered", pd.Series(False, index=res.index)).fillna(False).astype(bool) if not res.empty else pd.Series(dtype=bool)
                        if "dynamic_sl_mult_at_trigger" in res.columns and not res.empty and trig_mask.any():
                            avg_mult = float(pd.to_numeric(res.loc[trig_mask, "dynamic_sl_mult_at_trigger"], errors="coerce").dropna().mean())
                        else:
                            avg_mult = np.nan
                        sweep_rows.append(
                            {
                                "sl_floor": float(sl_floor),
                                "sl_ceiling": float(sl_ceiling),
                                "alpha": float(alpha),
                                "beta": float(beta),
                                "total_pnl": float(met.get("total_pnl_usd", 0.0)),
                                "win_rate": float(met.get("win_rate", 0.0)),
                                "max_drawdown": float(met.get("max_drawdown_usd", 0.0)),
                                "sharpe": float(met.get("sharpe_daily", 0.0)),
                                "sl_triggers": sl_trig,
                                "avg_sl_mult_at_trigger": avg_mult,
                            }
                        )
        sweep_df = pd.DataFrame(sweep_rows)
        if not sweep_df.empty:
            sweep_df = sweep_df.sort_values(["sharpe", "total_pnl"], ascending=[False, False]).reset_index(drop=True)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = out_dir / "dynamic_sl_comparison.csv"
    trigger_analysis_path = out_dir / "dynamic_sl_trigger_analysis.csv"
    potm_dist_path = out_dir / "dynamic_sl_P_OTM_distribution.csv"
    sweep_path = out_dir / "dynamic_sl_sweep.csv"

    comparison_df.to_csv(comparison_path, index=False)
    trigger_analysis_df.to_csv(trigger_analysis_path, index=False)
    potm_dist_df.to_csv(potm_dist_path, index=False)
    if run_sweep:
        sweep_df.to_csv(sweep_path, index=False)

    return {
        "base_results": base_results,
        "base_metrics": base_metrics,
        "fixed_results": fixed_results,
        "fixed_metrics": fixed_metrics,
        "comparison": comparison_df,
        "trigger_analysis": trigger_analysis_df,
        "potm_distribution": potm_dist_df,
        "sweep": sweep_df,
        "comparison_path": comparison_path,
        "trigger_analysis_path": trigger_analysis_path,
        "potm_dist_path": potm_dist_path,
        "sweep_path": sweep_path,
    }


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


def metrics_to_dataframe(metrics: Dict[str, Dict[str, float]], config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Convert nested metrics dict to a display-friendly dataframe."""
    df = pd.DataFrame(metrics).T
    if config is not None:
        params = _backtest_params(config)
        df["iv_scaling_factor"] = float(params["iv_scaling_factor"])
    return df


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
        dyn_floor = float(combo_params.get("sl_floor", params_base.get("dynamic_sl_floor", 1.5)))
        dyn_ceiling = float(combo_params.get("sl_ceiling", params_base.get("dynamic_sl_ceiling", 4.0)))

        row_base = dict(combo_params)
        row_base.setdefault("iv_scaling_factor", float(combo_params.get("iv_scaling_factor", params_base["iv_scaling_factor"])))

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

        # Invalid combo 4b: dynamic SL ceiling must exceed floor.
        if ("sl_floor" in combo_params or "sl_ceiling" in combo_params) and (dyn_ceiling <= dyn_floor):
            row_base["combo_skipped"] = True
            row_base["combo_skip_reason"] = "dynamic_sl_ceiling_lte_floor"
            rows.append(row_base)
            continue

        # Invalid combo 5: structurally impossible required win rate.
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
        if "iv_scaling_factor" in combo_params:
            cfg["iv_scaling_factor"] = float(combo_params["iv_scaling_factor"])
        if any(k in combo_params for k in ("sl_floor", "sl_ceiling", "alpha", "beta")):
            dyn_cfg = cfg.setdefault("dynamic_sl", {})
            if "sl_floor" in combo_params:
                dyn_cfg["sl_floor"] = float(combo_params["sl_floor"])
            if "sl_ceiling" in combo_params:
                dyn_cfg["sl_ceiling"] = float(combo_params["sl_ceiling"])
            if "alpha" in combo_params:
                dyn_cfg["alpha"] = float(combo_params["alpha"])
            if "beta" in combo_params:
                dyn_cfg["beta"] = float(combo_params["beta"])

        results = run_backtest(cfg, data=data)
        metrics = compute_metrics(results, cfg)

        out = dict(combo_params)
        out.setdefault("iv_scaling_factor", float(_backtest_params(cfg)["iv_scaling_factor"]))
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
