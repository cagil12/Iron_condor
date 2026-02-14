"""Calibrate global VIX->0DTE IV scaling factor from live journal fills."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.research.synthetic_backtest import _backtest_params, download_data, run_backtest


def _print_no_trades_message() -> None:
    print("⚠️ No real trades found. Cannot calibrate iv_scaling_factor.")
    print("   Run this script after accumulating 10+ trades from Phase 1 live trading.")
    print("   Current default: iv_scaling_factor = 1.0 (no scaling)")


def _load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_date_column(df: pd.DataFrame) -> pd.Series:
    for col in ("timestamp", "entry_time", "date"):
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().any():
                return parsed
    return pd.Series(pd.NaT, index=df.index)


def _load_journal(journal_path: Path) -> pd.DataFrame:
    if not journal_path.exists():
        return pd.DataFrame()

    raw = pd.read_csv(journal_path)
    if raw.empty:
        return pd.DataFrame()

    ts = _parse_date_column(raw)
    credits = pd.to_numeric(raw.get("entry_credit"), errors="coerce")
    wing_width = pd.to_numeric(raw.get("wing_width"), errors="coerce")
    target_delta = pd.to_numeric(raw.get("target_delta"), errors="coerce")
    trade_id = raw.get("trade_id", pd.Series(np.arange(len(raw)) + 1, index=raw.index))

    clean = pd.DataFrame(
        {
            "trade_id": trade_id,
            "timestamp": ts,
            "trade_date": ts.dt.normalize(),
            "entry_credit": credits,
            "wing_width": wing_width,
            "target_delta": target_delta,
        }
    )
    clean = clean[clean["timestamp"].notna()].copy()
    clean = clean[np.isfinite(clean["entry_credit"]) & (clean["entry_credit"] > 0)].copy()
    clean = clean.sort_values("timestamp").reset_index(drop=True)
    return clean


def _synthetic_credit_for_trade(
    trade: pd.Series,
    trade_day: pd.Timestamp,
    daily_row: pd.Series,
    base_config: Dict[str, Any],
) -> Tuple[float, str]:
    cfg = copy.deepcopy(base_config)
    day = pd.Timestamp(trade_day).normalize()

    # Force one-day run and avoid look-ahead ambiguity for calibration.
    cfg.setdefault("dates", {})["start"] = day.strftime("%Y-%m-%d")
    cfg.setdefault("dates", {})["end"] = day.strftime("%Y-%m-%d")
    cfg["iv_scaling_factor"] = 1.0
    cfg.setdefault("timing", {})["vix_source"] = "vix_open"

    # Disable entry filters so each real trade gets a synthetic quote.
    cfg.setdefault("entry_filters", {})["min_vix"] = 0.0
    cfg.setdefault("entry_filters", {})["max_vix"] = 1_000.0
    cfg.setdefault("credit_filters", {})["max_risk_reward"] = 1_000.0
    cfg.setdefault("default", {})["min_credit"] = 0.0

    if np.isfinite(trade.get("wing_width", np.nan)) and float(trade["wing_width"]) > 0:
        cfg.setdefault("default", {})["wing_width"] = float(trade["wing_width"])

    if np.isfinite(trade.get("target_delta", np.nan)) and float(trade["target_delta"]) > 0:
        cfg.setdefault("default", {})["delta_target"] = float(trade["target_delta"])

    one_day = pd.DataFrame(
        [
            {
                "date": day,
                "spx_open": float(daily_row["spx_open"]),
                "spx_high": float(daily_row["spx_high"]),
                "spx_low": float(daily_row["spx_low"]),
                "spx_close": float(daily_row["spx_close"]),
                # Spec requirement: use daily VIX close for calibration baseline.
                "vix_open": float(daily_row["vix_close"]),
                "vix_high": float(daily_row["vix_close"]),
                "vix_low": float(daily_row["vix_close"]),
                "vix_close": float(daily_row["vix_close"]),
            }
        ]
    )
    simulated = run_backtest(cfg, data=one_day)
    if simulated.empty:
        return np.nan, "empty_simulation"

    row = simulated.iloc[0]
    if bool(row.get("skipped", False)):
        return np.nan, str(row.get("skip_reason", "skipped"))

    syn_credit = float(row.get("entry_credit", np.nan))
    if not np.isfinite(syn_credit) or syn_credit <= 0:
        return np.nan, "nonpositive_synthetic_credit"
    return syn_credit, ""


def _plot_scatter(calib: pd.DataFrame, output_dir: Path) -> Path:
    file_path = output_dir / "iv_calibration_scatter.png"
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(calib["synthetic_credit"], calib["real_credit"], alpha=0.75, color="#1f77b4", edgecolor="white")

    lo = float(min(calib["synthetic_credit"].min(), calib["real_credit"].min()))
    hi = float(max(calib["synthetic_credit"].max(), calib["real_credit"].max()))
    lo = min(lo, 0.0)
    hi = max(hi, 0.05)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#d62728", linewidth=1.6, label="y=x")

    k_median = float(calib["k_ratio"].median())
    ax.plot([lo, hi], [k_median * lo, k_median * hi], linestyle="-.", color="#2ca02c", linewidth=1.6, label=f"y={k_median:.3f}x")
    ax.set_title("IV Calibration: Real Credit vs Synthetic Credit (k=1 baseline)")
    ax.set_xlabel("Synthetic credit")
    ax.set_ylabel("Real credit")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(file_path, dpi=150)
    plt.close(fig)
    return file_path


def _plot_k_timeseries(calib: pd.DataFrame, output_dir: Path) -> Path:
    file_path = output_dir / "iv_calibration_k_timeseries.png"
    ts = calib.sort_values("timestamp").copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ts["timestamp"], ts["k_ratio"], marker="o", linewidth=1.4, color="#9467bd")
    k_median = float(ts["k_ratio"].median())
    ax.axhline(k_median, linestyle="--", color="#2ca02c", linewidth=1.4, label=f"Median k={k_median:.3f}")
    ax.axhline(1.0, linestyle=":", color="black", linewidth=1.2, label="k=1.0")
    ax.set_title("Per-trade IV Scaling Ratio Over Time")
    ax.set_xlabel("Trade timestamp")
    ax.set_ylabel("k_i = real_credit / synthetic_credit")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(file_path, dpi=150)
    plt.close(fig)
    return file_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate VIX->0DTE IV scaling factor from live journal")
    parser.add_argument("--config", type=str, default="configs/backtest_config.yaml")
    parser.add_argument("--journal", type=str, default="data/trade_journal.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/backtest_results")
    args = parser.parse_args()

    config_path = Path(args.config)
    journal_path = Path(args.journal)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading config and journal...")
    config = _load_config(config_path)
    journal = _load_journal(journal_path)
    if journal.empty:
        _print_no_trades_message()
        return

    start_date = journal["trade_date"].min().strftime("%Y-%m-%d")
    end_date = journal["trade_date"].max().strftime("%Y-%m-%d")
    params = _backtest_params(config)
    print(f"[2/5] Loading SPX/VIX daily data for {start_date} -> {end_date} ...")
    daily = download_data(start_date=start_date, end_date=end_date, cache_file=params["cache_file"])
    if daily.empty:
        print("⚠️ Daily data is empty. Cannot run calibration.")
        return
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    by_date = daily.set_index("date")

    print(f"[3/5] Calibrating on {len(journal)} candidate journal trades...")
    rows = []
    skipped = 0
    for i, trade in journal.iterrows():
        if i % 5 == 0 or i == len(journal) - 1:
            print(f"   Progress: {i + 1}/{len(journal)}")

        day = pd.Timestamp(trade["trade_date"]).normalize()
        if day not in by_date.index:
            skipped += 1
            continue

        day_row = by_date.loc[day]
        if isinstance(day_row, pd.DataFrame):
            day_row = day_row.iloc[0]

        synthetic_credit, reason = _synthetic_credit_for_trade(trade, day, day_row, config)
        if not np.isfinite(synthetic_credit) or synthetic_credit <= 0:
            skipped += 1
            continue

        real_credit = float(trade["entry_credit"])
        k_ratio = real_credit / synthetic_credit
        rows.append(
            {
                "trade_id": trade.get("trade_id", i + 1),
                "timestamp": pd.Timestamp(trade["timestamp"]),
                "date": day,
                "real_credit": real_credit,
                "synthetic_credit": synthetic_credit,
                "k_ratio": k_ratio,
                "wing_width": float(trade["wing_width"]) if np.isfinite(trade.get("wing_width", np.nan)) else np.nan,
                "target_delta": float(trade["target_delta"]) if np.isfinite(trade.get("target_delta", np.nan)) else np.nan,
                "vix_close": float(day_row["vix_close"]),
                "skip_reason": reason,
            }
        )

    calib = pd.DataFrame(rows)
    if calib.empty:
        print("⚠️ No valid calibration pairs found after filtering.")
        _print_no_trades_message()
        return

    print("[4/5] Computing calibration statistics...")
    k_mean = float(calib["k_ratio"].mean())
    k_median = float(calib["k_ratio"].median())
    k_std = float(calib["k_ratio"].std(ddof=0))
    k_min = float(calib["k_ratio"].min())
    k_max = float(calib["k_ratio"].max())
    n = int(len(calib))

    print("\n════════════════════════════════════════════")
    print("IV SCALING FACTOR CALIBRATION")
    print("════════════════════════════════════════════")
    print(f"Trades used: {n}")
    print(f"Trades skipped: {skipped}")
    print(f"k_mean:   {k_mean:.4f}")
    print(f"k_median: {k_median:.4f}")
    print(f"k_std:    {k_std:.4f}")
    print(f"k_min:    {k_min:.4f}")
    print(f"k_max:    {k_max:.4f}")
    print(f"Recommendation: iv_scaling_factor = {k_median:.3f} (median)")
    if n < 10:
        print("⚠️ Insufficient sample — treat calibration as preliminary")
    if k_std > 0.3:
        print("⚠️ High variance in scaling factor — VIX may not be a stable proxy for 0DTE IV")

    samples_csv = output_dir / "iv_calibration_samples.csv"
    calib.sort_values("timestamp").to_csv(samples_csv, index=False)

    print("[5/5] Saving plots and samples...")
    scatter_png = _plot_scatter(calib, output_dir)
    ts_png = _plot_k_timeseries(calib, output_dir)
    print(f"Saved samples: {samples_csv}")
    print(f"Saved scatter plot: {scatter_png}")
    print(f"Saved time-series plot: {ts_png}")


if __name__ == "__main__":
    main()
