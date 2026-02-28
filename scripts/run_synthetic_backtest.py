"""CLI entry point for synthetic 0DTE Iron Condor backtesting."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

from src.research.synthetic_backtest import (
    _backtest_params,
    compute_metrics,
    download_data,
    metrics_to_dataframe,
    run_backtest,
    run_parameter_sweep,
    test_bs_pricer,
    test_settlement_pnl,
    validate_data,
)
from src.research.visualizations import generate_standard_charts, generate_sweep_charts


def _fmt_money(value: float) -> str:
    """Format numeric value as USD string."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "N/A"
    return f"${value:,.2f}"


def _fmt_pct(value: float) -> str:
    """Format decimal ratio as percentage string."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "N/A"
    return f"{value * 100:.1f}%"


def _print_header(start: str, end: str) -> None:
    """Print CLI banner."""
    print("\n" + "═" * 59)
    print("  SYNTHETIC 0DTE IRON CONDOR BACKTEST")
    print(f"  SPX OHLC + VIX | {start} to {end}")
    print("═" * 59)


def _print_config_summary(params: Dict[str, Any]) -> None:
    """Print key config knobs used in this run."""
    print("\n🔧 CONFIGURATION")
    print(
        f"   Wing Width: {params['wing_width']:.2f} | Delta: {params['target_delta']:.2f} | "
        f"Min Credit: ${params['min_credit']:.2f}"
    )
    print(
        f"   TP: {params['take_profit_pct']:.0%} | SL: {params['stop_loss_mult']:.1f}x | "
        f"Commission Model: {params['commission_model']}"
    )
    print(
        f"   Open Comm: ${params['open_commission']:.2f} | "
        f"Round-trip Comm: ${params['round_trip_commission']:.2f} | "
        f"Pricing Plan: {params['pricing_plan']}"
    )
    print(f"   Bid/Ask Haircut: {params['bid_ask_haircut']:.0%}")
    print(f"   IV Scaling Factor (k): {params['iv_scaling_factor']:.3f}")


def _print_metrics_table(metrics: Dict[str, Dict[str, float]]) -> None:
    """Print scenario comparison table."""
    order = ["hold_to_expiry", "worst_case", "tp50_or_expiry", "tp50_sl_capped", "dynamic_sl", "ev_based_exit"]
    labels = {
        "hold_to_expiry": "Hold-to-Expiry",
        "worst_case": "Worst Case",
        "tp50_or_expiry": "TP50/Expiry",
        "tp50_sl_capped": "TP50+SL Cap",
        "dynamic_sl": "Dynamic SL",
        "ev_based_exit": "EV Exit",
    }
    scenario_order = [s for s in order if s in metrics]

    def col(s: str, key: str) -> float:
        return metrics.get(s, {}).get(key, np.nan)

    def format_int(value: float) -> str:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "N/A"
        return str(int(value))

    def format_num(value: float) -> str:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "N/A"
        return f"{value:.2f}"

    def print_row(metric_label: str, metric_key: str, formatter) -> None:
        values = [formatter(col(s, metric_key)) for s in scenario_order]
        print(f"{metric_label:24} " + " ".join(f"{v:>14}" for v in values))

    print("\n📈 RESULTS BY SCENARIO")
    print(f"{'':24} " + " ".join(f"{labels[s]:>14}" for s in scenario_order))
    print("   " + "─" * (24 + 15 * len(scenario_order)))
    print_row("Trades Executed:", "total_trades", format_int)
    print_row("Trades Skipped:", "total_skipped", format_int)
    print_row("Win Rate:", "win_rate", _fmt_pct)
    print_row("Breakeven WR:", "breakeven_winrate", _fmt_pct)
    print_row("WR Margin:", "winrate_margin", _fmt_pct)
    print_row("Total PnL:", "total_pnl_usd", _fmt_money)
    print_row("Avg PnL/Trade:", "avg_pnl_per_trade_usd", _fmt_money)
    print_row("Sharpe (daily):", "sharpe_daily", format_num)
    print_row("Max Drawdown:", "max_drawdown_usd", _fmt_money)
    print_row("Max DD % of Capital:", "max_drawdown_pct", lambda v: f"{v:.1f}%" if np.isfinite(v) else "N/A")
    print_row("Worst Day:", "worst_day_usd", _fmt_money)
    print_row("CVaR 95%:", "cvar_95_usd", _fmt_money)
    print_row("Total Commissions:", "total_commissions_usd", _fmt_money)
    print_row("Commission % Gross:", "commission_pct_of_gross", _fmt_pct)


def _single_run_verdict(metrics: Dict[str, Dict[str, float]]) -> str:
    """Produce a coarse verdict for one backtest run."""
    hold = metrics["hold_to_expiry"]["total_pnl_usd"] > 0
    worst = metrics["worst_case"]["total_pnl_usd"] > 0
    tp = metrics["tp50_or_expiry"]["total_pnl_usd"] > 0
    tp_sl = metrics.get("tp50_sl_capped", {}).get("total_pnl_usd", -np.inf) > 0

    if hold and worst and tp and tp_sl:
        return "✅ VERDICT: STRATEGY HAS EDGE (all scenarios positive)"
    if not hold and not worst and not tp and not tp_sl:
        return "❌ VERDICT: NO EDGE FOUND (all scenarios negative)"
    return "⚠️ VERDICT: MARGINAL EDGE (scenario-dependent)"


def _sweep_verdict(sweep_df: pd.DataFrame) -> str:
    """Produce edge verdict from sweep robustness criteria."""
    if sweep_df.empty:
        return "❌ VERDICT: NO EDGE FOUND (no sweep results)"

    if "combo_skipped" in sweep_df.columns:
        mask = sweep_df["combo_skipped"] == False  # noqa: E712
    else:
        return "❌ VERDICT: NO EDGE FOUND (missing combo_skipped flag)"
    valid = sweep_df[mask].copy()
    if valid.empty:
        return "❌ VERDICT: NO EDGE FOUND (all combinations invalid)"

    needed = [
        "hold_to_expiry_total_pnl_usd",
        "worst_case_total_pnl_usd",
        "tp50_or_expiry_total_pnl_usd",
        "tp50_sl_capped_total_pnl_usd",
        "tp50_sl_capped_sharpe_daily",
        "tp50_sl_capped_winrate_margin",
    ]
    missing = [c for c in needed if c not in valid.columns]
    if missing:
        return "❌ VERDICT: NO EDGE FOUND (missing sweep metrics)"

    viable = valid[
        (valid["hold_to_expiry_total_pnl_usd"] > 0)
        & (valid["worst_case_total_pnl_usd"] > 0)
        & (valid["tp50_or_expiry_total_pnl_usd"] > 0)
        & (valid["tp50_sl_capped_total_pnl_usd"] > 0)
        & (valid["tp50_sl_capped_sharpe_daily"] > 0.5)
        & (valid["tp50_sl_capped_winrate_margin"] > 0)
    ]

    if len(viable) >= 3:
        return "✅ VERDICT: STRATEGY HAS EDGE (>=3 robust configs)"
    return "❌ VERDICT: NO EDGE FOUND (robust criteria not met)"


def _load_config(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """Run synthetic backtest CLI workflow."""
    parser = argparse.ArgumentParser(
        description="Synthetic 0DTE Iron Condor Backtester",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/backtest_config.yaml")
    parser.add_argument("--sweep", action="store_true", help="Run full parameter sweep")
    parser.add_argument("--start", type=str, default=None, help="Override start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="Override end date YYYY-MM-DD")
    parser.add_argument("--output-dir", type=str, default="outputs/backtest_results")
    parser.add_argument("--run-tests", action="store_true", help="Run pricing/settlement self-tests before backtest")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = _load_config(config_path)

    if args.start:
        config.setdefault("dates", {})["start"] = args.start
        config.setdefault("data", {})["start_date"] = args.start
    if args.end:
        config.setdefault("dates", {})["end"] = args.end
        config.setdefault("data", {})["end_date"] = args.end

    seed = int(config.get("random_seed", 42))
    np.random.seed(seed)

    params = _backtest_params(config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _print_header(params["start_date"], params["end_date"])

    if args.run_tests:
        print("\n🧪 Running internal tests...")
        test_bs_pricer()
        test_settlement_pnl()
        print("   Tests passed.")

    data = download_data(
        start_date=params["start_date"],
        end_date=params["end_date"],
        cache_file=params["cache_file"],
    )
    validate_data(data)
    _print_config_summary(params)

    results = run_backtest(config, data=data)
    metrics = compute_metrics(results, config)
    _print_metrics_table(metrics)

    results_csv = output_dir / "backtest_results.csv"
    results.to_csv(results_csv, index=False)
    metrics_df = metrics_to_dataframe(metrics, config)
    metrics_csv = output_dir / "metrics_summary.csv"
    metrics_df.to_csv(metrics_csv, index=True)

    chart_files = generate_standard_charts(
        results=results,
        metrics=metrics,
        config=config,
        output_dir=str(output_dir),
        start=params["start_date"],
        end=params["end_date"],
    )

    print("\n" + _single_run_verdict(metrics))

    if args.sweep:
        print("\n🔁 Running parameter sweep (this may take several minutes)...")
        sweep_grid = config.get("sweep", {})
        sweep_df = run_parameter_sweep(config, sweep_grid, data=data)
        sweep_csv = output_dir / "sweep_results.csv"
        sweep_df.to_csv(sweep_csv, index=False)
        sweep_charts = generate_sweep_charts(sweep_df, config, str(output_dir))
        print("\n" + _sweep_verdict(sweep_df))
        print(f"Sweep CSV saved to: {sweep_csv}")
        if not sweep_df.empty:
            sort_col = "tp50_sl_capped_total_pnl_usd" if "tp50_sl_capped_total_pnl_usd" in sweep_df.columns else "tp50_or_expiry_total_pnl_usd"
            label = "TP50+SL Capped PnL" if sort_col == "tp50_sl_capped_total_pnl_usd" else "TP50 PnL"
            print(f"\nTop 10 viable configurations (by {label}):")
            valid = sweep_df[sweep_df["combo_skipped"] == False]  # noqa: E712
            if sort_col in valid.columns:
                valid = valid.sort_values(sort_col, ascending=False)
            valid = valid.head(10)
            print(valid.to_string(index=False))
        for name, file_path in sweep_charts.items():
            if file_path:
                print(f"{name}: {file_path}")

    print(f"\nCharts saved to: {output_dir}")
    print(f"Results CSV saved to: {results_csv}")
    print(f"Metrics CSV saved to: {metrics_csv}")


if __name__ == "__main__":
    main()
