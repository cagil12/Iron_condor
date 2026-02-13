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
        f"Commission: ${params['commission_per_trade']:.2f}/trade"
    )
    print(f"   Bid/Ask Haircut: {params['bid_ask_haircut']:.0%}")


def _print_metrics_table(metrics: Dict[str, Dict[str, float]]) -> None:
    """Print scenario comparison table."""
    order = ["hold_to_expiry", "worst_case", "tp50_or_expiry"]
    labels = {
        "hold_to_expiry": "Hold-to-Expiry",
        "worst_case": "Worst Case",
        "tp50_or_expiry": "TP50/Expiry",
    }

    def col(s: str, key: str) -> float:
        return metrics[s].get(key, np.nan)

    print("\n📈 RESULTS BY SCENARIO")
    print(f"{'':24} {'Hold-to-Expiry':>16} {'Worst Case':>14} {'TP50/Expiry':>14}")
    print("   " + "─" * 64)
    print(
        f"{'Trades Executed:':24} "
        f"{int(col(order[0], 'total_trades')):>16} "
        f"{int(col(order[1], 'total_trades')):>14} "
        f"{int(col(order[2], 'total_trades')):>14}"
    )
    print(
        f"{'Trades Skipped:':24} "
        f"{int(col(order[0], 'total_skipped')):>16} "
        f"{int(col(order[1], 'total_skipped')):>14} "
        f"{int(col(order[2], 'total_skipped')):>14}"
    )
    print(
        f"{'Win Rate:':24} "
        f"{_fmt_pct(col(order[0], 'win_rate')):>16} "
        f"{_fmt_pct(col(order[1], 'win_rate')):>14} "
        f"{_fmt_pct(col(order[2], 'win_rate')):>14}"
    )
    print(
        f"{'Breakeven WR:':24} "
        f"{_fmt_pct(col(order[0], 'breakeven_winrate')):>16} "
        f"{_fmt_pct(col(order[1], 'breakeven_winrate')):>14} "
        f"{_fmt_pct(col(order[2], 'breakeven_winrate')):>14}"
    )
    print(
        f"{'WR Margin:':24} "
        f"{_fmt_pct(col(order[0], 'winrate_margin')):>16} "
        f"{_fmt_pct(col(order[1], 'winrate_margin')):>14} "
        f"{_fmt_pct(col(order[2], 'winrate_margin')):>14}"
    )
    print(
        f"{'Total PnL:':24} "
        f"{_fmt_money(col(order[0], 'total_pnl_usd')):>16} "
        f"{_fmt_money(col(order[1], 'total_pnl_usd')):>14} "
        f"{_fmt_money(col(order[2], 'total_pnl_usd')):>14}"
    )
    print(
        f"{'Avg PnL/Trade:':24} "
        f"{_fmt_money(col(order[0], 'avg_pnl_per_trade_usd')):>16} "
        f"{_fmt_money(col(order[1], 'avg_pnl_per_trade_usd')):>14} "
        f"{_fmt_money(col(order[2], 'avg_pnl_per_trade_usd')):>14}"
    )
    print(
        f"{'Sharpe (daily):':24} "
        f"{col(order[0], 'sharpe_daily'):>16.2f} "
        f"{col(order[1], 'sharpe_daily'):>14.2f} "
        f"{col(order[2], 'sharpe_daily'):>14.2f}"
    )
    print(
        f"{'Max Drawdown:':24} "
        f"{_fmt_money(col(order[0], 'max_drawdown_usd')):>16} "
        f"{_fmt_money(col(order[1], 'max_drawdown_usd')):>14} "
        f"{_fmt_money(col(order[2], 'max_drawdown_usd')):>14}"
    )
    print(
        f"{'Max DD % of Capital:':24} "
        f"{col(order[0], 'max_drawdown_pct'):>15.1f}% "
        f"{col(order[1], 'max_drawdown_pct'):>13.1f}% "
        f"{col(order[2], 'max_drawdown_pct'):>13.1f}%"
    )
    print(
        f"{'Worst Day:':24} "
        f"{_fmt_money(col(order[0], 'worst_day_usd')):>16} "
        f"{_fmt_money(col(order[1], 'worst_day_usd')):>14} "
        f"{_fmt_money(col(order[2], 'worst_day_usd')):>14}"
    )
    print(
        f"{'CVaR 95%:':24} "
        f"{_fmt_money(col(order[0], 'cvar_95_usd')):>16} "
        f"{_fmt_money(col(order[1], 'cvar_95_usd')):>14} "
        f"{_fmt_money(col(order[2], 'cvar_95_usd')):>14}"
    )
    print(
        f"{'Total Commissions:':24} "
        f"{_fmt_money(col(order[0], 'total_commissions_usd')):>16} "
        f"{_fmt_money(col(order[1], 'total_commissions_usd')):>14} "
        f"{_fmt_money(col(order[2], 'total_commissions_usd')):>14}"
    )
    print(
        f"{'Commission % Gross:':24} "
        f"{_fmt_pct(col(order[0], 'commission_pct_of_gross')):>16} "
        f"{_fmt_pct(col(order[1], 'commission_pct_of_gross')):>14} "
        f"{_fmt_pct(col(order[2], 'commission_pct_of_gross')):>14}"
    )


def _single_run_verdict(metrics: Dict[str, Dict[str, float]]) -> str:
    """Produce a coarse verdict for one backtest run."""
    hold = metrics["hold_to_expiry"]["total_pnl_usd"] > 0
    worst = metrics["worst_case"]["total_pnl_usd"] > 0
    tp = metrics["tp50_or_expiry"]["total_pnl_usd"] > 0

    if hold and worst and tp:
        return "✅ VERDICT: STRATEGY HAS EDGE (all scenarios positive)"
    if not hold and not worst and not tp:
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
        "tp50_or_expiry_sharpe_daily",
        "tp50_or_expiry_winrate_margin",
    ]
    missing = [c for c in needed if c not in valid.columns]
    if missing:
        return "❌ VERDICT: NO EDGE FOUND (missing sweep metrics)"

    viable = valid[
        (valid["hold_to_expiry_total_pnl_usd"] > 0)
        & (valid["worst_case_total_pnl_usd"] > 0)
        & (valid["tp50_or_expiry_total_pnl_usd"] > 0)
        & (valid["tp50_or_expiry_sharpe_daily"] > 0.5)
        & (valid["tp50_or_expiry_winrate_margin"] > 0)
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
        config.setdefault("data", {})["start_date"] = args.start
    if args.end:
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
    metrics_df = metrics_to_dataframe(metrics)
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
        sweep_csv = output_dir / "parameter_sweep_results.csv"
        sweep_df.to_csv(sweep_csv, index=False)
        sweep_charts = generate_sweep_charts(sweep_df, config, str(output_dir))
        print("\n" + _sweep_verdict(sweep_df))
        print(f"Sweep CSV saved to: {sweep_csv}")
        if not sweep_df.empty:
            print("\nTop 10 viable configurations (by TP50 PnL):")
            valid = sweep_df[sweep_df["combo_skipped"] == False].head(10)  # noqa: E712
            print(valid.to_string(index=False))
        for name, file_path in sweep_charts.items():
            if file_path:
                print(f"{name}: {file_path}")

    print(f"\nCharts saved to: {output_dir}")
    print(f"Results CSV saved to: {results_csv}")
    print(f"Metrics CSV saved to: {metrics_csv}")


if __name__ == "__main__":
    main()
