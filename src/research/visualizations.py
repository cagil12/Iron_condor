"""Visualization utilities for synthetic 0DTE backtest outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from .synthetic_backtest import (
    PRE_COMMISSION_COLUMNS,
    SCENARIO_COLUMNS,
    breakeven_winrate,
)


def _ensure_dir(output_dir: str) -> Path:
    """Create output directory if missing and return Path."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_equity_curves(results: pd.DataFrame, output_dir: str, start: str, end: str) -> Path:
    out = _ensure_dir(output_dir)
    df = results.copy()
    df["date"] = pd.to_datetime(df["date"])

    hold_eq = df["outcome_hold_to_expiry"].fillna(0.0).cumsum()
    worst_eq = df["outcome_worst_case"].fillna(0.0).cumsum()
    tp_eq = df["outcome_tp50_or_expiry"].fillna(0.0).cumsum()

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(df["date"], hold_eq, label="Hold to Expiry", color="#1f77b4", linewidth=1.8)
    ax.plot(df["date"], worst_eq, label="Worst Case", color="#d62728", linewidth=1.4)
    ax.plot(df["date"], tp_eq, label="TP50 or Expiry", color="#2ca02c", linewidth=1.6)
    ax.fill_between(df["date"], worst_eq, hold_eq, color="#9ecae1", alpha=0.25, label="Uncertainty Band")
    ax.axhline(0, color="black", linewidth=1.0, linestyle="--")
    ax.set_title(f"Synthetic IC Backtest: Equity Curves ({start} to {end})")
    ax.set_ylabel("Cumulative PnL (USD)")
    ax.set_xlabel("Date")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    file_path = out / "equity_curves.png"
    fig.savefig(file_path, dpi=140)
    plt.close(fig)
    return file_path


def plot_monthly_pnl_heatmap(results: pd.DataFrame, output_dir: str) -> Path:
    out = _ensure_dir(output_dir)
    df = results.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    scenarios = [
        ("outcome_hold_to_expiry", "Hold"),
        ("outcome_worst_case", "Worst"),
        ("outcome_tp50_or_expiry", "TP50"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, (col, title) in zip(axes, scenarios):
        monthly = df.groupby(["year", "month"])[col].sum().unstack(fill_value=0.0)
        monthly = monthly.reindex(columns=range(1, 13), fill_value=0.0)
        arr = monthly.to_numpy()

        vmax = np.nanmax(np.abs(arr)) if arr.size else 1.0
        vmax = max(vmax, 1e-6)
        im = ax.imshow(arr, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
        ax.set_title(f"Monthly PnL Heatmap ({title})")
        ax.set_xticks(range(12))
        ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], rotation=45)
        ax.set_yticks(range(len(monthly.index)))
        ax.set_yticklabels(monthly.index.astype(str))
        ax.set_xlabel("Month")

    axes[0].set_ylabel("Year")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("Monthly PnL (USD)")
    fig.subplots_adjust(wspace=0.25, right=0.92, bottom=0.2)

    file_path = out / "monthly_pnl_heatmap.png"
    fig.savefig(file_path, dpi=140)
    plt.close(fig)
    return file_path


def plot_winrate_by_vix_regime(results: pd.DataFrame, output_dir: str, scenario_col: str = "outcome_tp50_or_expiry") -> Path:
    out = _ensure_dir(output_dir)
    df = results.copy()
    df = df[(~df["skipped"]) & df[scenario_col].notna()].copy()
    if df.empty:
        return out / "winrate_by_vix_regime.png"

    bins = [10, 15, 20, 25, 30, np.inf]
    labels = ["10-15", "15-20", "20-25", "25-30", "30+"]
    df["vix_bucket"] = pd.cut(df["vix"], bins=bins, labels=labels, right=False)
    grouped = df.groupby("vix_bucket", observed=False)
    win_rate = grouped[scenario_col].apply(lambda x: (x > 0).mean() * 100.0).reindex(labels)
    count = grouped[scenario_col].count().reindex(labels).fillna(0)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(labels, win_rate.values, color="#1f77b4", alpha=0.75, label="Win Rate")
    ax1.set_ylabel("Win Rate (%)", color="#1f77b4")
    ax1.set_ylim(0, 100)
    ax1.set_xlabel("VIX Regime")
    ax1.grid(axis="y", alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(labels, count.values, color="#ff7f0e", marker="o", linewidth=2.0, label="Trades")
    ax2.set_ylabel("Trade Count", color="#ff7f0e")

    fig.suptitle("Win Rate by VIX Regime")
    fig.tight_layout()
    file_path = out / "winrate_by_vix_regime.png"
    fig.savefig(file_path, dpi=140)
    plt.close(fig)
    return file_path


def plot_pnl_distribution(results: pd.DataFrame, output_dir: str, scenario_col: str = "outcome_tp50_or_expiry") -> Path:
    out = _ensure_dir(output_dir)
    pnl = results.loc[(~results["skipped"]) & results[scenario_col].notna(), scenario_col].astype(float)
    if pnl.empty:
        return out / "pnl_distribution.png"

    fig, ax = plt.subplots(figsize=(10, 5))
    counts, bins, _ = ax.hist(pnl, bins=40, alpha=0.65, color="#2ca02c", edgecolor="black", label="PnL histogram")

    mu = pnl.mean()
    sigma = pnl.std(ddof=0)
    if sigma > 0:
        x = np.linspace(pnl.min(), pnl.max(), 300)
        pdf = norm.pdf(x, mu, sigma)
        scaled_pdf = pdf * len(pnl) * (bins[1] - bins[0])
        ax.plot(x, scaled_pdf, color="#d62728", linewidth=2.0, label="Normal fit")

    ax.axvline(mu, color="#1f77b4", linestyle="--", linewidth=1.8, label=f"Mean: {mu:.2f}")
    ax.axvline(pnl.median(), color="#9467bd", linestyle="--", linewidth=1.8, label=f"Median: {pnl.median():.2f}")
    ax.axvline(0.0, color="black", linewidth=1.2, label="Breakeven")

    sk = skew(pnl, bias=False) if len(pnl) > 2 else np.nan
    ku = kurtosis(pnl, bias=False) if len(pnl) > 3 else np.nan
    ax.set_title(f"PnL Distribution ({scenario_col}) | skew={sk:.2f}, kurt={ku:.2f}")
    ax.set_xlabel("Per-trade PnL (USD)")
    ax.set_ylabel("Frequency")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    file_path = out / "pnl_distribution.png"
    fig.savefig(file_path, dpi=140)
    plt.close(fig)
    return file_path


def plot_drawdown_chart(results: pd.DataFrame, output_dir: str, starting_capital: float, scenario_col: str = "outcome_tp50_or_expiry") -> Path:
    out = _ensure_dir(output_dir)
    df = results.copy()
    df["date"] = pd.to_datetime(df["date"])
    pnl = df[scenario_col].fillna(0.0).astype(float)
    equity = starting_capital + pnl.cumsum()
    peak = equity.cummax()
    drawdown = equity - peak

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["date"], drawdown, color="#d62728", linewidth=1.6)
    if len(drawdown):
        min_idx = int(drawdown.idxmin())
        ax.scatter(df.loc[min_idx, "date"], drawdown.loc[min_idx], color="black", s=30, label=f"Max DD: {drawdown.min():.2f}")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title(f"Drawdown ({scenario_col})")
    ax.set_ylabel("Drawdown (USD)")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()

    file_path = out / "drawdown_chart.png"
    fig.savefig(file_path, dpi=140)
    plt.close(fig)
    return file_path


def plot_parameter_sweep_heatmaps(sweep_df: pd.DataFrame, output_dir: str, metric_col: str = "tp50_or_expiry_total_pnl_usd") -> Optional[Path]:
    out = _ensure_dir(output_dir)
    if sweep_df.empty or metric_col not in sweep_df.columns:
        return None

    if "combo_skipped" in sweep_df.columns:
        mask = sweep_df["combo_skipped"] == False  # noqa: E712
    else:
        mask = pd.Series(False, index=sweep_df.index)
    valid = sweep_df[mask].copy()
    if valid.empty:
        return None

    pairs = [("wing_width", "target_delta"), ("min_credit", "bid_ask_haircut")]
    available_pairs = [p for p in pairs if p[0] in valid.columns and p[1] in valid.columns]
    if not available_pairs:
        return None

    fig, axes = plt.subplots(1, len(available_pairs), figsize=(7 * len(available_pairs), 5), squeeze=False)
    axes = axes[0]
    im = None
    for ax, (p1, p2) in zip(axes, available_pairs):
        pivot = valid.pivot_table(index=p2, columns=p1, values=metric_col, aggfunc="mean")
        arr = pivot.to_numpy()
        vmax = np.nanmax(np.abs(arr)) if arr.size else 1.0
        vmax = max(vmax, 1e-6)
        im = ax.imshow(arr, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{metric_col}\n{p2} vs {p1}")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{x:.2f}" if isinstance(x, float) else str(x) for x in pivot.columns], rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{x:.2f}" if isinstance(x, float) else str(x) for x in pivot.index])
        ax.set_xlabel(p1)
        ax.set_ylabel(p2)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
        cbar.set_label(metric_col)
    fig.tight_layout()

    file_path = out / "parameter_sweep_heatmaps.png"
    fig.savefig(file_path, dpi=140)
    plt.close(fig)
    return file_path


def plot_breakeven_analysis(config: Dict, output_dir: str) -> Path:
    out = _ensure_dir(output_dir)
    commission = float(config.get("costs", {}).get("commission_per_trade", 5.0))
    tp_pct = float(config.get("exit_rules", {}).get("take_profit_pct", 0.50))
    sl_mult = float(config.get("exit_rules", {}).get("stop_loss_mult", 3.0))
    wing_widths = list(config.get("sweep", {}).get("wing_width", [1.0, 2.0, 3.0, 5.0]))
    if not wing_widths:
        wing_widths = [1.0, 2.0, 3.0, 5.0]

    credits = np.linspace(0.05, 0.50, 200)
    fig, ax = plt.subplots(figsize=(11, 6))

    for wing in wing_widths:
        req = []
        for c in credits:
            req.append(breakeven_winrate(c, float(wing), commission, tp_pct, sl_mult)["with_tp_sl"] * 100.0)
        ax.plot(credits, req, linewidth=1.8, label=f"Wing {wing}")

    for wr in [70, 75, 80]:
        ax.axhline(wr, linestyle="--", linewidth=1.0, alpha=0.7, label=f"{wr}% WR")
    ax.axhspan(0, 75, alpha=0.1, color="#2ca02c", label="Viable zone (<75% req WR)")
    ax.set_title("Break-even Win Rate vs Net Credit")
    ax.set_xlabel("Net Credit per Share (USD)")
    ax.set_ylabel("Required Win Rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.2)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()

    file_path = out / "breakeven_analysis.png"
    fig.savefig(file_path, dpi=140)
    plt.close(fig)
    return file_path


def plot_commission_impact(sweep_df: pd.DataFrame, output_dir: str) -> Optional[Path]:
    out = _ensure_dir(output_dir)
    needed = {
        "tp50_or_expiry_total_pnl_usd",
        "tp50_or_expiry_pnl_before_commissions",
        "tp50_or_expiry_pnl_after_commissions",
    }
    if sweep_df.empty or not needed.issubset(set(sweep_df.columns)):
        return None

    if "combo_skipped" in sweep_df.columns:
        mask = sweep_df["combo_skipped"] == False  # noqa: E712
    else:
        mask = pd.Series(False, index=sweep_df.index)
    valid = sweep_df[mask].copy()
    if valid.empty:
        return None

    top = valid.sort_values("tp50_or_expiry_total_pnl_usd", ascending=False).head(5).copy()
    labels = [f"Cfg {i+1}" for i in range(len(top))]
    x = np.arange(len(top))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, top["tp50_or_expiry_pnl_before_commissions"], width, label="Before commissions")
    ax.bar(x + width / 2, top["tp50_or_expiry_pnl_after_commissions"], width, label="After commissions")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("PnL (USD)")
    ax.set_title("Commission Impact on Top 5 Configurations")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()

    file_path = out / "commission_impact.png"
    fig.savefig(file_path, dpi=140)
    plt.close(fig)
    return file_path


def generate_standard_charts(
    results: pd.DataFrame,
    metrics: Dict[str, Dict[str, float]],
    config: Dict[str, Any],
    output_dir: str,
    start: str,
    end: str,
) -> Dict[str, str]:
    """Generate all non-sweep charts and return file map."""
    output = {
        "equity_curves": str(plot_equity_curves(results, output_dir, start, end)),
        "monthly_heatmap": str(plot_monthly_pnl_heatmap(results, output_dir)),
        "winrate_vix": str(plot_winrate_by_vix_regime(results, output_dir)),
        "pnl_distribution": str(plot_pnl_distribution(results, output_dir)),
        "drawdown": str(
            plot_drawdown_chart(
                results,
                output_dir,
                starting_capital=float(config.get("capital", {}).get("starting_capital", 1580.0)),
            )
        ),
        "breakeven_analysis": str(plot_breakeven_analysis(config, output_dir)),
    }
    return output


def generate_sweep_charts(sweep_df: pd.DataFrame, config: Dict[str, Any], output_dir: str) -> Dict[str, Optional[str]]:
    """Generate sweep-specific charts and return file map."""
    heat = plot_parameter_sweep_heatmaps(sweep_df, output_dir)
    comm = plot_commission_impact(sweep_df, output_dir)
    return {
        "sweep_heatmaps": str(heat) if heat else None,
        "commission_impact": str(comm) if comm else None,
    }
