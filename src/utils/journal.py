"""
journal.py

Trade Journal (Bitácora de Trading) for learning and auditing.
Records every trade with full context for post-trade analysis.
"""
import os
import csv
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path


class TradeJournal:
    """
    Trade Journal that logs all trades to CSV for learning and auditing.
    
    Each entry includes:
    - Timestamp
    - Market Context (Spot, VIX)
    - Trade Setup (Strikes, Credit)
    - Greeks Snapshot (Delta, Theta, Gamma)
    - Status and PnL
    """
    
    COLUMNS = [
        'trade_id',
        'timestamp',
        'status',
        # Market Context
        'spot_price',
        'vix_value',
        # Trade Setup
        'short_put_strike',
        'short_call_strike',
        'wing_width',
        'entry_credit',
        'max_profit_usd',
        'max_loss_usd',
        'max_spread_val',   # NEW: Max spread value observed (Tail Risk)
        'rv_duration',      # NEW: Realized Volatility over duration (Placeholder)
        # Execution Quality (NEW)
        'initial_credit',   # NEW: Net credit at open
        'target_credit',    # Wanted Limit Price
        'slippage_usd',     # Entry - Target
        'commissions_est',  # Estimated Fees
        # Market State (NEW)
        'iv_entry_atm',     # IV at entry
        # Greeks
        'delta_net',
        'delta_put',      # Delta of short put at entry
        'delta_call',     # Delta of short call at entry
        'theta',
        'gamma',
        # Selection Context
        'selection_method',   # e.g., OTM_DISTANCE_PCT, DELTA_TARGET
        'target_delta',       # e.g., 0.10
        'otm_distance_pct',   # e.g., 1.5%
        # Outcome
        'exit_time',
        'exit_reason',
        'final_pnl_usd',
        'max_adverse_excursion', # NEW: Max pain ($ drawdown)
        'hold_duration_mins',
        # Audit
        'snapshot_json',      # Entry Snapshot
        'exit_snapshot_json', # NEW: Exit Snapshot
        # Notes
        'reasoning',
    ]
    
    def __init__(self, journal_path: str = "data/trade_journal.csv"):
        self.journal_path = Path(journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_file_exists()
        self._trade_counter = self._get_last_trade_id()
        
    def _ensure_file_exists(self):
        """Create CSV file with headers if it doesn't exist, or migrate schema if changed."""
        if not self.journal_path.exists():
            with open(self.journal_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
                writer.writeheader()
            return

        # Check existing headers
        try:
            with open(self.journal_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader, [])
                
            if headers != self.COLUMNS:
                print(f"⚠️ Journal schema mismatch. Backing up old journal...")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.journal_path.with_name(f"{self.journal_path.stem}_backup_{timestamp}.csv")
                
                os.rename(self.journal_path, backup_path)
                print(f"📦 Archived to {backup_path}")
                
                # Re-create fresh file
                with open(self.journal_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
                    writer.writeheader()
                    
        except Exception as e:
            print(f"⚠️ Error checking journal schema: {e}")
    
    def _get_last_trade_id(self) -> int:
        """Get the last trade ID from the journal."""
        try:
            with open(self.journal_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    return int(rows[-1].get('trade_id', 0))
        except (FileNotFoundError, ValueError):
            pass
        return 0
    
    def log_trade_open(
        self,
        spot_price: float,
        vix_value: float,
        short_put_strike: float,
        short_call_strike: float,
        wing_width: float,
        entry_credit: float,
        max_profit_usd: float,
        max_loss_usd: float,
        delta_net: float,
        delta_put: float = 0.0,
        delta_call: float = 0.0,
        theta: float = 0.0,
        gamma: float = 0.0,
        selection_method: str = "",
        target_delta: float = 0.0,
        otm_distance_pct: str = "",

        snapshot_json: str = "",
        target_credit: float = 0.0,       
        commissions_est: float = 2.60,
        initial_credit: float = 0.0,      # NEW
        iv_entry_atm: float = 0.0,        # NEW
        reasoning: str = ""
    ) -> int:
        """
        Log a new trade opening.
        
        Returns:
            trade_id: Unique ID for this trade
        """
        self._trade_counter += 1
        trade_id = self._trade_counter
        
        # Calculate Slippage
        slippage = 0.0
        if target_credit > 0:
            slippage = entry_credit - target_credit
        
        entry = {
            'trade_id': trade_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'OPEN',
            'spot_price': round(spot_price, 2),
            'vix_value': round(vix_value, 2),
            'short_put_strike': short_put_strike,
            'short_call_strike': short_call_strike,
            'wing_width': wing_width,
            'entry_credit': round(entry_credit, 4),
            'max_profit_usd': round(max_profit_usd, 2),
            'max_loss_usd': round(max_loss_usd, 2),
            # HFT Metrics
            'initial_credit': round(initial_credit, 4),
            'target_credit': round(target_credit, 4),
            'slippage_usd': round(slippage, 4),
            'commissions_est': round(commissions_est, 2),
            # Market State
            'iv_entry_atm': round(iv_entry_atm, 4),
            # Greeks
            'delta_net': round(delta_net, 4),
            'delta_put': round(delta_put, 4),       
            'delta_call': round(delta_call, 4),     
            'theta': round(theta, 4),
            'gamma': round(gamma, 6),
            'selection_method': selection_method,   
            'target_delta': round(target_delta, 4), 
            'otm_distance_pct': otm_distance_pct,   
            'exit_time': '',
            'exit_reason': '',
            'final_pnl_usd': '',
            'max_adverse_excursion': '', # NEW
            'max_spread_val': '', # NEW
            'rv_duration': '',    # NEW
            'hold_duration_mins': '',
            'snapshot_json': snapshot_json,         
            'exit_snapshot_json': '',    # NEW
            'reasoning': reasoning,
        }
        
        with open(self.journal_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writerow(entry)
        
        print(f"📝 Trade #{trade_id} logged to journal")
        return trade_id
    
    def log_trade_close(
        self,
        trade_id: int,
        exit_reason: str,
        final_pnl_usd: float,
        entry_timestamp: datetime,
        max_adverse_excursion: float = 0.0,
        max_favorable_excursion: float = 0.0,
        max_spread_val: float = 0.0,        # NEW
        rv_duration: float = 0.0,           # NEW
        exit_snapshot_json: str = ""        
    ):
        """
        Update a trade entry with close information.
        
        Args:
            trade_id: ID of the trade to update
            exit_reason: Why the trade was closed
            final_pnl_usd: Final PnL in dollars
            entry_timestamp: When the trade was opened
            max_adverse_excursion: Lowest PnL point
            max_favorable_excursion: Highest PnL point
            max_spread_val: Max price of spread (Tail Risk)
            exit_snapshot_json: Final market state
        """
        # Read all rows
        rows = []
        with open(self.journal_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Find and update the trade
        found = False
        for row in rows:
            if int(row.get('trade_id', 0)) == trade_id:
                row['status'] = 'CLOSED'
                row['exit_time'] = datetime.now().isoformat()
                row['exit_reason'] = exit_reason
                row['final_pnl_usd'] = round(final_pnl_usd, 2)
                row['max_adverse_excursion'] = round(max_adverse_excursion, 2)
                row['max_spread_val'] = max_spread_val
                row['rv_duration'] = round(rv_duration, 6) # NEW
                
                # Calculate hold duration
                duration_mins = ''
                try:
                    entry_dt = datetime.fromisoformat(row['timestamp'])
                    exit_dt = datetime.now()
                    duration_mins = (exit_dt - entry_dt).total_seconds() / 60
                    row['hold_duration_mins'] = round(duration_mins, 1)
                except:
                    row['hold_duration_mins'] = ''

                # Store MFE note without changing journal schema
                mfe_note = f"MFE=${max_favorable_excursion:.2f}"
                existing_reasoning = (row.get('reasoning') or '').strip()
                if existing_reasoning:
                    if "MFE=" not in existing_reasoning:
                        row['reasoning'] = f"{existing_reasoning} | {mfe_note}"
                else:
                    row['reasoning'] = mfe_note
                
                found = True
                break
        
        if not found:
            print(f"⚠️ Trade #{trade_id} not found in journal")
            return

        # Atomic write: tmp file + rename to prevent data loss on crash
        tmp_path = self.journal_path.with_suffix('.tmp')
        with open(tmp_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        tmp_path.replace(self.journal_path)  # atomic on same filesystem
        
        print(f"📝 Trade #{trade_id} closed in journal (PnL: ${final_pnl_usd:.2f})")
    
    def get_trade_summary(self) -> Dict[str, Any]:
        """Get summary statistics from the journal."""
        try:
            with open(self.journal_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except FileNotFoundError:
            return {'total_trades': 0}
        
        closed_trades = [r for r in rows if r.get('status') == 'CLOSED']
        
        if not closed_trades:
            return {
                'total_trades': len(rows),
                'closed_trades': 0,
                'open_trades': len(rows),
            }
        
        pnls = []
        maes = []
        slippages = []
        
        for r in closed_trades:
            try:
                pnls.append(float(r.get('final_pnl_usd', 0)))
                maes.append(float(r.get('max_adverse_excursion', 0)))
                slippages.append(float(r.get('slippage_usd', 0)))
            except ValueError:
                pass
        
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]
        
        return {
            'total_trades': len(rows),
            'closed_trades': len(closed_trades),
            'open_trades': len(rows) - len(closed_trades),
            'total_pnl': sum(pnls),
            'win_rate': len(winners) / len(pnls) * 100 if pnls else 0,
            'avg_win': sum(winners) / len(winners) if winners else 0,
            'avg_loss': sum(losers) / len(losers) if losers else 0,
            'best_trade': max(pnls) if pnls else 0,
            'worst_trade': min(pnls) if pnls else 0,
            'avg_mae': sum(maes) / len(maes) if maes else 0,       
            'total_slippage': sum(slippages) if slippages else 0,  
            'max_spread_val': 0.0, # Placeholder
        }
    
    def print_summary(self):
        """Print a formatted summary of trading performance."""
        stats = self.get_trade_summary()
        
        print("\n" + "═" * 50)
        print("  📊 TRADE JOURNAL SUMMARY")
        print("═" * 50)
        print(f"  Total Trades: {stats.get('total_trades', 0)}")
        print(f"  Closed: {stats.get('closed_trades', 0)} | Open: {stats.get('open_trades', 0)}")
        
        if stats.get('closed_trades', 0) > 0:
            print(f"\n  💰 Total PnL: ${stats.get('total_pnl', 0):.2f}")
            print(f"  📈 Win Rate: {stats.get('win_rate', 0):.1f}%")
            print(f"  🏆 Best Trade: ${stats.get('best_trade', 0):.2f}")
            print(f"  💀 Worst Trade: ${stats.get('worst_trade', 0):.2f}")
            print(f"  📉 Avg Pain (MAE): ${stats.get('avg_mae', 0):.2f}")
            print(f"  💸 Total Slippage: ${stats.get('total_slippage', 0):.2f}")
        
        print("═" * 50 + "\n")
