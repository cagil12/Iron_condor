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
        # Greeks
        'delta_net',
        'theta',
        'gamma',
        # Outcome
        'exit_time',
        'exit_reason',
        'final_pnl_usd',
        'hold_duration_mins',
        # Notes
        'reasoning',
    ]
    
    def __init__(self, journal_path: str = "data/trade_journal.csv"):
        self.journal_path = Path(journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_file_exists()
        self._trade_counter = self._get_last_trade_id()
        
    def _ensure_file_exists(self):
        """Create CSV file with headers if it doesn't exist."""
        if not self.journal_path.exists():
            with open(self.journal_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
                writer.writeheader()
    
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
        theta: float = 0.0,
        gamma: float = 0.0,
        reasoning: str = ""
    ) -> int:
        """
        Log a new trade opening.
        
        Returns:
            trade_id: Unique ID for this trade
        """
        self._trade_counter += 1
        trade_id = self._trade_counter
        
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
            'delta_net': round(delta_net, 4),
            'theta': round(theta, 4),
            'gamma': round(gamma, 6),
            'exit_time': '',
            'exit_reason': '',
            'final_pnl_usd': '',
            'hold_duration_mins': '',
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
        entry_timestamp: datetime
    ):
        """
        Update a trade entry with close information.
        
        Args:
            trade_id: ID of the trade to update
            exit_reason: Why the trade was closed
            final_pnl_usd: Final PnL in dollars
            entry_timestamp: When the trade was opened
        """
        # Read all rows
        rows = []
        with open(self.journal_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Find and update the trade
        for row in rows:
            if int(row.get('trade_id', 0)) == trade_id:
                row['status'] = 'CLOSED'
                row['exit_time'] = datetime.now().isoformat()
                row['exit_reason'] = exit_reason
                row['final_pnl_usd'] = round(final_pnl_usd, 2)
                
                # Calculate hold duration
                try:
                    entry_dt = datetime.fromisoformat(row['timestamp'])
                    exit_dt = datetime.now()
                    duration_mins = (exit_dt - entry_dt).total_seconds() / 60
                    row['hold_duration_mins'] = round(duration_mins, 1)
                except:
                    row['hold_duration_mins'] = ''
                break
        
        # Write all rows back
        with open(self.journal_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        
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
        for r in closed_trades:
            try:
                pnls.append(float(r.get('final_pnl_usd', 0)))
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
        
        print("═" * 50 + "\n")
