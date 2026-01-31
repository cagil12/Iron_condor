import pandas as pd
from typing import List, Generator
from datetime import datetime, time
from .condor_builder import CondorBuilder, IronCondorTrade
from .exits import ExitManager
from ..data.schema import OptionChain
from ..data.loaders import DataLoader
from ..research.regimes import MarketRegime

class Simulator:
    def __init__(self, config: dict, data_loader: DataLoader):
        self.config = config
        self.loader = data_loader
        self.builder = CondorBuilder(config['strategy'])
        self.exit_manager = ExitManager(config['exit'])
        self.regime_filter = MarketRegime(config)
        self.trades: List[IronCondorTrade] = []
        
    def run_day(self, date) -> List[IronCondorTrade]:
        daily_trades = []
        active_trade: IronCondorTrade = None
        entry_attempted = False  # Flag to only attempt entry once per day
        
        entry_time_str = self.config['strategy'].get('entry_time', "10:00")
        entry_hh, entry_mm = map(int, entry_time_str.split(':'))
        entry_time_obj = time(entry_hh, entry_mm)
        
        # Generator yields chains minute by minute
        chain_stream = self.loader.generate_day(date)
        
        for chain in chain_stream:
            # 1. Entry Logic
            if active_trade is None and not entry_attempted:
                # Check time: >= entry_time (first chain at or after target time)
                if chain.timestamp.time() >= entry_time_obj:
                    entry_attempted = True  # Only try once per day
                    
                    # TAREA 2: Regime Filter Check
                    if not self.regime_filter.is_safe(chain.timestamp, vix_value=None):
                        print(f"Trade Skipped: High Risk Regime at {chain.timestamp}")
                        continue

                    print(f"Attempting trade at {chain.timestamp}, Spot={chain.underlying_price:.2f}, Quotes={len(chain.quotes)}")
                    candidate = self.builder.build_trade(chain)
                    if candidate:
                        active_trade = candidate
                        daily_trades.append(active_trade)
                        print(f"ENTERED trade at {chain.timestamp}: Credit {active_trade.entry_credit:.2f}")
            
            # 2. Exit Logic (if in trade)
            if active_trade is not None:
                exit_signal = self.exit_manager.check_exit(active_trade, chain)
                if exit_signal:
                    active_trade.status = "CLOSED"
                    active_trade.exit_info = exit_signal
                    print(f"Exited trade at {chain.timestamp}: {exit_signal.exit_reason}")
                    active_trade = None
                    break
                    
                # Track last chain for EOD exit
                last_chain = chain
        
        # 3. EOD Auto-Exit: Close any remaining open trade at end of day
        if active_trade is not None and active_trade.status == "OPEN":
            # Calculate exit using last available chain
            try:
                exit_debit = self.exit_manager.calculate_debit(active_trade, last_chain)
                
                if exit_debit is not None:
                    pnl = (active_trade.entry_credit - exit_debit) * 100  # Per contract
                    pnl_pct = pnl / (active_trade.max_loss * 100) if active_trade.max_loss > 0 else 0
                    
                    from .exits import TradeExit
                    active_trade.status = "CLOSED"
                    active_trade.exit_info = TradeExit(
                        timestamp=last_chain.timestamp,
                        exit_price=exit_debit,
                        exit_reason="EOD",
                        pnl=pnl,
                        pnl_pct=pnl_pct
                    )
                    print(f"EOD Exit at {last_chain.timestamp}: PnL ${pnl:.2f}")
                else:
                    # Can't calculate - use stored max_loss
                    pnl = -active_trade.max_loss * 100  # Assume max loss
                    
                    from .exits import TradeExit
                    active_trade.status = "CLOSED"
                    active_trade.exit_info = TradeExit(
                        timestamp=last_chain.timestamp,
                        exit_price=active_trade.max_loss + active_trade.entry_credit,
                        exit_reason="EOD_NO_QUOTE",
                        pnl=pnl,
                        pnl_pct=-1.0
                    )
                    print(f"EOD Exit (No Quote) - Assumed Max Loss: ${pnl:.2f}")
                    
            except Exception as e:
                # If we can't calculate exit, mark as expired with max loss
                active_trade.status = "EXPIRED"
                print(f"EOD Exit failed: {e}")
                    
        return daily_trades

    def run_simulation(self) -> pd.DataFrame:
        """
        Runs simulation over range of dates in config.
        """
        # TODO: Implement date loop
        pass
