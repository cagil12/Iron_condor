import pandas as pd
from typing import List, Generator
from datetime import datetime, time
from .condor_builder import CondorBuilder, IronCondorTrade
from .exits import ExitManager
from ..data.schema import OptionChain
from ..data.loaders import DataLoader

class Simulator:
    def __init__(self, config: dict, data_loader: DataLoader):
        self.config = config
        self.loader = data_loader
        self.builder = CondorBuilder(config['strategy'])
        self.exit_manager = ExitManager(config['exit'])
        self.trades: List[IronCondorTrade] = []
        
    def run_day(self, date) -> List[IronCondorTrade]:
        daily_trades = []
        active_trade: IronCondorTrade = None
        
        entry_time_str = self.config['strategy'].get('entry_time', "10:00")
        entry_hh, entry_mm = map(int, entry_time_str.split(':'))
        entry_time_obj = time(entry_hh, entry_mm)
        
        # Generator yields chains minute by minute
        chain_stream = self.loader.generate_day(date)
        
        for chain in chain_stream:
            # 1. Entry Logic
            if active_trade is None:
                # Check time
                if chain.timestamp.time() == entry_time_obj:
                    candidate = self.builder.build_trade(chain)
                    if candidate:
                        active_trade = candidate
                        daily_trades.append(active_trade)
                        # print(f"Entered trade at {chain.timestamp}: Credit {active_trade.entry_credit:.2f}")
            
            # 2. Exit Logic (if in trade)
            else:
                exit_signal = self.exit_manager.check_exit(active_trade, chain)
                if exit_signal:
                    active_trade.status = "CLOSED"
                    active_trade.exit_info = exit_signal
                    # print(f"Exited trade at {chain.timestamp}: {exit_signal.exit_reason}, PnL {exit_signal.pnl:.2f}")
                    active_trade = None # Ready for next? (Usually one per day for 0DTE specific strategy, but logic supports multiple)
                    
                    # If we only want 1 trade per day, break here
                    break 
                    
        return daily_trades

    def run_simulation(self) -> pd.DataFrame:
        """
        Runs simulation over range of dates in config.
        """
        # TODO: Implement date loop
        pass
