import pandas as pd
import numpy as np
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
        
        # Financial Reality Params
        sim_config = config.get('simulation', {})
        self.contract_multiplier = sim_config.get('contract_multiplier', 100)
        self.commission_per_leg = sim_config.get('commission_per_leg', 1.50)
        self.slippage_per_leg = sim_config.get('slippage_per_leg', 0.05)
        self.max_daily_loss = sim_config.get('max_daily_loss', 3000)
        self.daily_pnl = 0.0
        
    def run_day(self, date) -> List[IronCondorTrade]:
        self.daily_pnl = 0.0 # Reset PnL at start of day
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
                # KILL SWITCH: Check Daily Loss Limit
                if self.daily_pnl < -self.max_daily_loss:
                     print(f"💀 DAILY LOSS LIMIT HIT (${self.daily_pnl:.2f} < -${self.max_daily_loss}). HALTING.")
                     break # Stop trading for the day (skip remaining chains)

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
                    
                    # Update Daily PnL
                    self.daily_pnl += exit_signal.pnl
                    
                    print(f"Exited trade at {chain.timestamp}: {exit_signal.exit_reason} | PnL: ${exit_signal.pnl:.2f}")
                    active_trade = None
                    break
                    
                # Track last chain for EOD exit
                last_chain = chain
        
        # End of Day Cleanup
        if active_trade is not None and active_trade.status == "OPEN":
            # Force Close at EOD using last available chain
            if last_chain:
                # Calculate exit price (Mark to Market)
                exit_debit = self.exit_manager.calculate_debit(active_trade, last_chain)
                
                # FALLBACK: If standard calculation returns None (missing quotes), 
                # try theoretical pricing for OTM options
                if exit_debit is None:
                    print(f"⚠️ Warning: Missing quotes at EOD. Attempting Fallback Pricing...")
                    exit_debit = self._calculate_fallback_debit(active_trade, last_chain)
                
                if exit_debit is not None:
                    # Calculate Net USD PnL (Standardized)
                    gross_pnl_points = active_trade.entry_credit - exit_debit
                    gross_pnl_usd = gross_pnl_points * self.contract_multiplier
                    
                    # Costs
                    num_legs = len(active_trade.legs)
                    costs = (self.commission_per_leg * num_legs * 2) + (self.slippage_per_leg * num_legs)
                    
                    pnl = gross_pnl_usd - costs
                    pnl_pct = pnl / (active_trade.max_loss * self.contract_multiplier) if active_trade.max_loss > 0 else 0
                    
                    from .exits import TradeExit
                    active_trade.status = "CLOSED"
                    active_trade.exit_info = TradeExit(
                        timestamp=last_chain.timestamp,
                        exit_price=exit_debit,
                        exit_reason="EOD",
                        pnl=pnl,
                        pnl_pct=pnl_pct
                    )
                    self.daily_pnl += pnl
                    print(f"EOD Exit at {last_chain.timestamp}: PnL ${pnl:.2f}")
                else:
                    # Only if Fallback also fails, assume max loss
                    # Max Loss in USD
                    pnl = -active_trade.max_loss * self.contract_multiplier
                    # Add costs? If max loss, usually you pay full width.
                    # Costs are already implicit in the width diff?
                    # Max Loss = Width - Credit. This is the net cash flow if expires ITM.
                    # But we usually pay commissions to close or expire? 
                    # If expires, usually no comms on OTM, but full comms on ITM?
                    # Assume worst case: Max Loss + Comms.
                    costs = (self.commission_per_leg * len(active_trade.legs) * 2) 
                    pnl = pnl - costs
                    
                    from .exits import TradeExit
                    active_trade.status = "CLOSED"
                    active_trade.exit_info = TradeExit(
                        timestamp=last_chain.timestamp,
                        exit_price=active_trade.max_loss + active_trade.entry_credit,
                        exit_reason="EOD_NO_QUOTE",
                        pnl=pnl,
                        pnl_pct=-1.0
                    )
                    self.daily_pnl += pnl
                    print(f"EOD Exit (No Quote) - Assumed Max Loss: ${pnl:.2f}")
                    
            except Exception as e:
                # If we can't calculate exit, mark as expired with max loss
                active_trade.status = "EXPIRED"
                print(f"EOD Exit failed: {e}")
                    
        return daily_trades
    
    def _calculate_fallback_debit(self, trade: IronCondorTrade, chain: OptionChain) -> float:
        """
        Calculate theoretical closing cost when quotes are missing.
        Uses Black-Scholes for theoretical pricing when market quotes unavailable.
        """
        from ..analytics.greeks import BlackScholesSolver
        from ..data.schema import OptionType
        
        total_debit = 0.0
        spot = chain.underlying_price
        bs_solver = BlackScholesSolver()
        
        # Estimate time to expiration (assume EOD = ~0 hours left)
        dte_years = 1 / (365 * 24)  # ~1 hour left in 0DTE
        
        for leg in trade.legs:
            q = chain.get_quote(leg.strike, leg.option_type)
            
            if q:
                # Quote exists - use real market price
                price = q.bid if leg.is_long else q.ask
            else:
                # Quote missing - use Black-Scholes theoretical price
                dist_pct = abs(spot - leg.strike) / spot if spot > 0 else 0
                
                if dist_pct > 0.01:  # More than 1% away from spot
                    # Use Black-Scholes with estimated IV
                    otype_str = 'call' if leg.option_type == OptionType.CALL else 'put'
                    
                    # ESTUDIO TITO: NO FALLBACKS. Get IV from quote or ABORT.
                    if q and q.implied_vol and q.implied_vol > 0 and not np.isnan(q.implied_vol):
                        estimated_iv = q.implied_vol
                    else:
                        print(f"  Fallback ABORTED: No valid IV for {leg.option_type.name} {leg.strike}")
                        return None  # Cannot price without IV
                    
                    # Calculate theoretical price
                    theo_price = bs_solver.calculate_price(
                        option_type=otype_str,
                        S=spot,
                        K=leg.strike,
                        T=dte_years,
                        sigma=estimated_iv
                    )
                    
                    # Floor at tick minimum
                    price = max(0.05, theo_price)
                    print(f"  Fallback BS: {leg.option_type.name} {leg.strike} @ ${price:.2f} (IV={estimated_iv:.0%})")
                else:
                    # Near ATM with missing quote - dangerous, can't price safely
                    print(f"  Fallback ABORTED: {leg.option_type.name} {leg.strike} near ATM")
                    return None
            
            # Accumulate debit
            if leg.is_long:
                total_debit -= price  # Credit (we receive)
            else:
                total_debit += price  # Debit (we pay)
                
        return total_debit

    def run_simulation(self) -> pd.DataFrame:
        """
        Runs simulation over range of dates in config.
        """
        # TODO: Implement date loop
        pass
