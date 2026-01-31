import pandas as pd
import re
from typing import Generator, Optional
from datetime import datetime, date, time
from .schema import OptionChain, Quote, OptionType
from ..analytics.greeks import BlackScholesSolver

# Global solver instance for reuse
_bs_solver = BlackScholesSolver()

class DataLoader:
    """
    Loads market data from CSV/Parquet and enriches with Greeks (IV, Delta).
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.symbol_pattern = re.compile(r"([A-Z]+)\s*(\d{2})(\d{2})(\d{2})([CP])(\d{8})")

    def generate_day(self, trade_date: date) -> Generator[OptionChain, None, None]:
        """
        Yields OptionChain snapshots for the given day, minute by minute.
        """
        date_str = trade_date.strftime("%Y%m%d")
        iso_date_str = trade_date.strftime("%Y-%m-%d")
        
        file_path_csv = f"{self.data_path}/opra-pillar-{date_str}.tcbbo.csv.zst"
        file_path_parquet = f"{self.data_path}/{iso_date_str}.parquet"
        
        df = None
        try:
            df = pd.read_csv(file_path_csv, compression='zstd', low_memory=False)
            
            if 'ts_event' in df.columns:
                df['ts_event'] = pd.to_datetime(df['ts_event'])
                df = df.set_index('ts_event')
            elif 'ts_recv' in df.columns:
                df['ts_recv'] = pd.to_datetime(df['ts_recv'])
                df = df.set_index('ts_recv')

        except FileNotFoundError:
            try:
                df = pd.read_parquet(file_path_parquet)
            except FileNotFoundError:
                print(f"Warning: No data found for {date_str} (checked csv.zst and parquet)")
                return

        if df is None or df.empty:
            return

        df = df.sort_index()
        grouper = df.groupby(pd.Grouper(freq='1min'))
        
        for ts, minute_df in grouper:
            if minute_df.empty:
                continue
            
            if ts.time() < time(9, 30) or ts.time() > time(16, 15):
                continue
            
            yield self._parse_chain(minute_df, ts, trade_date)

    def _parse_chain(self, df_slice: pd.DataFrame, timestamp: datetime, expiration: date) -> OptionChain:
        """
        Parse DataFrame into OptionChain with Greeks enrichment.
        """
        parsed_quotes = []
        
        # Calculate Time To Expiry (fraction of year)
        # For 0DTE: Assume market close 4:00 PM = 16:00
        # Time remaining = (16:00 - current_time) / (365 * 24 * 60) minutes
        # Simplified: Use fraction of trading day remaining
        current_time = timestamp.time()
        minutes_remaining = max(1, (16 * 60) - (current_time.hour * 60 + current_time.minute))
        dte_years = minutes_remaining / (365.25 * 24 * 60)  # ~minutes to year fraction
        
        # First pass: parse all rows
        for _, row in df_slice.iterrows():
            try:
                raw_symbol = str(row.get('symbol', ''))
                match = self.symbol_pattern.search(raw_symbol)
                if not match:
                    continue
                
                # Parse expiration from symbol (YYMMDD)
                exp_year = 2000 + int(match.group(2))
                exp_month = int(match.group(3))
                exp_day = int(match.group(4))
                option_expiry = date(exp_year, exp_month, exp_day)
                
                # Include all options - we'll use the closest expiration for short-term strategies
                # (The 0DTE filter was too strict for multi-expiry data files)
                
                otype_char = match.group(5)
                strike_str = match.group(6)
                
                otype = OptionType.CALL if otype_char == 'C' else OptionType.PUT
                strike = float(strike_str) / 1000.0
                
                bid = float(row.get('bid_px_00', 0))
                ask = float(row.get('ask_px_00', 0))
                
                if bid <= 0 or ask <= 0:
                    continue
                
                mid = (bid + ask) / 2
                
                parsed_quotes.append({
                    'strike': strike,
                    'type': otype,
                    'otype_str': 'call' if otype == OptionType.CALL else 'put',
                    'bid': bid,
                    'ask': ask,
                    'mid': mid,
                    'dte_years': dte_years
                })
                
            except Exception:
                continue

        if not parsed_quotes:
            return OptionChain(timestamp=timestamp, underlying_price=4000.0, expiration=expiration)

        # Infer Spot Price via Put-Call Parity
        strikes_dict = {}
        for p in parsed_quotes:
            k = p['strike']
            if k not in strikes_dict:
                strikes_dict[k] = {}
            strikes_dict[k][p['type']] = p['mid']
        
        found_spot = 4000.0
        min_diff = float('inf')
        
        for k, prices in strikes_dict.items():
            if OptionType.CALL in prices and OptionType.PUT in prices:
                diff = abs(prices[OptionType.CALL] - prices[OptionType.PUT])
                if diff < min_diff:
                    min_diff = diff
                    found_spot = k + (prices[OptionType.CALL] - prices[OptionType.PUT])

        # Second pass: Calculate IV and Delta for each option
        chain = OptionChain(timestamp=timestamp, underlying_price=found_spot, expiration=expiration)
        
        for p in parsed_quotes:
            strike = p['strike']
            otype = p['type']
            otype_str = p['otype_str']
            mid = p['mid']
            dte = p['dte_years']
            
            # Calculate IV from market price
            iv = _bs_solver.implied_volatility(
                option_type=otype_str,
                option_price=mid,
                S=found_spot,
                K=strike,
                T=dte
            )
            
            # Calculate Delta using IV
            delta = _bs_solver.calculate_delta(
                option_type=otype_str,
                S=found_spot,
                K=strike,
                T=dte,
                sigma=iv if iv > 0 else 0.20  # Fallback IV
            )
            
            q = Quote(
                bid=p['bid'],
                ask=p['ask'],
                mid=mid,
                implied_vol=iv,
                delta=delta
            )
            chain.quotes[(strike, otype)] = q
        
        return chain

