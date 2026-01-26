import pandas as pd
from typing import Generator, Optional
from datetime import datetime, date
from .schema import OptionChain, Quote, OptionType

class DataLoader:
    """
    Abstract base or concrete implementation to load market data.
    Currently a placeholder for Parquet/CSV loading logic.
    """
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_day(self, trade_date: date) -> Generator[OptionChain, None, None]:
        """
        Yields OptionChain snapshots for the given day, minute by minute.
        """
        # TODO: Implement actual parquet loading logic here
        # This would read underlying spot data + option chains
        # and merge them into OptionChain objects.
        pass

def parse_chain_from_pandas(df_slice: pd.DataFrame, timestamp: datetime, spot_price: float, expiration: date) -> OptionChain:
    """
    Helper to convert a pandas DataFrame slice (one timestamp) into an OptionChain object.
    Expected DF columns: strike, type, bid, ask, etc.
    """
    chain = OptionChain(timestamp=timestamp, underlying_price=spot_price, expiration=expiration)
    
    for _, row in df_slice.iterrows():
        otype = OptionType.CALL if row['option_type'] == 'C' else OptionType.PUT
        strike = float(row['strike'])
        
        q = Quote(
            bid=float(row['bid']),
            ask=float(row['ask']),
            mid=row.get('mid', (row['bid'] + row['ask'])/2),
            implied_vol=row.get('implied_vol'),
            delta=row.get('delta'),
            gamma=row.get('gamma'),
            vega=row.get('vega'),
            theta=row.get('theta'),
            volume=row.get('volume'),
            open_interest=row.get('open_interest')
        )
        chain.quotes[(strike, otype)] = q
        
    return chain
