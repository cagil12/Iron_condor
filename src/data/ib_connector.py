"""
ib_connector.py

Interactive Brokers connector using ib_insync.
Handles connection to TWS, market data subscriptions, and live price feeds.
Optimized for XSP (Mini-SPX) options trading.
"""
import time
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

try:
    from ib_insync import IB, Index, Option, Contract, util
except ImportError:
    raise ImportError("ib_insync not installed. Run: pip install ib_insync")


@dataclass
class SubscriptionStatus:
    """Status of a data subscription."""
    name: str
    is_valid: bool
    last_price: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    error_msg: Optional[str] = None


class IBConnector:
    """
    Interactive Brokers connector for live trading.
    
    Usage:
        connector = IBConnector()
        connector.connect(paper=True)
        price = connector.get_live_price('XSP')
        connector.disconnect()
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.ib = IB()
        self.config = config or self._default_config()
        self.connected = False
        self.account_id = None
        self.account_value = 0.0
        
    def _default_config(self) -> Dict[str, Any]:
        """Default IBKR connection config."""
        return {
            'host': '127.0.0.1',
            'paper_port': 7497,
            'live_port': 7496,
            'client_id': 1,
            'timeout': 30,
        }
    
    def connect(self, paper: bool = True) -> Tuple[bool, str]:
        """
        Connect to TWS.
        
        Args:
            paper: True for paper trading (7497), False for live (7496)
            
        Returns:
            Tuple of (success, message)
        """
        port = self.config['paper_port'] if paper else self.config['live_port']
        mode = "PAPER" if paper else "LIVE"
        
        try:
            self.ib.connect(
                host=self.config['host'],
                port=port,
                clientId=self.config['client_id'],
                timeout=self.config['timeout']
            )
            
            self.connected = True
            
            # Get account info
            accounts = self.ib.managedAccounts()
            if accounts:
                self.account_id = accounts[0]
                
                # Get account value
                account_values = self.ib.accountValues(self.account_id)
                for av in account_values:
                    if av.tag == 'NetLiquidation' and av.currency == 'USD':
                        self.account_value = float(av.value)
                        break
            
            return True, f"Connected to TWS ({mode}) - Account: {self.account_id}"
            
        except Exception as e:
            self.connected = False
            return False, f"Connection failed: {str(e)}"
    
    def disconnect(self):
        """Disconnect from TWS."""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
    
    def get_account_value(self) -> float:
        """Get current account net liquidation value in USD."""
        if not self.connected:
            return 0.0
        
        try:
            account_values = self.ib.accountValues(self.account_id)
            for av in account_values:
                if av.tag == 'NetLiquidation' and av.currency == 'USD':
                    return float(av.value)
        except Exception:
            pass
        
        return self.account_value
    
    def get_live_price(self, symbol: str = 'XSP') -> Optional[float]:
        """
        Get live price for an index.
        
        Args:
            symbol: Index symbol (default: XSP)
            
        Returns:
            Last price or None if unavailable
        """
        if not self.connected:
            return None
        
        try:
            contract = Index(symbol, 'CBOE')
            self.ib.qualifyContracts(contract)
            
            # Request market data (snapshot=False for continuous stream)
            ticker = self.ib.reqMktData(contract, '', snapshot=False)
            
            # Wait briefly for data
            for _ in range(10):
                self.ib.sleep(0.2)
                if ticker.last and ticker.last > 0:
                    self.ib.cancelMktData(contract)
                    return ticker.last
                if ticker.close and ticker.close > 0:
                    self.ib.cancelMktData(contract)
                    return ticker.close
            
            self.ib.cancelMktData(contract)
            return None
            
        except Exception as e:
            print(f"Error getting price for {symbol}: {e}")
            return None
    
    def get_option_quote(self, symbol: str, expiry: str, strike: float, 
                         right: str = 'C') -> Tuple[Optional[float], Optional[float]]:
        """
        Get bid/ask for an option.
        
        Args:
            symbol: Underlying symbol (e.g., 'XSP')
            expiry: Expiration date (YYYYMMDD format)
            strike: Strike price
            right: 'C' for Call, 'P' for Put
            
        Returns:
            Tuple of (bid, ask) or (None, None) if unavailable
        """
        if not self.connected:
            return None, None
        
        try:
            contract = Option(symbol, expiry, strike, right, 'SMART')
            self.ib.qualifyContracts(contract)
            
            ticker = self.ib.reqMktData(contract, '', snapshot=False)
            
            # Wait for data
            for _ in range(10):
                self.ib.sleep(0.2)
                if ticker.bid and ticker.ask and ticker.bid > 0:
                    self.ib.cancelMktData(contract)
                    return ticker.bid, ticker.ask
            
            self.ib.cancelMktData(contract)
            return None, None
            
        except Exception as e:
            print(f"Error getting option quote: {e}")
            return None, None
    
    def verify_subscriptions(self) -> Dict[str, SubscriptionStatus]:
        """
        Verify that all required data subscriptions are active.
        
        Returns:
            Dict with subscription status for each feed
        """
        results = {}
        
        # 1. Verify CBOE Index Feed (XSP Spot)
        xsp_price = self.get_live_price('XSP')
        if xsp_price and xsp_price > 0:
            results['cboe_index'] = SubscriptionStatus(
                name='CBOE INDEX FEED (XSP Spot)',
                is_valid=True,
                last_price=xsp_price
            )
        else:
            results['cboe_index'] = SubscriptionStatus(
                name='CBOE INDEX FEED (XSP Spot)',
                is_valid=False,
                error_msg='No data received - Check CBOE Indices subscription'
            )
        
        # 2. Verify OPRA Options Feed
        # Use a near-term ATM option as test
        if xsp_price and xsp_price > 0:
            # Round to nearest strike
            test_strike = round(xsp_price)
            # Use next Friday expiry (simplified)
            from datetime import date, timedelta
            today = date.today()
            days_ahead = 4 - today.weekday()  # Friday = 4
            if days_ahead <= 0:
                days_ahead += 7
            next_friday = today + timedelta(days=days_ahead)
            expiry = next_friday.strftime('%Y%m%d')
            
            bid, ask = self.get_option_quote('XSP', expiry, test_strike, 'C')
            
            if bid is not None and ask is not None and bid > 0:
                results['opra_options'] = SubscriptionStatus(
                    name='OPRA OPTIONS FEED (Prueba)',
                    is_valid=True,
                    bid=bid,
                    ask=ask
                )
            else:
                results['opra_options'] = SubscriptionStatus(
                    name='OPRA OPTIONS FEED (Prueba)',
                    is_valid=False,
                    error_msg='No option data - Check OPRA subscription'
                )
        else:
            results['opra_options'] = SubscriptionStatus(
                name='OPRA OPTIONS FEED (Prueba)',
                is_valid=False,
                error_msg='Cannot test - Index feed failed first'
            )
        
        return results
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures disconnect."""
        self.disconnect()
        return False
