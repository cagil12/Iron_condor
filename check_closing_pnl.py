from ib_insync import *
from datetime import datetime, timedelta

def get_closing_prices():
    ib = IB()
    try:
        ib.connect('127.0.0.1', 7497, clientId=97)
        
        # Contracts
        expiry = '20260206'
        legs = [
            Option('XSP', expiry, 668, 'P', 'SMART'), # Short Put
            Option('XSP', expiry, 667, 'P', 'SMART'), # Long Put
            Option('XSP', expiry, 695, 'C', 'SMART'), # Short Call
            Option('XSP', expiry, 696, 'C', 'SMART')  # Long Call
        ]
        
        ib.qualifyContracts(*legs)
        
        print(f"\n--- Hypothetical Exit Prices (15:45 vs 16:15) ---")
        
        for leg in legs:
            # Request historical data for today
            end_time = '' 
            bars = ib.reqHistoricalData(
                leg, 
                endDateTime='', 
                durationStr='1 D', 
                barSizeSetting='5 mins', # 5 min bars for today
                whatToShow='MIDPOINT', 
                useRTH=True
            )
            
            if not bars:
                print(f"No data for {leg.localSymbol}")
                continue

            print(f"--- Data for {leg.right}{leg.strike} (Last 10 bars) ---")
            for bar in bars[-10:]:
                 print(f"Time: {bar.date} | Close: {bar.close}")

        ib.disconnect()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    get_closing_prices()
