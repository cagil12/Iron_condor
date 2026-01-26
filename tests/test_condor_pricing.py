import unittest
from ingresarios_options_research.src.strategy.condor_builder import CondorBuilder, IronCondorTrade
from ingresarios_options_research.src.data.schema import OptionChain, OptionType, Quote
from datetime import datetime, date

class TestCondorPricing(unittest.TestCase):
    def test_entry_credit_calc(self):
        # Mock Chain
        chain = OptionChain(datetime.now(), 5000, date.today())
        
        # Setup specific quotes to verify math
        # Short Put 4900: Bid 10.0, Ask 11.0
        chain.quotes[(4900, OptionType.PUT)] = Quote(10.0, 11.0, delta=-0.10)
        # Long Put 4880: Bid 5.0, Ask 6.0
        chain.quotes[(4880, OptionType.PUT)] = Quote(5.0, 6.0, delta=-0.05)
        # Short Call 5100: Bid 10.0, Ask 11.0
        chain.quotes[(5100, OptionType.CALL)] = Quote(10.0, 11.0, delta=0.10)
        # Long Call 5120: Bid 5.0, Ask 6.0
        chain.quotes[(5120, OptionType.CALL)] = Quote(5.0, 6.0, delta=0.05)
        
        builder = CondorBuilder({
            'target_delta': 0.10, 
            'width': 20, 
            'min_credit': 0.0, 
            'min_ror': 0.0,
            'max_spread_pct': 1.0 # Allow wide spreads for this manual test case
        })
        trade = builder.build_trade(chain)
        
        self.assertIsNotNone(trade)
        
        # Put Credit = Abs(Short Bid - Long Ask) = 10.0 - 6.0 = 4.0?
        # Wait, Specs: Put credit = Bid(short_put) - Ask(long_put)
        # 10.0 - 6.0 = 4.0. Correct.
        
        # Call Credit = Bid(short_call) - Ask(long_call)
        # 10.0 - 6.0 = 4.0.
        
        # Total = 8.0
        self.assertAlmostEqual(trade.entry_credit, 8.0)
        
        # Max Loss = width - credit = 20 - 8 = 12
        self.assertAlmostEqual(trade.max_loss, 12.0)

if __name__ == '__main__':
    unittest.main()
