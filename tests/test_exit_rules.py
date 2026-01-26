import unittest
from datetime import datetime, date, timedelta
from ingresarios_options_research.src.strategy.exits import ExitManager, TradeExit
from ingresarios_options_research.src.strategy.condor_builder import IronCondorTrade, Leg
from ingresarios_options_research.src.data.schema import OptionChain, OptionType, Quote

class TestExitRules(unittest.TestCase):
    def test_stop_loss(self):
        # Trade with 1.0 credit
        legs = [Leg(OptionType.PUT, 100, False, 1.0)] # Dummy leg
        trade = IronCondorTrade(datetime.now(), legs, 1.0, 10.0)
        
        # SL Multiplier 2.0 -> Exit if Debit >= 2.0
        manager = ExitManager({'stop_loss_mult': 2.0})
        
        chain = OptionChain(datetime.now(), 100, date.today())
        # Current Price to Close Short: Ask = 2.5
        chain.quotes[(100, OptionType.PUT)] = Quote(2.4, 2.5)
        
        exit_signal = manager.check_exit(trade, chain)
        self.assertIsNotNone(exit_signal)
        self.assertEqual(exit_signal.exit_reason, 'SL')
        self.assertEqual(exit_signal.exit_price, 2.5)

if __name__ == '__main__':
    unittest.main()
