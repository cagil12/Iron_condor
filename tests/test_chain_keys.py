import unittest
from datetime import datetime, date
from src.data.schema import OptionChain, OptionType, Quote

class TestChainKeys(unittest.TestCase):
    def test_keys_collision(self):
        # Ensure Call and Put with same strike don't overwrite each other
        chain = OptionChain(datetime.now(), 5000, date.today())
        
        q_call = Quote(1.0, 1.1)
        q_put = Quote(2.0, 2.1)
        
        chain.quotes[(5000, OptionType.CALL)] = q_call
        chain.quotes[(5000, OptionType.PUT)] = q_put
        
        self.assertEqual(len(chain.quotes), 2)
        self.assertEqual(chain.get_quote(5000, OptionType.CALL).bid, 1.0)
        self.assertEqual(chain.get_quote(5000, OptionType.PUT).bid, 2.0)

if __name__ == '__main__':
    unittest.main()
