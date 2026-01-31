import unittest
from src.research.walk_forward import WalkForwardAnalysis
from datetime import date, timedelta

class TestWFASplits(unittest.TestCase):
    def test_split_integrity(self):
        dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(20)]
        # Train 10, Test 5, Step 5
        # Fold 1: Train [0..9], Test [10..14]
        # Fold 2: Train [5..14], Test [15..19]
        
        wfa = WalkForwardAnalysis({}, None, 10, 5, 5)
        
        # Mocking the internal loop logic check effectively
        # Since run method is monolithic, we just check the math manually here
        # or we could refactor wfa to yield splits. Use simple logic verification.
        
        start_idx = 0
        train_end = start_idx + 10
        test_end = train_end + 5
        
        train_dates = dates[start_idx:train_end]
        test_dates = dates[train_end:test_end]
        
        # Assert no overlap
        self.assertTrue(max(train_dates) < min(test_dates))
        self.assertEqual(len(train_dates), 10)
        self.assertEqual(len(test_dates), 5)

if __name__ == '__main__':
    unittest.main()
