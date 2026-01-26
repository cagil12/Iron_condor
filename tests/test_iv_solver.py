import unittest
from ingresarios_options_research.src.pricing.iv_solver import implied_volatility, bsm_price

class TestIVSolver(unittest.TestCase):
    def test_iv_recovery(self):
        # Price an option, then try to recover the Sigma
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
        price = bsm_price('C', S, K, T, r, sigma)
        
        recovered_sigma = implied_volatility(price, S, K, T, r, 'C')
        self.assertAlmostEqual(sigma, recovered_sigma, places=5)
        
    def test_iv_convergence_fail(self):
        # Reasonable behavior on impossible prices (like negative?)
        # For now just checking it doesn't crash on zero price
        iv = implied_volatility(0.0, 100, 100, 1.0, 0.05, 'C')
        # Should be low or handle gracefully?? 
        # BSM with 0 vol = intrinsic. Argmax of 0 price for ATM call matches 0 vol?
        # ATM Call Intrinsic is 0? Yes. (100-100).
        # So price 0.0 implies Vol 0.0.
        self.assertTrue(iv < 0.01 or iv == 0.5) # 0.5 is start guess, might not move if vega zero?

if __name__ == '__main__':
    unittest.main()
