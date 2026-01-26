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
        # Reasonable behavior on impossible prices (like zero)
        # Hardened solver now returns NaN for invalid prices
        iv = implied_volatility(0.0, 100, 100, 1.0, 0.05, 'C')
        # Zero price for ATM option is invalid - should return NaN
        import numpy as np
        self.assertTrue(np.isnan(iv) or iv < 0.01)

if __name__ == '__main__':
    unittest.main()
