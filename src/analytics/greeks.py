import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

class BlackScholesSolver:
    """
    Calculadora de Griegas y Volatilidad Implícita.
    Esencial para convertir precios brutos (TBBO) en decisiones de trading (Delta).
    """

    def __init__(self, risk_free_rate=0.0525):  # Tasa actual aprox 5.25%
        self.r = risk_free_rate

    def d1_d2(self, S, K, T, sigma):
        """Cálculo intermedio estándar de BS."""
        if T <= 0 or sigma <= 0:
            return 0, 0
        
        d1 = (np.log(S / K) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    def price_option(self, option_type, S, K, T, sigma):
        """Calcula el precio teórico."""
        if T <= 0:
            return max(0, S - K) if option_type == 'call' else max(0, K - S)
        
        d1, d2 = self.d1_d2(S, K, T, sigma)
        
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-self.r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def calculate_delta(self, option_type, S, K, T, sigma):
        """
        El Santo Grial: Nos dice la probabilidad de que la opción expire ITM.
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1, _ = self.d1_d2(S, K, T, sigma)
        
        if option_type == 'call':
            return norm.cdf(d1)
        else:  # Put
            return norm.cdf(d1) - 1.0

    def implied_volatility(self, option_type, option_price, S, K, T):
        """
        Resuelve la IV dado el precio de mercado.
        Usa el método Brent para encontrar la raíz (más estable que Newton).
        """
        if T <= 0:
            return 0.0
        if option_price <= 0:
            return 0.0
            
        # Límites de búsqueda para IV (1% a 400%)
        low, high = 0.01, 4.0
        
        def objective_function(sigma):
            return self.price_option(option_type, S, K, T, sigma) - option_price

        try:
            # Si el precio de mercado está fuera de los límites teóricos, falla rápido
            price_low = self.price_option(option_type, S, K, T, low)
            price_high = self.price_option(option_type, S, K, T, high)
            
            if option_price < price_low:
                return low
            if option_price > price_high:
                return high

            return brentq(objective_function, low, high, xtol=1e-6)
        except Exception:
            return float('nan')  # ESTUDIO TITO: No fallbacks, propagate failure

    def calculate_price(self, option_type, S, K, T, sigma):
        """Alias for price_option - used by simulator fallback."""
        return self.price_option(option_type, S, K, T, sigma)
