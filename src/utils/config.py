import yaml
from pathlib import Path
from typing import Dict, Any
import hashlib

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def merge_configs(base: Dict, override: Dict) -> Dict:
    """Deep merge two config dicts."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result

def get_config_hash(config: Dict) -> str:
    """Generate hash of config for reproducibility tracking."""
    config_str = yaml.dump(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]

def save_config_used(config: Dict, output_path: str):
    """Save config with hash for audit trail (per spec J)."""
    config_with_meta = config.copy()
    config_with_meta['_meta'] = {
        'config_hash': get_config_hash(config),
        'saved_at': str(Path(output_path).resolve())
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config_with_meta, f, default_flow_style=False)


# ══════════════════════════════════════════════════════════════════════════════
# LIVE TRADING CONFIGURATION (XSP Mini-SPX)
# ══════════════════════════════════════════════════════════════════════════════

LIVE_CONFIG = {
    # Trading Environment
    'trading_mode': 'PAPER',  # 'PAPER' or 'LIVE'
    'account_id': 'DUO988990', # Explicit Paper Trading Account
    'port': 7497,             # Default to Paper Port
    'entry_time': '10:00',    # Entry Time (ET) - Aligned with backtest entry_hour=10.0

    # Instrument Settings
    'symbol': 'XSP',
    'exchange': 'CBOE',
    'currency': 'USD',
    'contract_multiplier': 100,  # XSP is also x100
    
    # Strategy Parameters (Scaled for XSP)
    'wing_width': 2.0,  # 2-point wings (matched to base.yaml)
    'target_delta': 0.10,
    'min_credit': 0.18,  # $18 min credit floor for 2-wide IC; prefer higher when available
    'min_days_expiry': 0, # 0DTE
    
    # Risk Management
    'max_capital': 3000.0,  # $3,000 USD account size (risk housekeeping recalibration)
    'max_contracts': 1,    # Conservative start
    'max_daily_loss': 200.0,  # $200 daily loss limit (10% of account)
    'min_account_value': 2000.0, # HARD SWITCH: Equity Floor ($2000)
    # Kill Switches (Phase 2 prerequisites)
    'dd_max_pct': 0.15,              # L1: Max drawdown as % of max_capital
    'dd_pause_days': 5,              # L1: Calendar days pause after DD breach
    'dd_kill_enabled': True,         # L1: Portfolio drawdown kill switch
    'vix_gate_threshold': 25.0,      # L3: Do not trade if VIX > this value
    'vix_gate_enabled': True,        # L3: Volatility regime gate
    'streak_max_losses': 3,          # L5: Pause after N consecutive losses
    'streak_pause_days': 2,          # L5: Calendar days to pause after streak
    'streak_stop_enabled': True,     # L5: Consecutive loss streak stop
    
    # Exit Parameters (Phase 1)
    'take_profit_pct': 1.00,  # Hold winners to expiry (skip TP close orders)
    'stop_loss_mult': 2.0,    # Cut losers at 2x collected credit
    'commission_per_leg': 0.65,  # IBKR fixed pricing per contract-leg
    'legs_per_ic': 4,            # Iron condor = 4 legs
    
    # IBKR Connection
    'ibkr': {
        'host': '127.0.0.1',
        'paper_port': 7497,  # TWS Paper Trading
        'live_port': 7497,   # ⚠️ CHANGE TO 7496 BEFORE LIVE MIGRATION (standard TWS live port)
        'client_id': 777,
        'timeout': 30,
    },
    
    # Data Subscriptions Required
    'subscriptions': {
        'index': 'CBOE Indices',
        'options': 'OPRA (US Options)',
    }
}

def get_live_config() -> Dict[str, Any]:
    """Get live trading configuration."""
    return LIVE_CONFIG.copy()
