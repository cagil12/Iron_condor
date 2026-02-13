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
    'entry_time': '09:30',    # Entry Time (ET) - Updated for Open

    # Instrument Settings
    'symbol': 'XSP',
    'exchange': 'CBOE',
    'currency': 'USD',
    'contract_multiplier': 100,  # XSP is also x100
    
    # Strategy Parameters (Scaled for XSP)
    'wing_width': 2.0,  # 2-point wings (matched to base.yaml)
    'target_delta': 0.10,
    'min_credit': 0.20,  # $20 min credit per contract (ratio 4:1 max)
    'min_days_expiry': 0, # 0DTE
    
    # Risk Management
    'max_capital': 2000.0,  # $2,000 USD account size
    'max_contracts': 1,    # Conservative start
    'max_daily_loss': 200.0,  # $200 daily loss limit (10% of account)
    'max_vix': 25.0,       # KILL SWITCH: No trading if VIX > 25
    'min_account_value': 1400.0, # HARD SWITCH: Equity Floor ($1400)
    
    # Exit Parameters (Phase 1)
    'take_profit_pct': 1.00,  # Hold winners to expiry (skip TP close orders)
    'stop_loss_mult': 2.0,    # Cut losers at 2x collected credit
    'commission_per_leg': 0.65,  # IBKR fixed pricing per contract-leg
    'legs_per_ic': 4,            # Iron condor = 4 legs
    
    # IBKR Connection
    'ibkr': {
        'host': '127.0.0.1',
        'paper_port': 7497,  # TWS Paper Trading
        'live_port': 7497,   # TWS Live Trading (User Override)
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
