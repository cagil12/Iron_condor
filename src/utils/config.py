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
