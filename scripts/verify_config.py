
"""
verify_config.py
Simple check to confirm config is pointing to Paper Trading.
"""
import sys
import os

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import get_live_config

def main():
    config = get_live_config()
    
    print("\n🔍 VERIFYING CONFIGURATION FOR TOMORROW...\n")
    print(f"  Trading Mode:  {config.get('trading_mode')}")
    print(f"  Account ID:    {config.get('account_id')}")
    print(f"  TWS Port:      {config.get('port')}")
    print(f"  Entry Time:    {config.get('entry_time')}")
    print(f"  Max Capital:   ${config.get('max_capital')}")
    print("-" * 40)
    
    errors = []
    if config.get('trading_mode') != 'PAPER':
        errors.append("❌ TRADING_MODE Should be 'PAPER'")
    if config.get('account_id') != 'DUO988990':
        errors.append("❌ ACCOUNT_ID Should be 'DUO988990'")
    if config.get('port') != 7497:
        errors.append("❌ PORT Should be 7497")
    if config.get('entry_time') != '10:00':
        errors.append("❌ ENTRY_TIME Should be '10:00'")
        
    if errors:
        for e in errors:
            print(e)
        print("\n⛔ CONFIGURATION FAILED!")
        sys.exit(1)
    else:
        print("✅ ALL CHECKS PASSED. Ready for Paper Trading.")
        sys.exit(0)

if __name__ == "__main__":
    main()
