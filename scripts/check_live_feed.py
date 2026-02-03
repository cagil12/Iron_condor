#!/usr/bin/env python3
"""
check_live_feed.py

Diagnostic script to verify IBKR connection and data subscriptions.
Run this before starting live trading to ensure all feeds are active.

Usage:
    python scripts/check_live_feed.py          # Paper trading (default)
    python scripts/check_live_feed.py --live   # Live trading
"""
import sys
import time
import argparse
from datetime import datetime

# Add project root to path
sys.path.insert(0, '.')

from src.data.ib_connector import IBConnector
from src.utils.config import get_live_config


def print_header():
    """Print diagnostic header."""
    print("\n" + "═" * 60)
    print("  🔍 IBKR LIVE FEED DIAGNOSTIC - XSP Mini-SPX")
    print("═" * 60 + "\n")


def print_status(label: str, value: str, is_ok: bool):
    """Print a status line with emoji indicator."""
    status = "✅" if is_ok else "❌"
    print(f"  {label}: {value} {status}")


def run_diagnostic(paper: bool = True):
    """
    Run the live feed diagnostic.
    
    Args:
        paper: True for paper trading, False for live
    """
    print_header()
    config = get_live_config()
    
    connector = IBConnector(config.get('ibkr', {}))
    
    try:
        # ══════════════════════════════════════════════════════════════════
        # STEP 1: Connect to TWS
        # ══════════════════════════════════════════════════════════════════
        mode = "PAPER" if paper else "🔴 LIVE"
        print(f"🔌 CONECTANDO A TWS ({mode})...", end=" ", flush=True)
        
        success, msg = connector.connect(paper=paper)
        
        if not success:
            print(f"FALLÓ\n   ❌ {msg}")
            print("\n   💡 Asegúrate de que TWS esté corriendo y:")
            print("      - API habilitado en Configure > API > Settings")
            print(f"      - Puerto {'7497 (Paper)' if paper else '7496 (Live)'} activo")
            print("      - 'Read-Only API' deshabilitado")
            return False
        
        print("OK")
        print(f"   👤 Cuenta: {connector.account_id}")
        
        # ══════════════════════════════════════════════════════════════════
        # STEP 2: Check Account Capital
        # ══════════════════════════════════════════════════════════════════
        capital = connector.get_account_value()
        max_contracts = int(capital / 100)  # Rough estimate: $100 per contract margin
        
        print(f"\n💰 CAPITAL: ${capital:,.2f}", end=" ")
        
        if capital >= config['max_capital']:
            print(f"(Suficiente para {max_contracts} contratos XSP) ✅")
        else:
            print(f"⚠️ (Mínimo recomendado: ${config['max_capital']})")
        
        # ══════════════════════════════════════════════════════════════════
        # STEP 3: Validate Data Subscriptions
        # ══════════════════════════════════════════════════════════════════
        print("\n📡 VALIDANDO SUSCRIPCIONES:")
        print("-" * 40)
        
        subscriptions = connector.verify_subscriptions()
        all_valid = True
        
        # CBOE Index Feed
        idx_status = subscriptions.get('cboe_index')
        if idx_status:
            if idx_status.is_valid:
                print(f"  [1] {idx_status.name}:  {idx_status.last_price:.2f}  ✅ (Data Viva)")
            else:
                print(f"  [1] {idx_status.name}:  ❌ FALLÓ")
                print(f"      🚨 ALERTA ROJA: {idx_status.error_msg}")
                all_valid = False
        
        # OPRA Options Feed
        opt_status = subscriptions.get('opra_options')
        if opt_status:
            if opt_status.is_valid:
                print(f"  [2] {opt_status.name}:  Bid: {opt_status.bid:.2f} / Ask: {opt_status.ask:.2f} ✅ (Data Viva)")
            else:
                print(f"  [2] {opt_status.name}:  ❌ FALLÓ")
                print(f"      🚨 ALERTA ROJA: {opt_status.error_msg}")
                all_valid = False
        
        # ══════════════════════════════════════════════════════════════════
        # FINAL STATUS
        # ══════════════════════════════════════════════════════════════════
        print("\n" + "═" * 60)
        
        if all_valid:
            print("🟢 SISTEMA LISTO PARA OPERAR.")
            print("═" * 60 + "\n")
            
            # Continuous monitoring loop
            print("📊 Modo Monitor Continuo (Ctrl+C para salir):\n")
            
            try:
                while True:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    xsp_price = connector.get_live_price('XSP')
                    
                    if xsp_price and xsp_price > 0:
                        print(f"  [{timestamp}] XSP: ${xsp_price:.2f}", end="\r")
                    else:
                        print(f"  [{timestamp}] XSP: --- (waiting)", end="\r")
                    
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print("\n\n👋 Monitor detenido.")
        else:
            print("🔴 SISTEMA NO LISTO - Revisa las suscripciones faltantes.")
            print("═" * 60)
            print("\n💡 Para activar suscripciones en IBKR:")
            print("   1. Ir a Account Management > Settings > Market Data")
            print("   2. Suscribirse a:")
            print("      - CBOE Indices (para XSP spot)")
            print("      - OPRA (US Options) (para opciones)")
            print("   3. Esperar 24h para activación")
            return False
        
        return True
        
    finally:
        # Always disconnect
        connector.disconnect()
        print("🔌 Desconectado de TWS.")


def main():
    parser = argparse.ArgumentParser(description='IBKR Live Feed Diagnostic')
    parser.add_argument('--live', action='store_true', 
                        help='Connect to live trading (default: paper)')
    args = parser.parse_args()
    
    if args.live:
        print("\n⚠️  ATENCIÓN: Conectando a cuenta LIVE. Ctrl+C para cancelar...")
        time.sleep(3)
    
    success = run_diagnostic(paper=not args.live)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
