#!/usr/bin/env python3
"""
Test risk management calculations
"""

from risk_management import RiskManager

# Initialize risk manager
risk_mgr = RiskManager()

# Test SHORT trade (like the one in the diagnostic)
print("="*70)
print("TEST: SHORT TRADE")
print("="*70)

trade_params = risk_mgr.calculate_trade_parameters(
    signal='short',
    confidence=1.0,
    entry_price=4502.0,
    atr=15.0,  # 15 points ATR
    regime='trending_high_vol',
    account_balance=25000.00
)

print("\nTrade Parameters:")
print(f"  Signal: {trade_params['signal'].upper()}")
print(f"  Entry: ${trade_params['entry_price']:.2f}")
print(f"  Stop Loss: ${trade_params['stop_loss']:.2f}")
print(f"  Take Profit: ${trade_params['take_profit']:.2f}")
print(f"  Stop Distance: {trade_params['stop_distance']:.2f} points")
print(f"  Target Distance: {trade_params['target_distance']:.2f} points")
print(f"  Risk/Reward: {trade_params['risk_reward']:.2f}")
print(f"  Contracts: {trade_params['contracts']}")
print(f"  Risk: ${trade_params['risk_dollars']:.2f} ({trade_params['risk_pct']:.2%})")
print(f"  Regime: {trade_params['regime']}")

print("\n" + "="*70)
print("EXPECTED FOR SHORT:")
print("  Stop Loss should be ABOVE entry (e.g., 4540)")
print("  Take Profit should be BELOW entry (e.g., 4465)")
print("="*70)

print("\n" + "="*70)
print("TEST: LONG TRADE")
print("="*70)

trade_params = risk_mgr.calculate_trade_parameters(
    signal='long',
    confidence=0.75,
    entry_price=4500.0,
    atr=15.0,
    regime='trending_normal',
    account_balance=25000.00
)

print("\nTrade Parameters:")
print(f"  Signal: {trade_params['signal'].upper()}")
print(f"  Entry: ${trade_params['entry_price']:.2f}")
print(f"  Stop Loss: ${trade_params['stop_loss']:.2f}")
print(f"  Take Profit: ${trade_params['take_profit']:.2f}")
print(f"  Stop Distance: {trade_params['stop_distance']:.2f} points")
print(f"  Target Distance: {trade_params['target_distance']:.2f} points")
print(f"  Risk/Reward: {trade_params['risk_reward']:.2f}")
print(f"  Contracts: {trade_params['contracts']}")
print(f"  Risk: ${trade_params['risk_dollars']:.2f} ({trade_params['risk_pct']:.2%})")

print("\n" + "="*70)
print("EXPECTED FOR LONG:")
print("  Stop Loss should be BELOW entry (e.g., 4475)")
print("  Take Profit should be ABOVE entry (e.g., 4545)")
print("="*70)
