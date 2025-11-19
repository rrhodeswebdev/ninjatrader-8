"""
Futures Contract Configuration

Provides contract specifications for different futures instruments.
Makes it easy to switch between ES, NQ, and other contracts.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ContractSpec:
    """Specifications for a futures contract"""
    symbol: str
    name: str
    tick_size: float          # Minimum price increment
    point_value: float        # Dollar value per point
    tick_value: float         # Dollar value per tick
    initial_margin: float     # Initial margin requirement
    maintenance_margin: float # Maintenance margin
    overnight_margin: float   # Overnight margin
    exchange: str
    trading_hours: str


# Contract specifications
CONTRACTS: Dict[str, ContractSpec] = {
    'ES': ContractSpec(
        symbol='ES',
        name='E-mini S&P 500',
        tick_size=0.25,
        point_value=50.0,        # $50 per point
        tick_value=12.50,        # $12.50 per tick (0.25 * $50)
        initial_margin=12500.0,
        maintenance_margin=11400.0,
        overnight_margin=12500.0,
        exchange='CME',
        trading_hours='9:30 AM - 4:00 PM ET (RTH)'
    ),

    'NQ': ContractSpec(
        symbol='NQ',
        name='E-mini NASDAQ-100',
        tick_size=0.25,
        point_value=20.0,        # $20 per point
        tick_value=5.0,          # $5 per tick (0.25 * $20)
        initial_margin=17600.0,
        maintenance_margin=16000.0,
        overnight_margin=17600.0,
        exchange='CME',
        trading_hours='9:30 AM - 4:00 PM ET (RTH)'
    ),

    'YM': ContractSpec(
        symbol='YM',
        name='E-mini Dow',
        tick_size=1.0,
        point_value=5.0,         # $5 per point
        tick_value=5.0,          # $5 per tick
        initial_margin=10450.0,
        maintenance_margin=9500.0,
        overnight_margin=10450.0,
        exchange='CBOT',
        trading_hours='9:30 AM - 4:00 PM ET (RTH)'
    ),

    'RTY': ContractSpec(
        symbol='RTY',
        name='E-mini Russell 2000',
        tick_size=0.10,
        point_value=50.0,        # $50 per point
        tick_value=5.0,          # $5 per tick (0.10 * $50)
        initial_margin=5500.0,
        maintenance_margin=5000.0,
        overnight_margin=5500.0,
        exchange='CME',
        trading_hours='9:30 AM - 4:00 PM ET (RTH)'
    ),

    'MES': ContractSpec(
        symbol='MES',
        name='Micro E-mini S&P 500',
        tick_size=0.25,
        point_value=5.0,         # $5 per point (1/10 of ES)
        tick_value=1.25,         # $1.25 per tick
        initial_margin=1250.0,
        maintenance_margin=1140.0,
        overnight_margin=1250.0,
        exchange='CME',
        trading_hours='9:30 AM - 4:00 PM ET (RTH)'
    ),

    'MNQ': ContractSpec(
        symbol='MNQ',
        name='Micro E-mini NASDAQ-100',
        tick_size=0.25,
        point_value=2.0,         # $2 per point (1/10 of NQ)
        tick_value=0.50,         # $0.50 per tick
        initial_margin=1760.0,
        maintenance_margin=1600.0,
        overnight_margin=1760.0,
        exchange='CME',
        trading_hours='9:30 AM - 4:00 PM ET (RTH)'
    ),
}


def get_contract(symbol: str) -> ContractSpec:
    """
    Get contract specifications by symbol.

    Args:
        symbol: Contract symbol (ES, NQ, YM, RTY, MES, MNQ)

    Returns:
        ContractSpec with all specifications

    Raises:
        ValueError: If contract symbol not found
    """
    symbol = symbol.upper()

    if symbol not in CONTRACTS:
        available = ', '.join(CONTRACTS.keys())
        raise ValueError(
            f"Unknown contract: {symbol}\n"
            f"Available contracts: {available}"
        )

    return CONTRACTS[symbol]


def list_contracts() -> None:
    """Print all available contracts and their specs"""
    print("\n" + "="*70)
    print("AVAILABLE FUTURES CONTRACTS")
    print("="*70)

    for symbol, spec in CONTRACTS.items():
        print(f"\n{symbol} - {spec.name}")
        print(f"  Tick Size:         ${spec.tick_size:.2f}")
        print(f"  Point Value:       ${spec.point_value:.2f}")
        print(f"  Tick Value:        ${spec.tick_value:.2f}")
        print(f"  Initial Margin:    ${spec.initial_margin:,.2f}")
        print(f"  Maintenance:       ${spec.maintenance_margin:,.2f}")
        print(f"  Exchange:          {spec.exchange}")
        print(f"  Trading Hours:     {spec.trading_hours}")

    print("\n" + "="*70 + "\n")


def calculate_pnl(
    entry_price: float,
    exit_price: float,
    direction: str,
    contracts: int,
    symbol: str = 'ES'
) -> float:
    """
    Calculate P&L for a trade.

    Args:
        entry_price: Entry price
        exit_price: Exit price
        direction: 'long' or 'short'
        contracts: Number of contracts
        symbol: Contract symbol (default: ES)

    Returns:
        P&L in dollars
    """
    spec = get_contract(symbol)

    if direction.lower() == 'long':
        points = exit_price - entry_price
    else:
        points = entry_price - exit_price

    pnl = points * contracts * spec.point_value

    return pnl


def calculate_margin_requirement(
    contracts: int,
    symbol: str = 'ES',
    overnight: bool = False
) -> float:
    """
    Calculate margin requirement.

    Args:
        contracts: Number of contracts
        symbol: Contract symbol (default: ES)
        overnight: Use overnight margin (default: False)

    Returns:
        Required margin in dollars
    """
    spec = get_contract(symbol)

    if overnight:
        margin_per_contract = spec.overnight_margin
    else:
        margin_per_contract = spec.initial_margin

    return contracts * margin_per_contract


if __name__ == '__main__':
    # Display all available contracts
    list_contracts()

    # Example usage
    print("Example: ES Trade P&L")
    print("-" * 40)
    pnl = calculate_pnl(
        entry_price=4800.0,
        exit_price=4810.0,
        direction='long',
        contracts=1,
        symbol='ES'
    )
    print(f"Long 1 ES @ 4800, Exit @ 4810")
    print(f"P&L: ${pnl:.2f}")

    print("\nExample: NQ Trade P&L")
    print("-" * 40)
    pnl_nq = calculate_pnl(
        entry_price=17000.0,
        exit_price=17050.0,
        direction='long',
        contracts=1,
        symbol='NQ'
    )
    print(f"Long 1 NQ @ 17000, Exit @ 17050")
    print(f"P&L: ${pnl_nq:.2f}")

    print("\nExample: Margin Requirements")
    print("-" * 40)
    margin_es = calculate_margin_requirement(2, 'ES')
    margin_nq = calculate_margin_requirement(2, 'NQ')
    print(f"2 ES contracts: ${margin_es:,.2f}")
    print(f"2 NQ contracts: ${margin_nq:,.2f}")
