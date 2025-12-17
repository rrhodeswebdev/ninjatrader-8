"""
Trade log matcher for RNN backtester vs backintime exports vs actual executions.

Usage:
    # Compare RNN vs backintime:
    python trade_log_matcher.py \
        --rnn results/rnn_backtester_trades.csv \
        --bti results/backintime/backintime_trades_converted.csv \
        --tol-min 2 \
        --max-offset-min 240 \
        --point-value 2.0

    # Parse actual NinjaTrader executions:
    python trade_log_matcher.py --actual ../backtester/data/NinjaTraderGrid2025-12-8-12_executions.csv

The script:
  - Parses NinjaTrader execution CSVs and calculates actual P&L with commissions.
  - Parses both CSVs (expects entry_time/exit_time columns; backintime file is the converted log).
  - Sweeps time offsets (in minutes) to find the offset with the most matched trades.
  - Prints summary counts and sample unmatched trades for the best offset.
  - Reports simple P&L deltas for matched trades using the given point value.
"""

import argparse
from datetime import timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import pandas as pd
import numpy as np


# =============================================================================
# NinjaTrader Actual Execution Parsing
# =============================================================================

def parse_ninja_executions(csv_path: str, point_value: float = 2.0) -> pd.DataFrame:
    """
    Parse NinjaTrader execution CSV into paired trades.

    Args:
        csv_path: Path to NinjaTrader execution CSV
        point_value: Dollar value per point (default: 2.0 for MNQ)

    Returns:
        DataFrame with paired entry/exit trades and P&L
    """
    df = pd.read_csv(csv_path)

    # Parse commission (remove $ sign)
    df['Commission'] = df['Commission'].str.replace('$', '', regex=False).astype(float)

    # Parse time
    df['Time'] = pd.to_datetime(df['Time'], format='%m/%d/%Y %I:%M:%S %p')

    # Sort by time
    df = df.sort_values('Time').reset_index(drop=True)

    # Pair entries with exits
    trades = []
    i = 0
    while i < len(df) - 1:
        entry = df.iloc[i]
        exit_row = df.iloc[i + 1]

        # Validate pairing: entry E/X should be Entry, exit should be Exit
        if entry['E/X'] != 'Entry' or exit_row['E/X'] != 'Exit':
            print(f"Warning: Unexpected pairing at index {i}: {entry['E/X']} / {exit_row['E/X']}")
            i += 1
            continue

        # Determine direction
        if entry['Action'] == 'Buy':
            direction = 'long'
            pnl_points = exit_row['Price'] - entry['Price']
        else:  # Sell
            direction = 'short'
            pnl_points = entry['Price'] - exit_row['Price']

        gross_pnl = pnl_points * entry['Quantity'] * point_value
        total_commission = entry['Commission'] + exit_row['Commission']
        net_pnl = gross_pnl - total_commission

        trades.append({
            'entry_time': entry['Time'],
            'exit_time': exit_row['Time'],
            'direction': direction,
            'contracts': entry['Quantity'],
            'entry_price': entry['Price'],
            'exit_price': exit_row['Price'],
            'pnl_points': pnl_points,
            'gross_pnl': gross_pnl,
            'commission': total_commission,
            'pnl': net_pnl,  # 'pnl' key for consistency with backtesters
            'entry_id': entry['ID'],
            'exit_id': exit_row['ID']
        })

        i += 2

    return pd.DataFrame(trades)


def calculate_actual_stats(trades_df: pd.DataFrame) -> Dict:
    """Calculate statistics from actual execution trades."""
    if len(trades_df) == 0:
        return {'total_trades': 0}

    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] < 0]

    return {
        'total_trades': len(trades_df),
        'winning_trades': len(wins),
        'losing_trades': len(losses),
        'win_rate': len(wins) / len(trades_df) if len(trades_df) > 0 else 0,
        'total_pnl': trades_df['pnl'].sum(),
        'gross_pnl': trades_df['gross_pnl'].sum(),
        'total_commission': trades_df['commission'].sum(),
        'avg_trade_pnl': trades_df['pnl'].mean(),
        'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
        'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
        'largest_win': trades_df['pnl'].max(),
        'largest_loss': trades_df['pnl'].min(),
        'profit_factor': abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else float('inf'),
        'long_trades': len(trades_df[trades_df['direction'] == 'long']),
        'short_trades': len(trades_df[trades_df['direction'] == 'short']),
        'avg_holding_mins': (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds().mean() / 60
    }


def daily_breakdown(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Get daily P&L breakdown."""
    if len(trades_df) == 0:
        return pd.DataFrame()

    trades_df = trades_df.copy()
    trades_df['date'] = trades_df['entry_time'].dt.date

    daily = trades_df.groupby('date').agg({
        'pnl': ['sum', 'count'],
        'gross_pnl': 'sum',
        'commission': 'sum'
    }).round(2)

    daily.columns = ['net_pnl', 'trade_count', 'gross_pnl', 'commission']
    return daily


def print_actual_summary(trades_df: pd.DataFrame, stats: Dict):
    """Print summary of actual execution trades."""
    print("\n" + "=" * 70)
    print("ACTUAL EXECUTION SUMMARY")
    print("=" * 70)
    print(f"\nTotal Trades:        {stats['total_trades']}")
    print(f"Winning Trades:      {stats['winning_trades']} ({stats['win_rate']*100:.1f}%)")
    print(f"Losing Trades:       {stats['losing_trades']}")
    print(f"\nLong Trades:         {stats['long_trades']}")
    print(f"Short Trades:        {stats['short_trades']}")
    print(f"\nGross P&L:           ${stats['gross_pnl']:,.2f}")
    print(f"Total Commission:    ${stats['total_commission']:,.2f}")
    print(f"Net P&L:             ${stats['total_pnl']:,.2f}")
    print(f"\nAvg Trade P&L:       ${stats['avg_trade_pnl']:,.2f}")
    print(f"Avg Win:             ${stats['avg_win']:,.2f}")
    print(f"Avg Loss:            ${stats['avg_loss']:,.2f}")
    print(f"Largest Win:         ${stats['largest_win']:,.2f}")
    print(f"Largest Loss:        ${stats['largest_loss']:,.2f}")
    print(f"Profit Factor:       {stats['profit_factor']:.2f}")
    print(f"Avg Holding Time:    {stats['avg_holding_mins']:.1f} mins")

    # Daily breakdown
    print("\n" + "=" * 70)
    print("DAILY BREAKDOWN")
    print("=" * 70)
    daily = daily_breakdown(trades_df)
    print(daily.to_string())

    # Sample trades
    print("\n" + "=" * 70)
    print("SAMPLE TRADES (first 10)")
    print("=" * 70)
    print(trades_df[['entry_time', 'direction', 'entry_price', 'exit_price', 'pnl']].head(10).to_string())


def print_comparison_report(
    actual_stats: Dict,
    backtester_stats: Optional[Dict] = None,
    backintime_stats: Optional[Dict] = None
):
    """Print a comparison report between actual and backtested results."""
    print("\n" + "=" * 70)
    print("ACTUAL vs BACKTESTER COMPARISON")
    print("=" * 70)

    # Build comparison rows
    metrics = [
        ('Total Trades', 'total_trades', 'd'),
        ('Winning Trades', 'winning_trades', 'd'),
        ('Losing Trades', 'losing_trades', 'd'),
        ('Win Rate', 'win_rate', '.1%'),
        ('Total P&L', 'total_pnl', ',.2f'),
        ('Avg Trade P&L', 'avg_trade_pnl', ',.2f'),
        ('Avg Win', 'avg_win', ',.2f'),
        ('Avg Loss', 'avg_loss', ',.2f'),
        ('Largest Win', 'largest_win', ',.2f'),
        ('Largest Loss', 'largest_loss', ',.2f'),
        ('Profit Factor', 'profit_factor', '.2f'),
    ]

    # Header
    header = f"{'Metric':<20} {'Actual':>15}"
    if backtester_stats:
        header += f" {'RNN Backtester':>15} {'Diff':>10}"
    if backintime_stats:
        header += f" {'backintime':>15} {'Diff':>10}"
    print("\n" + header)
    print("-" * len(header))

    for label, key, fmt in metrics:
        actual_val = actual_stats.get(key, 0)

        if fmt == '.1%':
            actual_str = f"{actual_val*100:>14.1f}%"
        elif fmt == 'd':
            actual_str = f"{actual_val:>15d}"
        else:
            actual_str = f"${actual_val:>14{fmt}}"

        row = f"{label:<20} {actual_str}"

        if backtester_stats:
            bt_val = backtester_stats.get(key, 0)
            if fmt == '.1%':
                bt_str = f"{bt_val*100:>14.1f}%"
                diff = (bt_val - actual_val) * 100
                diff_str = f"{diff:>+9.1f}%"
            elif fmt == 'd':
                bt_str = f"{bt_val:>15d}"
                diff = bt_val - actual_val
                diff_str = f"{diff:>+10d}"
            else:
                bt_str = f"${bt_val:>14{fmt}}"
                diff = bt_val - actual_val
                diff_str = f"${diff:>+9.2f}"
            row += f" {bt_str} {diff_str}"

        if backintime_stats:
            bti_val = backintime_stats.get(key, 0)
            if fmt == '.1%':
                bti_str = f"{bti_val*100:>14.1f}%"
                diff = (bti_val - actual_val) * 100
                diff_str = f"{diff:>+9.1f}%"
            elif fmt == 'd':
                bti_str = f"{bti_val:>15d}"
                diff = bti_val - actual_val
                diff_str = f"{diff:>+10d}"
            else:
                bti_str = f"${bti_val:>14{fmt}}"
                diff = bti_val - actual_val
                diff_str = f"${diff:>+9.2f}"
            row += f" {bti_str} {diff_str}"

        print(row)

    print("=" * 70)


# =============================================================================
# Backtester Trade Comparison (RNN vs backintime)
# =============================================================================

def load_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["entry_time", "exit_time"])
    if "direction" not in df.columns:
        raise ValueError(f"Missing 'direction' column in {path}")
    return df.sort_values("entry_time").reset_index(drop=True)


def match_trades(
    rnn: pd.DataFrame,
    bti: pd.DataFrame,
    offset: timedelta,
    tol: timedelta,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Greedy one-to-one matcher by time/direction with given offset."""
    rnn_shift = rnn.copy()
    rnn_shift["entry_time"] = rnn_shift["entry_time"] + offset

    rnn_shift = rnn_shift.sort_values("entry_time").reset_index(drop=True)
    bti = bti.sort_values("entry_time").reset_index(drop=True)

    matches: List[Tuple[int, int]] = []
    unmatched_rnn: List[int] = []
    unmatched_bti: List[int] = []
    j = 0
    for i, rt in rnn_shift.iterrows():
        matched = False
        # advance unmatched bti if too early
        while j < len(bti) and bti.loc[j, "entry_time"] < rt["entry_time"] - tol:
            unmatched_bti.append(j)
            j += 1
        k = j
        while k < len(bti) and bti.loc[k, "entry_time"] <= rt["entry_time"] + tol:
            bt = bti.loc[k]
            if bt["direction"][0] == rt["direction"][0]:
                matches.append((i, k))
                matched = True
                j = k + 1
                break
            k += 1
        if not matched:
            unmatched_rnn.append(i)
    # remaining bti unmatched
    unmatched_bti.extend(range(j, len(bti)))
    return matches, unmatched_rnn, unmatched_bti


def calc_bti_pnl(row: pd.Series, point_value: float) -> float:
    if row["direction"] == "long":
        return (row["exit_price"] - row["entry_price"]) * point_value
    return (row["entry_price"] - row["exit_price"]) * point_value


def main() -> None:
    ap = argparse.ArgumentParser(description="Match RNN vs backintime trade logs, or parse actual NinjaTrader executions.")
    ap.add_argument("--actual", type=Path, help="Parse actual NinjaTrader execution CSV (standalone mode)")
    ap.add_argument("--rnn", type=Path, help="RNN trade CSV (rnn_backtester_trades.csv)")
    ap.add_argument("--bti", type=Path, help="Backintime converted trade CSV")
    ap.add_argument("--tol-min", type=int, default=2, help="Time tolerance in minutes for matching (default 2)")
    ap.add_argument("--max-offset-min", type=int, default=240, help="Max absolute offset to sweep in minutes (default 240)")
    ap.add_argument("--point-value", type=float, default=2.0, help="Point value for P&L comparison (default 2.0 for MNQ)")
    ap.add_argument("--output", type=Path, help="Output path for parsed trades CSV")
    args = ap.parse_args()

    # Mode 1: Parse actual NinjaTrader executions
    if args.actual:
        if not args.actual.exists():
            print(f"Error: File not found: {args.actual}")
            return

        print(f"\nParsing actual NinjaTrader executions from: {args.actual}")
        trades_df = parse_ninja_executions(str(args.actual), point_value=args.point_value)
        stats = calculate_actual_stats(trades_df)

        print_actual_summary(trades_df, stats)

        # Save parsed trades
        if args.output:
            output_path = args.output
        else:
            output_path = Path("results") / "actual_trades.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(output_path, index=False)
        print(f"\nParsed trades saved to: {output_path}")
        return

    # Mode 2: Compare RNN vs backintime (original behavior)
    if not args.rnn or not args.bti:
        print("Error: Either --actual or both --rnn and --bti are required")
        ap.print_help()
        return

    rnn = load_trades(args.rnn)
    bti = load_trades(args.bti)

    tol = timedelta(minutes=args.tol_min)
    best_score = -1
    best = None

    # Sweep offsets from -max to +max minutes, step 1 minute
    for off_min in range(-args.max_offset_min, args.max_offset_min + 1):
        offset = timedelta(minutes=off_min)
        matches, un_r, un_b = match_trades(rnn, bti, offset, tol)
        score = len(matches)
        if score > best_score:
            best_score = score
            best = {"offset": offset, "matches": matches, "un_r": un_r, "un_b": un_b}

    if best is None:
        print("No matches found at any offset.")
        return

    offset = best["offset"]
    matches = best["matches"]
    un_r = best["un_r"]
    un_b = best["un_b"]

    print(f"Best offset: {offset} (sweep ±{args.max_offset_min} minutes)")
    print(f"RNN trades: {len(rnn)}, Backintime trades: {len(bti)}")
    print(f"Matched: {len(matches)}, Unmatched RNN: {len(un_r)}, Unmatched BTI: {len(un_b)}")

    # Sample matches
    for i, (ir, ib) in enumerate(matches[:5]):
        rt = rnn.iloc[ir]
        bt = bti.iloc[ib]
        print(f"  Match {i+1}: RNN {rt['entry_time']} dir={rt['direction']} -> BTI {bt['entry_time']}")

    # Sample unmatched
    if un_r:
        print("\nFirst 5 unmatched RNN trades:")
        print(rnn.iloc[un_r[:5]][["entry_time", "direction", "pnl"]])
    if un_b:
        print("\nFirst 5 unmatched BTI trades:")
        print(bti.iloc[un_b[:5]][["entry_time", "direction"]])

    # P&L comparison for matched trades
    if matches:
        pnl_pairs = []
        for ir, ib in matches:
            rt = rnn.iloc[ir]
            bt = bti.iloc[ib]
            bti_pnl = calc_bti_pnl(bt, point_value=args.point_value)
            pnl_pairs.append((rt["pnl"], bti_pnl))
        pnl_df = pd.DataFrame(pnl_pairs, columns=["pnl_rnn", "pnl_bti"])
        print("\nP&L stats on matched trades:")
        print(pnl_df.describe())


if __name__ == "__main__":
    main()
