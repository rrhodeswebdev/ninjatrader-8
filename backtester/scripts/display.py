import argparse
import importlib.util
import itertools
import os
import typing as t
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from itertools import chain

import mplfinance as mpf
import pandas as pd

from backintime.analyser.analyser import Analyser, AnalyserBuffer
from backintime.analyser.indicators.base import IndicatorOptions
from backintime.analyser.indicators.constants import (CLOSE, HIGH, LOW, OPEN,
                                                      VOLUME)
from backintime.data.csv import CSVCandlesFactory, CSVCandlesSchema
from backintime.data.data_provider import (DataProvider, DataProviderError,
                                           DataProviderFactory)
from backintime.indicators import *
from backintime.timeframes import Timeframes as tf
from backintime.timeframes import estimate_close_time, estimate_open_time
from backintime.utils import _get_prefetch_count, _reserve_space


def prefetch_values(indicators, ohlcv,
                    data_provider_factory: DataProviderFactory,
                    start_date: datetime) -> AnalyserBuffer:
    """
    Prefetch market data required to compute indicators.
    Return AnalyserBuffer populated with data.
    """
    base_timeframe = data_provider_factory.timeframe
    indicator_params = list(chain.from_iterable([ x.input_timeseries for x in indicators ]))
    until = start_date
    count = _get_prefetch_count(base_timeframe, indicator_params)

    print(f"Prefetch count: {count}")
    # Get `count` OHLCV candles before the `start_date` 
    before = ohlcv[ohlcv.index < start_date].tail(count)
    if not len(before):
        raise Exception(f"Need {count} candles before {start_date} "
                        f"to compute indicators. Only found {len(before)}.")
    since = before.index[0].to_pydatetime() # the first date

    print(f"Prefetch since: {since} until {until}")
    analyser_buffer = AnalyserBuffer(since)
    _reserve_space(analyser_buffer, indicator_params)

    data = data_provider_factory.create(since, until)
    for candle in data:
        analyser_buffer.update(candle)
    return analyser_buffer


def get_tpsl_lines(orders, until):
    """
    TP/SL lines are the lines from the Date Created to the Date Updated time
    at the Trigger Price level.
    """
    tp_sl = []

    for date_created, order in orders.iterrows():
        if order["Order Type"] == 'TAKE_PROFIT' or \
                order["Order Type"] == 'STOP_LOSS' or \
                order["Order Type"] == 'TAKE_PROFIT_LIMIT' or \
                order["Order Type"] == 'STOP_LOSS_LIMIT':
            if order["Date Updated"] <= until:  # Drop if it was update after the `until` date
                tp_sl.append([
                    (date_created, order["Trigger Price"]),
                    (order["Date Updated"], order["Trigger Price"])
                ])  # TP/SL lines

    return tp_sl


def get_buys(ohlcv, orders):
    import numpy as np
    signals = []
    for date, value in ohlcv.iterrows():
        item = orders[
                    (orders["Date Updated"] == value["close_time"]) &
                    (orders["Side"] == 'BUY') &
                    (orders["Status"] == 'EXECUTED')
        ]
        if len(item):
            signals.append(item["Fill Price"][0])
        else:
            signals.append(np.nan)
    return signals


def get_sells(ohlcv, orders):
    import numpy as np
    signals = []
    for date, value in ohlcv.iterrows():
        item = orders[
                    (orders["Date Updated"] == value["close_time"]) &
                    (orders["Side"] == 'SELL') &
                    (orders["Status"] == 'EXECUTED')
        ]
        if len(item):
            signals.append(item["Fill Price"][0])
        else:
            signals.append(np.nan)
    return signals


def make_indicators_plot(indicators: pd.DataFrame) -> t.Tuple[list, int]:
    plots = []
    dmi_lines = set()  # adhoc for the DMI case
    macd_lines = set()
    panel_counter = 1
    for column in indicators.columns:
        name = column.split(' ')[0]
        if name in {'dmi_adx', 'dmi_positive_di', 'dmi_negative_di'}:
            dmi_lines.add(column)

        elif name in {'macd_macd', 'macd_signal', 'macd_hist'}:
            macd_lines.add(column)

        elif name in {'adx', 'atr', 'rsi'}:
            plots.append(mpf.make_addplot((indicators[column]), panel=panel_counter, 
                                secondary_y=True, title=name.upper()))
            panel_counter += 1

        else:
            plots.append(mpf.make_addplot((indicators[column]), panel=0, secondary_y=False))
    # Handle DMI case specially: all three lines are on the same panel at the bottom
    if dmi_lines:
        for column in dmi_lines:
            name = column.split(' ')[0]
            plots.append(mpf.make_addplot((indicators[column]), panel=panel_counter, 
                                    secondary_y=True, title=name.upper()))
        panel_counter += 1

    if macd_lines:
        macd_hist_column = ""
        for column in macd_lines:
            name = column.split(' ')[0]
            if name == 'macd_hist':
                macd_hist_column = column
            else:
                plots.append(mpf.make_addplot((indicators[column]), panel=panel_counter, 
                                        secondary_y=True, title=name.upper()))
        panel_counter += 1
        
        if macd_hist_column:
            plots.append(mpf.make_addplot((indicators[macd_hist_column]), panel=panel_counter, 
                                        secondary_y=True, title='MACD_HIST'))
            panel_counter += 1

    return plots, panel_counter


def make_panel_ratios(num_panels: int):
    return [1] if num_panels == 1 else [4] + [ 1 for x in range(num_panels - 1) ]


def plot(ohlcv: pd.DataFrame, indicators: pd.DataFrame, orders: pd.DataFrame, until: datetime) -> None:
    indicators_plot, num_panels = make_indicators_plot(indicators)
    plots = indicators_plot + [
        mpf.make_addplot(get_buys(ohlcv, orders), type='scatter', markersize=200, marker='^'),
        mpf.make_addplot(get_sells(ohlcv, orders), type='scatter', markersize=200, marker='v')
    ]

    mpf.plot(ohlcv, type='candle', style='yahoo', 
             alines=get_tpsl_lines(orders, until=ohlcv.index[-1]), 
             addplot=plots, panel_ratios=make_panel_ratios(num_panels), 
             num_panels=num_panels)


def get_since_until(ohlcv: pd.DataFrame, trades: pd.DataFrame, start_id, last_id, padding: int = 10):
    """
    Return two dates, since and until, it is used to limit 
    the amount of data to be plotted.
    """
    start_time = trades.loc[start_id]["Date Created"]
    last_time = trades.loc[last_id]["Date Created"]

    since = estimate_open_time(start_time, tf.M1, -padding)
    until = estimate_close_time(last_time, tf.M1, padding)
    # Make sure the computed date is present in OHLCV dataframe
    try:
        ohlcv[since]
    except KeyError:
        earlier = ohlcv[ohlcv.index < since]
        if not len(earlier):
            raise Exception(f"Neither {since} found in OHLCV data, "
                            f"nor the date just before that.")
    else:
        since = earlier.index[-1].to_pydatetime()

    return since, until


def get_datetime_part(filename: str) -> datetime:
    """
    Get datetime part of the filename named under the 
    <strategy_title>_<datetime>.csv convention.
    """
    datetime_format = "%Y%m%d%H%M%S"
    without_extension = filename.split('.')[0]
    datetime_part = without_extension.split('_')[-1]
    return datetime.strptime(datetime_part, datetime_format)


def get_trades_filename(strategy_prefix, path: str) -> str:
    """
    Get the file containing trades with the most recent date of backtesting run.
    """
    files = os.listdir(path)
    filter_ = filter(lambda x: x.startswith(f"{strategy_prefix}_trades"), files)
    try:
        most_recent = max(filter_, key=get_datetime_part)
    except ValueError as e:
        raise Exception(f"No trades were found under {path}") from e
    else:
        return os.path.join(path, most_recent) if most_recent else ''


def get_trades(strategy_prefix, filename=None, path='.', sep=';') -> pd.DataFrame:
    """
    Load trades file and return a pandas DataFrame.
    If no filename was specified, the file is searched in the `path` dir,
    using the bakctester's default naming convention for trades.
    If multiple files were found this way, the one with the most recent
    date of the backtesting run is used.
    """
    filename = filename or get_trades_filename(strategy_prefix, path)
    if not filename:
        raise Exception(f"Neither filename was provided nor CSV file with trades found in the {path}")
    print(f"Picked up trades {filename}")
    trades = pd.read_csv(filename, sep=sep, 
                         index_col=[0], parse_dates=True)
    trades["Date Created"] = pd.to_datetime(trades["Date Created"])
    return trades


def get_orders_filename(strategy_prefix, path: str) -> str:
    """
    Get the file containing orders with the most recent date of backtesting run.
    """
    files = os.listdir(path)
    filter_ = filter(lambda x: x.startswith(f"{strategy_prefix}_orders"), files)
    try:
        most_recent = max(filter_, key=get_datetime_part)
    except ValueError as e:
        raise Exception(f"No orders were found under {path}") from e
    else:
        return os.path.join(path, most_recent) if most_recent else ''


def get_orders(strategy_prefix, filename=None, path='.', sep=';') -> pd.DataFrame:
    """
    Load orders file and return a pandas DataFrame.
    If no filename was specified, the file is searched in the `path` dir,
    using the bakctester's default naming convention for orders.
    If multiple files were found this way, the one with the most recent
    date of the backtesting run is used.
    """
    filename = filename or get_orders_filename(strategy_prefix, path)
    if not filename:
        raise Exception(f"Neither filename was provided nor CSV file with orders found in the {path}")
    print(f"Picked up orders {filename}")
    orders = pd.read_csv(filename, sep=sep,
                         index_col=[5], parse_dates=True)
    orders["Date Updated"] = pd.to_datetime(orders["Date Updated"])
    return orders


def get_ohlcv(filename: str, sep=',') -> pd.DataFrame:
    """
    Load file with OHLCV data and return a pandas DataFrame.
    """
    ohlcv = pd.read_csv(filename, sep=sep, index_col=[0], parse_dates=True)
    print(f"Picked up OHLCV {filename}")
    return ohlcv


def get_functor(analyser, indicator_name):
    # TODO: pattern matching?
    functors = {
        'ADX': lambda x: x.adx,
        'ATR': lambda x: x.atr,
        'BBANDS': lambda x: x.bbands,
        'DMI': lambda x: x.dmi,
        'EMA': lambda x: x.ema,
        'KELTNER_CHANNEL': lambda x: x.keltner_channel,
        'MACD': lambda x: x.macd,
        'PIVOT': lambda x: x.pivot,
        'PIVOT_CLASSIC': lambda x: x.pivot_classic,
        'PIVOT_FIB': lambda x: x.pivot_fib,
        'RSI': lambda x: x.rsi,
        'SMA': lambda x: x.sma
    }

    return functors[indicator_name](analyser)


def compute_indicators(opts, ohlcv_file: str, ohlcv: pd.DataFrame, 
            since: datetime, until: datetime) -> pd.DataFrame:
    """Compute indicators on the since:until range."""
    candles = CSVCandlesFactory(ohlcv_file, 'MNQUSD', tf.M1, delimiter=',', 
                        schema=CSVCandlesSchema(open_time=0, open=1,
                                                high=2, low=3, close=4,
                                                close_time=6, volume=5))
    # We need to prefetch some data in order for indicators to have values already by the `since` time
    analyser_buffer = prefetch_values(opts, ohlcv, candles, since)
    candles = candles.create(since, until)
    analyser = Analyser(analyser_buffer)
    # TODO: iterate over the opts, title -> lower? if title is KELTNER, add its lines instead
    indicators_names = []
    for opt in opts:
        if opt.indicator_name == 'MACD':
            indicators_names.append(f"macd_macd {hash(opt)}")
            indicators_names.append(f"macd_signal {hash(opt)}")
            indicators_names.append(f"macd_hist {hash(opt)}")

        elif opt.indicator_name == 'KELTNER_CHANNEL':  # also for bbands and pivot points
            indicators_names.append(f"kc_upper_band {hash(opt)}")
            indicators_names.append(f"kc_lower_band {hash(opt)}")

        elif opt.indicator_name == 'DMI':
            indicators_names.append(f"dmi_adx {hash(opt)}")
            indicators_names.append(f"dmi_positive_di {hash(opt)}")
            indicators_names.append(f"dmi_negative_di {hash(opt)}")

        elif opt.indicator_name == 'BBANDS':
            indicators_names.append(f"bbands_upper_band {hash(opt)}")
            indicators_names.append(f"bbands_middle_band {hash(opt)}")
            indicators_names.append(f"bbands_lower_band {hash(opt)}")

        elif opt.indicator_name == 'PIVOT':
            indicators_names.append(f"pivot_trad_pivot {hash(opt)}")
            indicators_names.append(f"pivot_trad_s1 {hash(opt)}")
            indicators_names.append(f"pivot_trad_s2 {hash(opt)}")
            indicators_names.append(f"pivot_trad_s3 {hash(opt)}")
            indicators_names.append(f"pivot_trad_s4 {hash(opt)}")
            indicators_names.append(f"pivot_trad_s5 {hash(opt)}")
            indicators_names.append(f"pivot_trad_r1 {hash(opt)}")
            indicators_names.append(f"pivot_trad_r2 {hash(opt)}")
            indicators_names.append(f"pivot_trad_r3 {hash(opt)}")
            indicators_names.append(f"pivot_trad_r4 {hash(opt)}")
            indicators_names.append(f"pivot_trad_r5 {hash(opt)}")

        elif opt.indicator_name == 'PIVOT_FIB':
            indicators_names.append(f"pivot_fib_pivot {hash(opt)}")
            indicators_names.append(f"pivot_fib_s1 {hash(opt)}")
            indicators_names.append(f"pivot_fib_s2 {hash(opt)}")
            indicators_names.append(f"pivot_fib_s3 {hash(opt)}")
            indicators_names.append(f"pivot_fib_r1 {hash(opt)}")
            indicators_names.append(f"pivot_fib_r2 {hash(opt)}")
            indicators_names.append(f"pivot_fib_r3 {hash(opt)}")

        elif opt.indicator_name == 'PIVOT_CLASSIC':
            indicators_names.append(f"pivot_classic_pivot {hash(opt)}")
            indicators_names.append(f"pivot_classic_s1 {hash(opt)}")
            indicators_names.append(f"pivot_classic_s2 {hash(opt)}")
            indicators_names.append(f"pivot_classic_s3 {hash(opt)}")
            indicators_names.append(f"pivot_classic_s4 {hash(opt)}")
            indicators_names.append(f"pivot_classic_r1 {hash(opt)}")
            indicators_names.append(f"pivot_classic_r2 {hash(opt)}")
            indicators_names.append(f"pivot_classic_r3 {hash(opt)}")
            indicators_names.append(f"pivot_classic_r4 {hash(opt)}")
        else:
            indicators_names.append(f"{opt.indicator_name.lower()} {hash(opt)}")

    indicators_names = sorted(indicators_names)
    indicators = pd.DataFrame(columns=indicators_names)
    indicators_values = dict()

    for candle in candles:
        # Update buffer
        analyser_buffer.update(candle)
        # Compute indicators
        for opt in opts:
            func = get_functor(analyser, opt.indicator_name)
            params = opt.indicator_options
            result = func(**params)[-1]

            if opt.indicator_name == 'MACD':
                indicators_values[f"macd_macd {hash(opt)}"] = result.macd
                indicators_values[f"macd_signal {hash(opt)}"] = result.signal
                indicators_values[f"macd_hist {hash(opt)}"] = result.hist

            elif opt.indicator_name == 'KELTNER_CHANNEL':
                indicators_values[f"kc_upper_band {hash(opt)}"] = result.upper_band
                indicators_values[f"kc_lower_band {hash(opt)}"] = result.lower_band

            elif opt.indicator_name == 'DMI':
                indicators_values[f"dmi_adx {hash(opt)}"] = result.adx
                indicators_values[f"dmi_positive_di {hash(opt)}"] = result.positive_di
                indicators_values[f"dmi_negative_di {hash(opt)}"] = result.negative_di

            elif opt.indicator_name == 'BBANDS':
                indicators_values[f"bbands_upper_band {hash(opt)}"] = result.upper_band
                indicators_values[f"bbands_middle_band {hash(opt)}"] = result.middle_band
                indicators_values[f"bbands_lower_band {hash(opt)}"] = result.lower_band

            elif opt.indicator_name == 'PIVOT':
                indicators_values[f"pivot_trad_pivot {hash(opt)}"] = result.pivot
                indicators_values[f"pivot_trad_s1 {hash(opt)}"] = result.s1
                indicators_values[f"pivot_trad_s1 {hash(opt)}"] = result.s2
                indicators_values[f"pivot_trad_s1 {hash(opt)}"] = result.s3
                indicators_values[f"pivot_trad_s1 {hash(opt)}"] = result.s4
                indicators_values[f"pivot_trad_s1 {hash(opt)}"] = result.s5
                indicators_values[f"pivot_trad_s1 {hash(opt)}"] = result.r1
                indicators_values[f"pivot_trad_s1 {hash(opt)}"] = result.r2
                indicators_values[f"pivot_trad_s1 {hash(opt)}"] = result.r3
                indicators_values[f"pivot_trad_s1 {hash(opt)}"] = result.r4
                indicators_values[f"pivot_trad_s1 {hash(opt)}"] = result.r5

            elif opt.indicator_name == 'PIVOT_FIB':
                indicators_values[f"pivot_fib_pivot {hash(opt)}"] = result.pivot
                indicators_values[f"pivot_fib_s1 {hash(opt)}"] = result.s1
                indicators_values[f"pivot_fib_s2 {hash(opt)}"] = result.s2
                indicators_values[f"pivot_fib_s3 {hash(opt)}"] = result.s3
                indicators_values[f"pivot_fib_r1 {hash(opt)}"] = result.r1
                indicators_values[f"pivot_fib_r2 {hash(opt)}"] = result.r2
                indicators_values[f"pivot_fib_r3 {hash(opt)}"] = result.r3

            elif opt.indicator_name == 'PIVOT_CLASSIC':
                indicators_values[f"pivot_classic_pivot {hash(opt)}"] = result.pivot
                indicators_values[f"pivot_classic_s1 {hash(opt)}"] = result.s1
                indicators_values[f"pivot_classic_s2 {hash(opt)}"] = result.s2
                indicators_values[f"pivot_classic_s3 {hash(opt)}"] = result.s3
                indicators_values[f"pivot_classic_s4 {hash(opt)}"] = result.s4
                indicators_values[f"pivot_classic_r1 {hash(opt)}"] = result.r1
                indicators_values[f"pivot_classic_r2 {hash(opt)}"] = result.r2
                indicators_values[f"pivot_classic_r3 {hash(opt)}"] = result.r3
                indicators_values[f"pivot_classic_r4 {hash(opt)}"] = result.r4

            else:
                indicators_values[f"{opt.indicator_name.lower()} {hash(opt)}"] = result

        sorted_keys = sorted(indicators_values)
        indicators.loc[candle.open_time] = [ indicators_values[k] for k in sorted_keys ]
    return indicators


timeframes_map = {
    'S1': tf.S1,
    'S5': tf.S5,
    'S15': tf.S15,
    'S30': tf.S30,
    # minutes
    'M1': tf.M1,
    'M3': tf.M3,
    'M5': tf.M5,
    'M15': tf.M15,
    'M30': tf.M30,
    'M45': tf.M45,
    # hours
    'H1': tf.H1,
    'H2': tf.H2,
    'H3': tf.H3,
    'H4': tf.H4,
    # day
    'D1': tf.D1,
    # week
    'W1': tf.W1
}


ohlcv_attrs_map = {
    'OPEN': OPEN,
    'HIGH': HIGH,
    'LOW': LOW,
    'CLOSE': CLOSE,
    'VOLUME': VOLUME
}


indicator_options_map = {
    'ADX': ADX,
    'ATR': ATR,
    'BBANDS': BBANDS,
    'DMI': DMI,
    'EMA': EMA,
    'KELTNER_CHANNEL': KELTNER_CHANNEL,
    'MACD': MACD,
    'PIVOT': PIVOT,
    'PIVOT_FIB': PIVOT_FIB,
    'PIVOT_CLASSIC': PIVOT_CLASSIC,
    'RSI': RSI,
    'SMA': SMA
}


def get_indicator_prefix(some):
    return ''.join(list(itertools.takewhile(lambda x: x != '(', some)))


def get_indicators_options(opts_str: str) -> t.Set[IndicatorOptions]:
    indicators = set()
    if len(opts_str):
        opts = opts_str.split('), ')
        for opt in opts:
            # Drop commas
            opt = opt.strip(')')
            indicator_name = get_indicator_prefix(opt)
            opt = opt.removeprefix(f"{indicator_name}(")
            # Get key-value pairs
            kv_pairs = opt.split(', ')
            # Parse key value pairs to dicts
            params = dict()
            for pair in kv_pairs:
                key, value = pair.split('=')
                if key == 'timeframe':
                    value = timeframes_map[value]
                elif key == 'candle_property':
                    value = ohlcv_attrs_map[value]
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        value = float(value)
                params[key] = value
            indicator_options = indicator_options_map[indicator_name]
            indicators.add(indicator_options(**params))
    return indicators


def get_stats_filename(strategy_prefix, path: str) -> str:
    """
    Get the file containing stats with the most recent date of backtesting run.
    """
    files = os.listdir(path)
    filter_ = filter(lambda x: x.startswith(f"{strategy_prefix}_stats"), files)
    try:
        most_recent = max(filter_, key=get_datetime_part)
    except ValueError as e:
        raise Exception(f"No stats of {strategy_t} strategy were found under {path}") from e
    else:
        return os.path.join(path, most_recent) if most_recent else ''


def get_stats(strategy_prefix: str, path='.', sep=';') -> pd.DataFrame:
    filename = get_stats_filename(strategy_prefix, path)
    if not filename:
        raise Exception(f"Neither filename was provided nor CSV file with stats found in the {path}")
    print(f"Picked up stats {filename}")
    stats = pd.read_csv(filename, sep=sep, parse_dates=True)
    return stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix",
                        help="Prefix of the files (order, trades) associated with displayed strategy",
                        type=str, required=True)
    parser.add_argument("--trades", 
                        help="trade ids range to display, e.g. 0:10.",
                        type=str, required=True)
    parser.add_argument("--ohlcv", 
                        help="filename with OHLCV data. Searched in the data/ dir.",
                        type=str, default='mnq_1m_20240310_fixed.csv')
    return parser.parse_args()


def main():
    args = parse_args()
    # Parse trade ids range to display
    start_id, last_id = args.trades.split(':')
    start_id = int(start_id)
    last_id = int(last_id)
    # Parse strategy prefix
    strategy_prefix = args.prefix
    # Parse filename
    ohlcv_filename = args.ohlcv
    # Evaluate lookup dirs
    root_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(root_dir, 'data')
    results_dir = os.path.join(root_dir, 'results')

    stats = get_stats(strategy_prefix, path=results_dir)
    trades = get_trades(strategy_prefix, path=results_dir)
    orders = get_orders(strategy_prefix, path=results_dir)

    ohlcv_filename = os.path.join(data_dir, ohlcv_filename)
    ohlcv = get_ohlcv(ohlcv_filename)
    full_ohlcv = ohlcv      # unsliced OHLCV
    # Evaluate dates to display the data
    since, until = get_since_until(ohlcv, trades, start_id, last_id)
    print(f"Since: {since}, until: {until}")

    orders = orders.loc[since:until,:]  # slice orders
    ohlcv = ohlcv.loc[since:until,:]    # slice OHLCV data

    opts_str = stats['Strategy Indicators'].loc[0]
    indicators_options: set = get_indicators_options(opts_str)
    indicators: pd.DataFrame = compute_indicators(indicators_options, 
                                ohlcv_filename, full_ohlcv, since, until)

    plot(ohlcv, indicators, orders, until)


if __name__ == '__main__':
    main()
