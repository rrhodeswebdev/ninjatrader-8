import logging
import typing as t
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from itertools import chain

from .analyser.analyser import Analyser, AnalyserBuffer
from .analyser.indicators.base import IndicatorParam
from .analyser.indicators.constants import CandleProperties
from .broker.base import BrokerException
from .broker.futures.broker import FuturesBroker
from .broker.futures.proxy import FuturesBrokerProxy
from .candles import Candles, CandlesBuffer
from .data.data_provider import (DataProvider, DataProviderError,
                                 DataProviderFactory)
from .declarative.declarative import DeclarativeStrategy, generic_strategy
from .result.result import BacktestingResult
from .session import FuturesSession
from .timeframes import (Timeframes, estimate_close_time, estimate_open_time,
                         get_timeframes_ratio)
from .trading_strategy import AbstractStrategyFactory, FuturesStrategy


class PrefetchOptions(Enum):
    PREFETCH_SINCE = "PREFETCH_SINCE"
    PREFETCH_UNTIL = "PREFETCH_UNTIL"
    PREFETCH_NONE = "PREFETCH_NONE"


PREFETCH_SINCE = PrefetchOptions.PREFETCH_SINCE
PREFETCH_UNTIL = PrefetchOptions.PREFETCH_UNTIL
PREFETCH_NONE = PrefetchOptions.PREFETCH_NONE


def _get_indicators_params(
        strategy_t: t.Type[FuturesStrategy]) -> t.List[IndicatorParam]:
    """Get list of all indicators params of the strategy."""
    return list(chain.from_iterable(strategy_t.indicators))


def flat_list(seq):
    return list(chain.from_iterable(seq))


def _reserve_space(analyser_buffer: AnalyserBuffer, 
                   indicator_params: t.Iterable[IndicatorParam]) -> None:
    """Reserve space in `analyser_buffer` for all `indicator_params`."""
    for param in indicator_params:
        analyser_buffer.reserve(param.timeframe, 
                                param.candle_property,
                                param.quantity)


def _get_prefetch_count(base_timeframe: Timeframes, 
                        indicator_params: t.List[IndicatorParam]) -> int:
    """
    Get the number of `base_timeframe` candles needed to 
    prefetch all data for indicators. 
    """
    max_quantity = 0
    tf_quantity: t.Dict[Timeframes, int] = {}
    # Map timeframe to max quantity for that timeframe
    for param in indicator_params:
        tf_qty = tf_quantity.get(param.timeframe, 0)
        tf_quantity[param.timeframe] = max(tf_qty, param.quantity)
    # Scale quantity with timeframes ratio and find max
    for timeframe, quantity in tf_quantity.items():
        # NOTE: there must be no remainder
        tf_ratio, _ = get_timeframes_ratio(timeframe, base_timeframe)
        max_quantity = max(max_quantity, quantity*tf_ratio)

    return max_quantity


def prefetch_values(input_timeseries: t.List[IndicatorParam],
                    ohlcv_timeframes,
                    data_provider_factory: DataProviderFactory,
                    prefetch_option: PrefetchOptions,
                    start_date: datetime) -> t.Tuple[AnalyserBuffer, datetime]:
    base_timeframe = data_provider_factory.timeframe

    if prefetch_option is PREFETCH_SINCE:
        # Prefetch values since `start_date`
        count = _get_prefetch_count(base_timeframe, input_timeseries)
        candles_buffer = CandlesBuffer(start_date, ohlcv_timeframes)
        analyser_buffer = AnalyserBuffer(start_date)

        _reserve_space(analyser_buffer, input_timeseries)
        since = start_date
        until = estimate_open_time(since, base_timeframe, count)

        logger = logging.getLogger("backintime")
        logger.info("Start prefetching...")
        logger.info(f"count: {count}")
        logger.info(f"since: {since}")
        logger.info(f"until: {until}")

        data = data_provider_factory.create(since, until)
        for candle in data:
            candles_buffer.update(candle)
            analyser_buffer.update(candle)

        logger.info("Prefetching is done")
        return analyser_buffer, candles_buffer, until

    elif prefetch_option is PREFETCH_UNTIL:
        # Prefetch values until `start_date`
        until = start_date
        count = _get_prefetch_count(base_timeframe, input_timeseries)
        since = estimate_open_time(until, base_timeframe, -count)
        candles_buffer = CandlesBuffer(start_date, ohlcv_timeframes)
        analyser_buffer = AnalyserBuffer(since)
        _reserve_space(analyser_buffer, input_timeseries)

        logger = logging.getLogger("backintime")
        logger.info("Start prefetching...")
        logger.info(f"count: {count}")
        logger.info(f"since: {since}")
        logger.info(f"until: {until}")

        data = data_provider_factory.create(since, until)
        for candle in data:
            candles_buffer.update(candle)
            analyser_buffer.update(candle)
        logger.info("Prefetching is done")
        return analyser_buffer, candles_buffer, until

    else:   # `PREFETCH_NONE` or any other
        # Don't prefetch
        candles_buffer = CandlesBuffer(start_date, ohlcv_timeframes)
        analyser_buffer = AnalyserBuffer(start_date)
        _reserve_space(analyser_buffer, input_timeseries)
        return analyser_buffer, candles_buffer, start_date


class IncompatibleTimeframe(Exception):
    def __init__(self, 
                 timeframe: Timeframes, 
                 incompatibles: t.Iterable[Timeframes], 
                 strategy_t: type,
                 strategy_title: str):
        message = (f"Input candles timeframe is {timeframe} which can\'t be "
                   f"used to represent timeframes: {incompatibles} "
                   f"(required for \"{strategy_title}\" ({strategy_t}).")
        super().__init__(message)


def validate_timeframes(strategy_t: type,
                        strategy_title: str,
                        ohlcv_timeframes: t.Set[Timeframes],
                        input_timeseries: t.List[IndicatorParam],
                        data_provider_factory: DataProviderFactory) -> None:
    """
    Check whether all timeframes required for `strategy_t` can be
    represented by candlesticks from data provider.
    """
    indicator_timeframes = { x.timeframe for x in input_timeseries}
    #
    ohlcv_timeframes = ohlcv_timeframes if isinstance(ohlcv_timeframes, set) else set(ohlcv_timeframes.keys())
    timeframes = indicator_timeframes | ohlcv_timeframes
    base_timeframe = data_provider_factory.timeframe
    # Timeframes are incompatible if there is non zero remainder
    is_incompatible = lambda tf: get_timeframes_ratio(tf, base_timeframe)[1]
    incompatibles = list(filter(is_incompatible, timeframes))
    if incompatibles:
        raise IncompatibleTimeframe(base_timeframe, 
                    incompatibles, strategy_t, strategy_title)


UNTIL = PrefetchOptions.PREFETCH_UNTIL


def run_backtest(strategy_t: t.Union[t.Type[FuturesStrategy], t.Type[DeclarativeStrategy]],
                 data_provider_factory: DataProviderFactory,
                 start_money: t.Union[int, str],
                 since: datetime, 
                 until: datetime,
                 strategy_factory: t.Optional[AbstractStrategyFactory] = None,
                 session: t.Optional[FuturesSession] = None,
                 prefetch_option: PrefetchOptions = UNTIL,
                 min_usd: Decimal = Decimal('0.01'),
                 min_contracts: Decimal = Decimal('1'),
                 per_contract_init_margin = Decimal('1699.64'),
                 additional_collateral = Decimal('0'),
                 contract_quotient: Decimal = Decimal('2'),
                 per_contract_fee: Decimal = Decimal('0.62'),
                 check_margin_call: bool = False,
                 per_contract_maintenance_margin = Decimal('0'),
                 per_contract_overnight_margin = Decimal('0')) -> BacktestingResult:
    if issubclass(strategy_t, FuturesStrategy):
        return run_imperative(
                 strategy_t=strategy_t,
                 data_provider_factory=data_provider_factory,
                 start_money=start_money,
                 since=since, 
                 until=until,
                 strategy_factory=strategy_factory,
                 session=session,
                 prefetch_option=prefetch_option,
                 min_usd=min_usd,
                 min_contracts=min_contracts,
                 per_contract_init_margin=per_contract_init_margin,
                 additional_collateral=additional_collateral,
                 contract_quotient=contract_quotient,
                 per_contract_fee=per_contract_fee,
                 check_margin_call=check_margin_call,
                 per_contract_maintenance_margin=per_contract_maintenance_margin,
                 per_contract_overnight_margin=per_contract_overnight_margin)

    elif issubclass(strategy_t, DeclarativeStrategy):
        return run_declarative(
                 strategy_t=strategy_t,
                 data_provider_factory=data_provider_factory,
                 start_money=start_money,
                 since=since, 
                 until=until,
                 session=session,
                 prefetch_option=prefetch_option,
                 min_usd=min_usd,
                 min_contracts=min_contracts,
                 per_contract_init_margin=per_contract_init_margin,
                 additional_collateral=additional_collateral,
                 contract_quotient=contract_quotient,
                 per_contract_fee=per_contract_fee,
                 check_margin_call=check_margin_call,
                 per_contract_maintenance_margin=per_contract_maintenance_margin,
                 per_contract_overnight_margin=per_contract_overnight_margin)
    else:
        raise TypeError('Unsupported strategy type')


def run_imperative(strategy_t: t.Type[FuturesStrategy],
                   data_provider_factory: DataProviderFactory,
                   start_money: t.Union[int, str],
                   since: datetime, 
                   until: datetime,
                   strategy_factory: t.Optional[AbstractStrategyFactory] = None,
                   session: t.Optional[FuturesSession] = None,
                   prefetch_option: PrefetchOptions = UNTIL,
                   min_usd: Decimal = Decimal('0.01'),
                   min_contracts: Decimal = Decimal('1'),
                   per_contract_init_margin = Decimal('1699.64'),
                   additional_collateral = Decimal('0'),
                   contract_quotient: Decimal = Decimal('2'),
                   per_contract_fee: Decimal = Decimal('0.62'),
                   check_margin_call: bool = False,
                   per_contract_maintenance_margin = Decimal('0'),
                   per_contract_overnight_margin = Decimal('0')) -> BacktestingResult:
    """Run backtesting."""
    indicators_meta = strategy_t.indicators
    input_timeseries = [ x.input_timeseries for x in indicators_meta ]
    input_timeseries = flat_list(input_timeseries)
    ohlcv_timeframes = strategy_t.candle_timeframes
    validate_timeframes(strategy_t, strategy_t.get_title(),
                        ohlcv_timeframes, input_timeseries, 
                        data_provider_factory)
    # Create shared `Broker` for `BrokerProxy`
    start_money = Decimal(start_money)
    broker = FuturesBroker(start_money, 
                 session,
                 min_usd,
                 min_contracts,
                 per_contract_init_margin,
                 additional_collateral,
                 contract_quotient,
                 per_contract_fee,
                 check_margin_call,
                 per_contract_maintenance_margin)
    broker_proxy = FuturesBrokerProxy(broker)
    # Create shared buffer for `Analyser`
    analyser_buffer, candles_buffer, since = prefetch_values(
                input_timeseries, ohlcv_timeframes,
                data_provider_factory, prefetch_option, since)
    analyser = Analyser(analyser_buffer)
    # Create shared buffer for `Candles`
    candles = Candles(candles_buffer)

    strategy = \
            strategy_factory.create(broker_proxy, analyser, candles) \
            if strategy_factory \
            else strategy_t(broker_proxy, analyser, candles)

    market_data = data_provider_factory.create(since, until)
    logger = logging.getLogger("backintime")
    logger.info("Start backtesting...")

    max_equity = Decimal('0')
    min_equity = broker.current_equity

    try:
        for candle in market_data:
            broker.update(candle)           # Review whether orders can be executed
            candles_buffer.update(candle)   # Update candles on required timeframes
            analyser_buffer.update(candle)  # Store data for indicators calculation

            if not session or session.is_open(candle.close_time):
                strategy.tick()                 # Trading strategy logic here
            min_equity = min(broker.current_equity, min_equity)
            max_equity = max(broker.current_equity, max_equity)

    except (BrokerException, DataProviderError) as e:
        # These are more or less expected, so don't raise
        name = e.__class__.__name__
        logger.error(f"{name}: {str(e)}\nStop backtesting...")

    logger.info("Backtesting is done")
    return BacktestingResult(strategy.get_title(),
                             strategy.describe_params(),
                             strategy.repr_indicators(),
                             market_data,
                             start_money,
                             broker.balance.usd_balance,
                             broker.current_equity,
                             min_equity,
                             max_equity,
                             broker.get_trades(),
                             broker.get_orders())


def run_declarative(strategy_t: t.Type[DeclarativeStrategy],
                    data_provider_factory: DataProviderFactory,
                    start_money: t.Union[int, str],
                    since: datetime, 
                    until: datetime,
                    session: t.Optional[FuturesSession] = None,
                    prefetch_option: PrefetchOptions = UNTIL,
                    min_usd: Decimal = Decimal('0.01'),
                    min_contracts: Decimal = Decimal('1'),
                    per_contract_init_margin = Decimal('1699.64'),
                    additional_collateral = Decimal('0'),
                    contract_quotient: Decimal = Decimal('2'),
                    per_contract_fee: Decimal = Decimal('0.62'),
                    check_margin_call: bool = False,
                    per_contract_maintenance_margin = Decimal('0'),
                    per_contract_overnight_margin = Decimal('0')) -> BacktestingResult:
    """Run backtesting."""
    indicators_meta = strategy_t.get_indicators_meta()
    input_timeseries = [ x.input_timeseries for x in indicators_meta ]
    input_timeseries = flat_list(input_timeseries)
    ohlcv_timeframes = strategy_t.get_candle_timeframes()
    validate_timeframes(strategy_t, strategy_t.title,
                        ohlcv_timeframes, input_timeseries, 
                        data_provider_factory)
    # Create shared `Broker` for `BrokerProxy`
    start_money = Decimal(start_money)
    broker = FuturesBroker(start_money, 
                 session,
                 min_usd,
                 min_contracts,
                 per_contract_init_margin,
                 additional_collateral,
                 contract_quotient,
                 per_contract_fee,
                 check_margin_call,
                 per_contract_maintenance_margin)
    broker_proxy = FuturesBrokerProxy(broker)
    # Create shared buffer for `Analyser`
    analyser_buffer, candles_buffer, since = prefetch_values(
                input_timeseries, ohlcv_timeframes,
                data_provider_factory, prefetch_option, since)
    analyser = Analyser(analyser_buffer)
    # Create shared buffer for `Candles`
    candles = Candles(candles_buffer)

    market_data = data_provider_factory.create(since, until)
    logger = logging.getLogger("backintime")
    logger.info("Start backtesting...")

    max_equity = Decimal('0')
    min_equity = broker.current_equity

    try:
        for candle in market_data:
            broker.update(candle)           # Review whether orders can be executed
            candles_buffer.update(candle)   # Update candles on required timeframes
            analyser_buffer.update(candle)  # Store data for indicators calculation

            if not session or session.is_open(candle.close_time):
                generic_strategy(strategy_t, broker, analyser, candles)  # Trading strategy logic here
            min_equity = min(broker.current_equity, min_equity)
            max_equity = max(broker.current_equity, max_equity)

    except (BrokerException, DataProviderError) as e:
        # These are more or less expected, so don't raise
        name = e.__class__.__name__
        logger.error(f"{name}: {str(e)}\nStop backtesting...")

    logger.info("Backtesting is done")
    return BacktestingResult(strategy_t.title,
                             strategy_t.params,
                             strategy_t.repr_indicators(),
                             market_data,
                             start_money,
                             broker.balance.usd_balance,
                             broker.current_equity,
                             min_equity,
                             max_equity,
                             broker.get_trades(),
                             broker.get_orders())
