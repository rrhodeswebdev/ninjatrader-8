import os
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from backintime.data.csv import CSVCandlesFactory
from backintime.analyser.analyser import Analyser, AnalyserBuffer
from backintime.analyser.indicators.macd import macd_params
from backintime.analyser.indicators.sma import sma_params
from backintime.analyser.indicators.ema import ema_params
from backintime.analyser.indicators.atr import atr_params
from backintime.analyser.indicators.rsi import rsi_params
from backintime.analyser.indicators.bbands import bbands_params
from backintime.analyser.indicators.dmi import dmi_params
from backintime.analyser.indicators.adx import adx_params
from backintime.analyser.indicators.pivot import pivot_params
from backintime.analyser.indicators.keltner_channel import keltner_channel_params
from backintime.analyser.indicators.constants import OPEN, HIGH, LOW, CLOSE
from backintime.timeframes import Timeframes as tf
from backintime.timeframes import estimate_open_time


def test_macd():
    """
    Ensure that calculated macd values match expected 
    with at least 2 floating points precision,
    using valid MACD for 2022-30-11 23:59 UTC, H4 (Binance) 
    as a reference value.
    """
    params = macd_params(tf.H4)
    quantity = params[0].quantity

    dirname = os.path.dirname(__file__)
    test_file = os.path.join(dirname, "test_h4.csv")
    until = datetime.fromisoformat('2022-12-01 00:00+00:00')
    since = estimate_open_time(until, tf.H4, -quantity)
    candles = CSVCandlesFactory(test_file, 'BTCUSDT', tf.H4)
    candles = candles.create(since, until)

    analyser_buffer = AnalyserBuffer(since)
    analyser_buffer.reserve(tf.H4, CLOSE, quantity)
    analyser = Analyser(analyser_buffer)

    expected_macd = Decimal('151.30')
    expected_signal = Decimal('66.56')
    expected_hist = Decimal('84.74')
    expected_precision = Decimal('0.01')

    for candle in candles:
        analyser_buffer.update(candle)

    macd = analyser.macd(tf.H4)[-1]

    macd_diff = (Decimal(macd.macd) - expected_macd).copy_abs()
    macd_diff = macd_diff.quantize(expected_precision, ROUND_HALF_UP)

    signal_diff = (Decimal(macd.signal) - expected_signal).copy_abs()
    signal_diff = signal_diff.quantize(expected_precision, ROUND_HALF_UP)

    hist_diff = (Decimal(macd.hist) - expected_hist).copy_abs()
    hist_diff = hist_diff.quantize(expected_precision, ROUND_HALF_UP)

    assert macd_diff <= expected_precision
    assert signal_diff <= expected_precision
    assert hist_diff <= expected_precision


def test_macd_len():
    """
    Ensure that MACD calculation results in a sequence 
    with expected length.
    """
    params = macd_params(tf.H4)
    quantity = params[0].quantity
    expected_len = quantity

    dirname = os.path.dirname(__file__)
    test_file = os.path.join(dirname, "test_h4.csv")
    until = datetime.fromisoformat('2022-12-01 00:00+00:00')
    since = estimate_open_time(until, tf.H4, -quantity)
    candles = CSVCandlesFactory(test_file, 'BTCUSDT', tf.H4)
    candles = candles.create(since, until)

    analyser_buffer = AnalyserBuffer(since)
    analyser_buffer.reserve(tf.H4, CLOSE, quantity)
    analyser = Analyser(analyser_buffer)

    for candle in candles:
        analyser_buffer.update(candle)

    macd = analyser.macd(tf.H4)
    assert len(macd) == expected_len


def test_sma():
    """
    Ensure that calculated SMA with period of 9 
    matches expected with at least 2 floating points precision,
    using valid SMA for 2022-30-11 23:59 UTC, H4 (Binance) 
    as a reference value.
    """
    params = sma_params(tf.H4, CLOSE, 9)
    quantity = params[0].quantity

    dirname = os.path.dirname(__file__)
    test_file = os.path.join(dirname, "test_h4.csv")
    until = datetime.fromisoformat('2022-12-01 00:00+00:00')
    since = estimate_open_time(until, tf.H4, -quantity)
    candles = CSVCandlesFactory(test_file, 'BTCUSDT', tf.H4)
    candles = candles.create(since, until)

    analyser_buffer = AnalyserBuffer(since)
    analyser_buffer.reserve(tf.H4, CLOSE, quantity)
    analyser = Analyser(analyser_buffer)

    expected_sma = Decimal('16773.72')
    expected_precision = Decimal('0.01')

    for candle in candles:
        analyser_buffer.update(candle)

    sma = analyser.sma(tf.H4, period=9)[-1]
    sma_diff = (Decimal(sma) - expected_sma).copy_abs()
    sma_diff = sma_diff.quantize(expected_precision, ROUND_HALF_UP)

    assert sma_diff <= expected_precision


def test_sma_len():
    """
    Ensure that SMA calculation results in a sequence 
    with expected length.
    """
    params = sma_params(tf.H4, CLOSE, 9)
    quantity = params[0].quantity
    expected_len = quantity

    dirname = os.path.dirname(__file__)
    test_file = os.path.join(dirname, "test_h4.csv")
    until = datetime.fromisoformat('2022-12-01 00:00+00:00')
    since = estimate_open_time(until, tf.H4, -quantity)
    candles = CSVCandlesFactory(test_file, 'BTCUSDT', tf.H4)
    candles = candles.create(since, until)

    analyser_buffer = AnalyserBuffer(since)
    analyser_buffer.reserve(tf.H4, CLOSE, quantity)
    analyser = Analyser(analyser_buffer)

    for candle in candles:
        analyser_buffer.update(candle)

    sma = analyser.sma(tf.H4, period=9)
    assert len(sma) == expected_len


def test_ema_9():
    """
    Ensure that calculated EMA with period of 9 
    matches expected with at least 2 floating points precision,
    using valid EMA for 2022-30-11 23:59 UTC, H4 (Binance) 
    as a reference value.
    """
    params = ema_params(tf.H4, CLOSE, 9)
    quantity = params[0].quantity

    dirname = os.path.dirname(__file__)
    test_file = os.path.join(dirname, "test_h4.csv")
    until = datetime.fromisoformat('2022-12-01 00:00+00:00')
    since = estimate_open_time(until, tf.H4, -quantity)
    candles = CSVCandlesFactory(test_file, 'BTCUSDT', tf.H4)
    candles = candles.create(since, until)

    analyser_buffer = AnalyserBuffer(since)
    analyser_buffer.reserve(tf.H4, CLOSE, quantity)
    analyser = Analyser(analyser_buffer)

    expected_ema = Decimal('16833.64')
    expected_precision = Decimal('0.01')

    for candle in candles:
        analyser_buffer.update(candle)

    ema = analyser.ema(tf.H4, period=9)[-1]
    ema_diff = (Decimal(ema) - expected_ema).copy_abs()
    ema_diff = ema_diff.quantize(expected_precision, ROUND_HALF_UP)

    assert ema_diff <= expected_precision


def test_ema_9_len():
    """
    Ensure that EMA calculation with period of 9 
    results in a sequence with expected length.
    """
    params = ema_params(tf.H4, CLOSE, 9)
    quantity = params[0].quantity
    expected_len = quantity

    dirname = os.path.dirname(__file__)
    test_file = os.path.join(dirname, "test_h4.csv")
    until = datetime.fromisoformat('2022-12-01 00:00+00:00')
    since = estimate_open_time(until, tf.H4, -quantity)
    candles = CSVCandlesFactory(test_file, 'BTCUSDT', tf.H4)
    candles = candles.create(since, until)

    analyser_buffer = AnalyserBuffer(since)
    analyser_buffer.reserve(tf.H4, CLOSE, quantity)
    analyser = Analyser(analyser_buffer)

    for candle in candles:
        analyser_buffer.update(candle)

    ema = analyser.ema(tf.H4, period=9)
    assert len(ema) == expected_len


def test_ema_100():
    """
    Ensure that calculated EMA with period of 100 
    matches expected with at least 2 floating points precision,
    using valid EMA for 2022-30-11 23:59 UTC, H4 (Binance) 
    as a reference value.
    """
    params = ema_params(tf.H4, CLOSE, 100)
    quantity = params[0].quantity

    dirname = os.path.dirname(__file__)
    test_file = os.path.join(dirname, "test_h4.csv")
    until = datetime.fromisoformat('2022-12-01 00:00+00:00')
    since = estimate_open_time(until, tf.H4, -quantity)
    candles = CSVCandlesFactory(test_file, 'BTCUSDT', tf.H4)
    candles = candles.create(since, until)

    analyser_buffer = AnalyserBuffer(since)
    analyser_buffer.reserve(tf.H4, CLOSE, quantity)
    analyser = Analyser(analyser_buffer)

    expected_ema = Decimal('16814.00')
    expected_precision = Decimal('0.01')

    for candle in candles:
        analyser_buffer.update(candle)

    ema = analyser.ema(tf.H4, period=100)[-1]
    ema_diff = (Decimal(ema) - expected_ema).copy_abs()
    ema_diff = ema_diff.quantize(expected_precision, ROUND_HALF_UP)

    assert ema_diff <= expected_precision


def test_atr():
    """
    Ensure that calculated ATR with period of 14 
    matches expected with at least 2 floating points precision,
    using valid ATR for 2022-30-11 23:59 UTC, H4 (Binance) 
    as a reference value.
    """
    params = atr_params(tf.H4)
    quantity = params[0].quantity

    dirname = os.path.dirname(__file__)
    test_file = os.path.join(dirname, "test_h4.csv")
    until = datetime.fromisoformat('2022-12-01 00:00+00:00')
    since = estimate_open_time(until, tf.H4, -quantity)
    candles = CSVCandlesFactory(test_file, 'BTCUSDT', tf.H4)
    candles = candles.create(since, until)

    analyser_buffer = AnalyserBuffer(since)
    analyser_buffer.reserve(tf.H4, HIGH, quantity)
    analyser_buffer.reserve(tf.H4, LOW, quantity)
    analyser_buffer.reserve(tf.H4, CLOSE, quantity)
    analyser = Analyser(analyser_buffer)

    expected_atr = Decimal('211.47')
    expected_precision = Decimal('0.01')

    for candle in candles:
        analyser_buffer.update(candle)

    atr = analyser.atr(tf.H4)[-1]
    atr_diff = (Decimal(atr) - expected_atr).copy_abs()
    atr_diff = atr_diff.quantize(expected_precision, ROUND_HALF_UP)

    assert atr_diff <= expected_precision


def test_atr_len():
    """
    Ensure that ATR calculation results in a sequence 
    with expected length.
    """
    params = atr_params(tf.H4)
    quantity = params[0].quantity
    expected_len = quantity

    dirname = os.path.dirname(__file__)
    test_file = os.path.join(dirname, "test_h4.csv")
    until = datetime.fromisoformat('2022-12-01 00:00+00:00')
    since = estimate_open_time(until, tf.H4, -quantity)
    candles = CSVCandlesFactory(test_file, 'BTCUSDT', tf.H4)
    candles = candles.create(since, until)

    analyser_buffer = AnalyserBuffer(since)
    analyser_buffer.reserve(tf.H4, HIGH, quantity)
    analyser_buffer.reserve(tf.H4, LOW, quantity)
    analyser_buffer.reserve(tf.H4, CLOSE, quantity)
    analyser = Analyser(analyser_buffer)

    for candle in candles:
        analyser_buffer.update(candle)

    atr = analyser.atr(tf.H4)
    assert len(atr) == expected_len


def test_rsi():
    """
    Ensure that calculated RSI with period of 14 
    matches expected with at least 2 floating points precision,
    using valid RSI for 2022-30-11 23:59 UTC, H4 (Binance) 
    as a reference value.
    """
    params = rsi_params(tf.H4)
    quantity = params[0].quantity

    dirname = os.path.dirname(__file__)
    test_file = os.path.join(dirname, "test_h4.csv")
    until = datetime.fromisoformat('2022-12-01 00:00+00:00')
    since = estimate_open_time(until, tf.H4, -quantity)
    candles = CSVCandlesFactory(test_file, 'BTCUSDT', tf.H4)
    candles = candles.create(since, until)

    analyser_buffer = AnalyserBuffer(since)
    analyser_buffer.reserve(tf.H4, CLOSE, quantity)
    analyser = Analyser(analyser_buffer)

    expected_rsi = Decimal('75.35')
    expected_precision = Decimal('0.01')

    for candle in candles:
        analyser_buffer.update(candle)

    rsi = analyser.rsi(tf.H4)[-1]
    rsi_diff = (Decimal(rsi) - expected_rsi).copy_abs()
    rsi_diff = rsi_diff.quantize(expected_precision, ROUND_HALF_UP)

    assert rsi_diff <= expected_precision


def test_rsi_len():
    """
    Ensure that RSI calculation results in a sequence 
    with expected length.
    """
    params = rsi_params(tf.H4)
    quantity = params[0].quantity
    expected_len = quantity

    dirname = os.path.dirname(__file__)
    test_file = os.path.join(dirname, "test_h4.csv")
    until = datetime.fromisoformat('2022-12-01 00:00+00:00')
    since = estimate_open_time(until, tf.H4, -quantity)
    candles = CSVCandlesFactory(test_file, 'BTCUSDT', tf.H4)
    candles = candles.create(since, until)

    analyser_buffer = AnalyserBuffer(since)
    analyser_buffer.reserve(tf.H4, CLOSE, quantity)
    analyser = Analyser(analyser_buffer)

    for candle in candles:
        analyser_buffer.update(candle)

    rsi = analyser.rsi(tf.H4)
    assert len(rsi) == expected_len


def test_bbands():
    """
    Ensure that calculated BBANDS values match expected 
    with at least 2 floating points precision,
    using valid BBANDS for 2022-30-11 23:59 UTC, H4 (Binance) 
    as a reference value.
    """
    params = bbands_params(tf.H4)
    quantity = params[0].quantity

    dirname = os.path.dirname(__file__)
    test_file = os.path.join(dirname, "test_h4.csv")
    until = datetime.fromisoformat('2022-12-01 00:00+00:00')
    since = estimate_open_time(until, tf.H4, -quantity)
    candles = CSVCandlesFactory(test_file, 'BTCUSDT', tf.H4)
    candles = candles.create(since, until)

    analyser_buffer = AnalyserBuffer(since)
    analyser_buffer.reserve(tf.H4, CLOSE, quantity)
    analyser = Analyser(analyser_buffer)

    expected_middle_band = Decimal('16519.69')
    expected_upper_band = Decimal('17138.52')
    expected_lower_band = Decimal('15900.85')
    expected_precision = Decimal('0.01')

    for candle in candles:
        analyser_buffer.update(candle)

    bbands = analyser.bbands(tf.H4)[-1]

    upper_band = bbands.upper_band
    upper_band_diff = (Decimal(upper_band) - expected_upper_band).copy_abs()
    upper_band_diff = upper_band_diff.quantize(expected_precision, ROUND_HALF_UP)

    mid_band = bbands.middle_band
    mid_band_diff = (Decimal(mid_band) - expected_middle_band).copy_abs()
    mid_band_diff = mid_band_diff.quantize(expected_precision, ROUND_HALF_UP)

    lower_band = bbands.lower_band
    lower_band_diff = (Decimal(lower_band) - expected_lower_band).copy_abs()
    lower_band_diff = lower_band_diff.quantize(expected_precision, ROUND_HALF_UP)

    assert upper_band_diff <= expected_precision
    assert mid_band_diff <= expected_precision
    assert lower_band_diff <= expected_precision


def test_bbands_len():
    """
    Ensure that BBANDS calculation results in a sequence 
    with expected length.
    """
    params = bbands_params(tf.H4)
    quantity = params[0].quantity
    expected_len = quantity

    dirname = os.path.dirname(__file__)
    test_file = os.path.join(dirname, "test_h4.csv")
    until = datetime.fromisoformat('2022-12-01 00:00+00:00')
    since = estimate_open_time(until, tf.H4, -quantity)
    candles = CSVCandlesFactory(test_file, 'BTCUSDT', tf.H4)
    candles = candles.create(since, until)

    analyser_buffer = AnalyserBuffer(since)
    analyser_buffer.reserve(tf.H4, CLOSE, quantity)
    analyser = Analyser(analyser_buffer)

    for candle in candles:
        analyser_buffer.update(candle)

    bbands = analyser.bbands(tf.H4)
    assert len(bbands) == expected_len


def test_dmi():
    """
    Ensure that calculated DMI values match expected 
    with at least 2 floating points precision,
    using valid DMI for 2022-30-11 23:59 UTC, H4 (Binance) 
    as a reference value.
    """
    params = dmi_params(tf.H4)
    quantity = params[0].quantity

    dirname = os.path.dirname(__file__)
    test_file = os.path.join(dirname, "test_h4.csv")
    until = datetime.fromisoformat('2022-12-01 00:00+00:00')
    since = estimate_open_time(until, tf.H4, -quantity)
    candles = CSVCandlesFactory(test_file, 'BTCUSDT', tf.H4)
    candles = candles.create(since, until)

    analyser_buffer = AnalyserBuffer(since)
    analyser_buffer.reserve(tf.H4, HIGH, quantity)
    analyser_buffer.reserve(tf.H4, LOW, quantity)
    analyser_buffer.reserve(tf.H4, CLOSE, quantity)
    analyser = Analyser(analyser_buffer)

    expected_adx = Decimal('27.1603')
    expected_positive_di = Decimal('34.2968')
    expected_negative_di = Decimal('14.7384')
    expected_precision = Decimal('0.01')

    for candle in candles:
        analyser_buffer.update(candle)

    dmi = analyser.dmi(tf.H4)[-1]

    positive_di = dmi.positive_di
    pos_di_diff = (Decimal(positive_di) - expected_positive_di).copy_abs()
    pos_di_diff = pos_di_diff.quantize(expected_precision, ROUND_HALF_UP)

    negative_di = dmi.negative_di
    neg_di_diff = (Decimal(negative_di) - expected_negative_di).copy_abs()
    neg_di_diff = neg_di_diff.quantize(expected_precision, ROUND_HALF_UP)

    adx = dmi.adx
    adx_diff = (Decimal(adx) - expected_adx).copy_abs()
    adx_diff = adx_diff.quantize(expected_precision, ROUND_HALF_UP)

    assert adx_diff <= expected_precision
    assert pos_di_diff <= expected_precision
    assert neg_di_diff <= expected_precision


def test_dmi_len():
    """
    Ensure that DMI calculation results in a sequence 
    with expected length.
    """
    params = dmi_params(tf.H4)
    quantity = params[0].quantity
    expected_len = quantity

    dirname = os.path.dirname(__file__)
    test_file = os.path.join(dirname, "test_h4.csv")
    until = datetime.fromisoformat('2022-12-01 00:00+00:00')
    since = estimate_open_time(until, tf.H4, -quantity)
    candles = CSVCandlesFactory(test_file, 'BTCUSDT', tf.H4)
    candles = candles.create(since, until)

    analyser_buffer = AnalyserBuffer(since)
    analyser_buffer.reserve(tf.H4, HIGH, quantity)
    analyser_buffer.reserve(tf.H4, LOW, quantity)
    analyser_buffer.reserve(tf.H4, CLOSE, quantity)
    analyser = Analyser(analyser_buffer)

    for candle in candles:
        analyser_buffer.update(candle)

    dmi = analyser.dmi(tf.H4)
    assert len(dmi) == expected_len


def test_adx():
    """
    Ensure that calculated ADX value match expected 
    with at least 2 floating points precision,
    using valid ADX for 2022-30-11 23:59 UTC, H4 (Binance) 
    as a reference value.
    """
    params = adx_params(tf.H4)
    quantity = params[0].quantity

    dirname = os.path.dirname(__file__)
    test_file = os.path.join(dirname, "test_h4.csv")
    until = datetime.fromisoformat('2022-12-01 00:00+00:00')
    since = estimate_open_time(until, tf.H4, -quantity)
    candles = CSVCandlesFactory(test_file, 'BTCUSDT', tf.H4)
    candles = candles.create(since, until)

    analyser_buffer = AnalyserBuffer(since)
    analyser_buffer.reserve(tf.H4, HIGH, quantity)
    analyser_buffer.reserve(tf.H4, LOW, quantity)
    analyser_buffer.reserve(tf.H4, CLOSE, quantity)
    analyser = Analyser(analyser_buffer)

    expected_adx = Decimal('27.1603')
    expected_precision = Decimal('0.01')

    for candle in candles:
        analyser_buffer.update(candle)

    adx = analyser.adx(tf.H4)[-1]
    adx_diff = (Decimal(adx) - expected_adx).copy_abs()
    adx_diff = adx_diff.quantize(expected_precision, ROUND_HALF_UP)

    assert adx_diff <= expected_precision


def test_adx_len():
    """
    Ensure that ADX calculation results in a sequence 
    with expected length.
    """
    params = adx_params(tf.H4)
    quantity = params[0].quantity
    expected_len = quantity

    dirname = os.path.dirname(__file__)
    test_file = os.path.join(dirname, "test_h4.csv")
    until = datetime.fromisoformat('2022-12-01 00:00+00:00')
    since = estimate_open_time(until, tf.H4, -quantity)
    candles = CSVCandlesFactory(test_file, 'BTCUSDT', tf.H4)
    candles = candles.create(since, until)

    analyser_buffer = AnalyserBuffer(since)
    analyser_buffer.reserve(tf.H4, HIGH, quantity)
    analyser_buffer.reserve(tf.H4, LOW, quantity)
    analyser_buffer.reserve(tf.H4, CLOSE, quantity)
    analyser = Analyser(analyser_buffer)

    for candle in candles:
        analyser_buffer.update(candle)

    adx = analyser.adx(tf.H4)
    assert len(adx) == expected_len


def test_classic_pivot():
    """
    Ensure that calculated PIVOT values (classic) with daily period
    match expected with at least 2 floating points precision,
    using valid PIVOT for 2022-30-11 23:59 UTC, H4 (Binance) 
    as reference values.
    """
    params = pivot_params(tf.D1)
    quantity = params[0].quantity

    dirname = os.path.dirname(__file__)
    test_file = os.path.join(dirname, "test_h4.csv")
    until = datetime.fromisoformat('2022-12-01 00:00+00:00')
    since = estimate_open_time(until, tf.D1, -quantity)
    candles = CSVCandlesFactory(test_file, 'BTCUSDT', tf.H4)
    candles = candles.create(since, until)

    analyser_buffer = AnalyserBuffer(since)
    analyser_buffer.reserve(tf.D1, HIGH, quantity)
    analyser_buffer.reserve(tf.D1, LOW, quantity)
    analyser_buffer.reserve(tf.D1, CLOSE, quantity)
    analyser = Analyser(analyser_buffer)

    expected_pivot = Decimal('16363.75')
    expected_s1 = Decimal('16178.78')
    expected_s2 = Decimal('15915.04')
    expected_s3 = Decimal('15466.33')
    expected_s4 = Decimal('15017.62')
    expected_r1 = Decimal('16627.49')
    expected_r2 = Decimal('16812.46')
    expected_r3 = Decimal('17261.17')
    expected_r4 = Decimal('17709.88')
    expected_precision = Decimal('0.01')

    for candle in candles:
        analyser_buffer.update(candle)

    pivot = analyser.pivot_classic(tf.D1)[-1]
    pivot_diff = (Decimal(pivot.pivot) - expected_pivot).copy_abs()
    pivot_diff = pivot_diff.quantize(expected_precision, ROUND_HALF_UP)

    s1_diff = (Decimal(pivot.s1) - expected_s1).copy_abs()
    s1_diff = s1_diff.quantize(expected_precision, ROUND_HALF_UP)

    s2_diff = (Decimal(pivot.s2) - expected_s2).copy_abs()
    s2_diff = s2_diff.quantize(expected_precision, ROUND_HALF_UP)

    s3_diff = (Decimal(pivot.s3) - expected_s3).copy_abs()
    s3_diff = s3_diff.quantize(expected_precision, ROUND_HALF_UP)

    s4_diff = (Decimal(pivot.s4) - expected_s4).copy_abs()
    s4_diff = s4_diff.quantize(expected_precision, ROUND_HALF_UP)

    r1_diff = (Decimal(pivot.r1) - expected_r1).copy_abs()
    r1_diff = r1_diff.quantize(expected_precision, ROUND_HALF_UP)

    r2_diff = (Decimal(pivot.r2) - expected_r2).copy_abs()
    r2_diff = r2_diff.quantize(expected_precision, ROUND_HALF_UP)

    r3_diff = (Decimal(pivot.r3) - expected_r3).copy_abs()
    r3_diff = r3_diff.quantize(expected_precision, ROUND_HALF_UP)

    r4_diff = (Decimal(pivot.r4) - expected_r4).copy_abs()
    r4_diff = r4_diff.quantize(expected_precision, ROUND_HALF_UP)

    assert pivot_diff <= expected_precision
    assert s1_diff <= expected_precision
    assert s2_diff <= expected_precision
    assert s3_diff <= expected_precision
    assert s4_diff <= expected_precision
    assert r1_diff <= expected_precision
    assert r2_diff <= expected_precision
    assert r3_diff <= expected_precision
    assert r4_diff <= expected_precision


def test_classic_pivot_len():
    """
    Ensure that PIVOT (classic) calculation results 
    in a sequence with expected length.
    """
    params = pivot_params(tf.D1)
    quantity = params[0].quantity
    expected_len = quantity - 1

    dirname = os.path.dirname(__file__)
    test_file = os.path.join(dirname, "test_h4.csv")
    until = datetime.fromisoformat('2022-12-01 00:00+00:00')
    since = estimate_open_time(until, tf.D1, -quantity)
    candles = CSVCandlesFactory(test_file, 'BTCUSDT', tf.H4)
    candles = candles.create(since, until)

    analyser_buffer = AnalyserBuffer(since)
    analyser_buffer.reserve(tf.D1, HIGH, quantity)
    analyser_buffer.reserve(tf.D1, LOW, quantity)
    analyser_buffer.reserve(tf.D1, CLOSE, quantity)
    analyser = Analyser(analyser_buffer)

    for candle in candles:
        analyser_buffer.update(candle)

    pivot = analyser.pivot_classic(tf.D1)
    assert len(pivot) == expected_len


def test_traditional_pivot():
    """
    Ensure that calculated PIVOT values (traditional) with daily period
    match expected with at least 2 floating points precision,
    using valid PIVOT for 2022-30-11 23:59 UTC, H4 (Binance) 
    as reference values.
    """
    params = pivot_params(tf.D1)
    quantity = params[0].quantity

    dirname = os.path.dirname(__file__)
    test_file = os.path.join(dirname, "test_h4.csv")
    until = datetime.fromisoformat('2022-12-01 00:00+00:00')
    since = estimate_open_time(until, tf.D1, -quantity)
    candles = CSVCandlesFactory(test_file, 'BTCUSDT', tf.H4)
    candles = candles.create(since, until)

    analyser_buffer = AnalyserBuffer(since)
    analyser_buffer.reserve(tf.D1, HIGH, quantity)
    analyser_buffer.reserve(tf.D1, LOW, quantity)
    analyser_buffer.reserve(tf.D1, CLOSE, quantity)
    analyser = Analyser(analyser_buffer)

    expected_pivot = Decimal('16363.75')
    expected_s1 = Decimal('16178.78')
    expected_s2 = Decimal('15915.04')
    expected_s3 = Decimal('15730.07')
    expected_s4 = Decimal('15545.11')
    expected_s5 = Decimal('15360.15')
    expected_r1 = Decimal('16627.49')
    expected_r2 = Decimal('16812.46')
    expected_r3 = Decimal('17076.20')
    expected_r4 = Decimal('17339.95')
    expected_r5 = Decimal('17603.70')
    expected_precision = Decimal('0.01')

    for candle in candles:
        analyser_buffer.update(candle)

    pivot = analyser.pivot(tf.D1)[-1]
    pivot_diff = (Decimal(pivot.pivot) - expected_pivot).copy_abs()
    pivot_diff = pivot_diff.quantize(expected_precision, ROUND_HALF_UP)

    s1_diff = (Decimal(pivot.s1) - expected_s1).copy_abs()
    s1_diff = s1_diff.quantize(expected_precision, ROUND_HALF_UP)

    s2_diff = (Decimal(pivot.s2) - expected_s2).copy_abs()
    s2_diff = s2_diff.quantize(expected_precision, ROUND_HALF_UP)

    s3_diff = (Decimal(pivot.s3) - expected_s3).copy_abs()
    s3_diff = s3_diff.quantize(expected_precision, ROUND_HALF_UP)

    s4_diff = (Decimal(pivot.s4) - expected_s4).copy_abs()
    s4_diff = s4_diff.quantize(expected_precision, ROUND_HALF_UP)

    s5_diff = (Decimal(pivot.s5) - expected_s5).copy_abs()
    s5_diff = s5_diff.quantize(expected_precision, ROUND_HALF_UP)

    r1_diff = (Decimal(pivot.r1) - expected_r1).copy_abs()
    r1_diff = r1_diff.quantize(expected_precision, ROUND_HALF_UP)

    r2_diff = (Decimal(pivot.r2) - expected_r2).copy_abs()
    r2_diff = r2_diff.quantize(expected_precision, ROUND_HALF_UP)

    r3_diff = (Decimal(pivot.r3) - expected_r3).copy_abs()
    r3_diff = r3_diff.quantize(expected_precision, ROUND_HALF_UP)

    r4_diff = (Decimal(pivot.r4) - expected_r4).copy_abs()
    r4_diff = r4_diff.quantize(expected_precision, ROUND_HALF_UP)

    r5_diff = (Decimal(pivot.r5) - expected_r5).copy_abs()
    r5_diff = r5_diff.quantize(expected_precision, ROUND_HALF_UP)

    assert pivot_diff <= expected_precision
    assert s1_diff <= expected_precision
    assert s2_diff <= expected_precision
    assert s3_diff <= expected_precision
    assert s4_diff <= expected_precision
    assert s5_diff <= expected_precision
    assert r1_diff <= expected_precision
    assert r2_diff <= expected_precision
    assert r3_diff <= expected_precision
    assert r4_diff <= expected_precision
    assert r5_diff <= expected_precision


def test_traditional_pivot_len():
    """
    Ensure that PIVOT (traditional) calculation results 
    in a sequence with expected length.
    """
    params = pivot_params(tf.D1)
    quantity = params[0].quantity
    expected_len = quantity - 1

    dirname = os.path.dirname(__file__)
    test_file = os.path.join(dirname, "test_h4.csv")
    until = datetime.fromisoformat('2022-12-01 00:00+00:00')
    since = estimate_open_time(until, tf.D1, -quantity)
    candles = CSVCandlesFactory(test_file, 'BTCUSDT', tf.H4)
    candles = candles.create(since, until)

    analyser_buffer = AnalyserBuffer(since)
    analyser_buffer.reserve(tf.D1, HIGH, quantity)
    analyser_buffer.reserve(tf.D1, LOW, quantity)
    analyser_buffer.reserve(tf.D1, CLOSE, quantity)
    analyser = Analyser(analyser_buffer)

    for candle in candles:
        analyser_buffer.update(candle)

    pivot = analyser.pivot(tf.D1)
    assert len(pivot) == expected_len


def test_fibonacci_pivot():
    """
    Ensure that calculated PIVOT values (fibonacci) with daily period
    match expected with at least 2 floating points precision,
    using valid PIVOT for 2022-30-11 23:59 UTC, H4 (Binance) 
    as reference values.
    """
    params = pivot_params(tf.D1)
    quantity = params[0].quantity

    dirname = os.path.dirname(__file__)
    test_file = os.path.join(dirname, "test_h4.csv")
    until = datetime.fromisoformat('2022-12-01 00:00+00:00')
    since = estimate_open_time(until, tf.D1, -quantity)
    candles = CSVCandlesFactory(test_file, 'BTCUSDT', tf.H4)
    candles = candles.create(since, until)

    analyser_buffer = AnalyserBuffer(since)
    analyser_buffer.reserve(tf.D1, HIGH, quantity)
    analyser_buffer.reserve(tf.D1, LOW, quantity)
    analyser_buffer.reserve(tf.D1, CLOSE, quantity)
    analyser = Analyser(analyser_buffer)

    expected_pivot = Decimal('16363.75')
    expected_s1 = Decimal('16192.34')
    expected_s2 = Decimal('16086.44')
    expected_s3 = Decimal('15915.04')
    expected_r1 = Decimal('16535.15')
    expected_r2 = Decimal('16641.05')
    expected_r3 = Decimal('16812.46')
    expected_precision = Decimal('0.01')

    for candle in candles:
        analyser_buffer.update(candle)

    pivot = analyser.pivot_fib(tf.D1)[-1]
    pivot_diff = (Decimal(pivot.pivot) - expected_pivot).copy_abs()
    pivot_diff = pivot_diff.quantize(expected_precision, ROUND_HALF_UP)

    s1_diff = (Decimal(pivot.s1) - expected_s1).copy_abs()
    s1_diff = s1_diff.quantize(expected_precision, ROUND_HALF_UP)

    s2_diff = (Decimal(pivot.s2) - expected_s2).copy_abs()
    s2_diff = s2_diff.quantize(expected_precision, ROUND_HALF_UP)

    s3_diff = (Decimal(pivot.s3) - expected_s3).copy_abs()
    s3_diff = s3_diff.quantize(expected_precision, ROUND_HALF_UP)

    r1_diff = (Decimal(pivot.r1) - expected_r1).copy_abs()
    r1_diff = r1_diff.quantize(expected_precision, ROUND_HALF_UP)

    r2_diff = (Decimal(pivot.r2) - expected_r2).copy_abs()
    r2_diff = r2_diff.quantize(expected_precision, ROUND_HALF_UP)

    r3_diff = (Decimal(pivot.r3) - expected_r3).copy_abs()
    r3_diff = r3_diff.quantize(expected_precision, ROUND_HALF_UP)

    assert pivot_diff <= expected_precision
    assert s1_diff <= expected_precision
    assert s2_diff <= expected_precision
    assert s3_diff <= expected_precision
    assert r1_diff <= expected_precision
    assert r2_diff <= expected_precision
    assert r3_diff <= expected_precision


def test_fibonacci_pivot_len():
    """
    Ensure that PIVOT (fibonacci) calculation results 
    in a sequence with expected length.
    """
    params = pivot_params(tf.D1)
    quantity = params[0].quantity
    expected_len = quantity - 1

    dirname = os.path.dirname(__file__)
    test_file = os.path.join(dirname, "test_h4.csv")
    until = datetime.fromisoformat('2022-12-01 00:00+00:00')
    since = estimate_open_time(until, tf.D1, -quantity)
    candles = CSVCandlesFactory(test_file, 'BTCUSDT', tf.H4)
    candles = candles.create(since, until)

    analyser_buffer = AnalyserBuffer(since)
    analyser_buffer.reserve(tf.D1, HIGH, quantity)
    analyser_buffer.reserve(tf.D1, LOW, quantity)
    analyser_buffer.reserve(tf.D1, CLOSE, quantity)
    analyser = Analyser(analyser_buffer)

    for candle in candles:
        analyser_buffer.update(candle)

    pivot = analyser.pivot_fib(tf.D1)
    assert len(pivot) == expected_len


def test_keltner_channel():
    """
    Ensure that calculated Keltner Channel with period of 20
    matches expected with at least 2 floating points precision,
    using valid Keltner Channel (ATR, EMA formula) for 
    2022-30-11 23:59 UTC, H4 (Binance) as a reference value.
    """
    params = keltner_channel_params(tf.H4)
    quantity = params[0].quantity

    dirname = os.path.dirname(__file__)
    test_file = os.path.join(dirname, "test_h4.csv")
    until = datetime.fromisoformat('2022-12-01 00:00+00:00')
    since = estimate_open_time(until, tf.H4, -quantity)
    candles = CSVCandlesFactory(test_file, 'BTCUSDT', tf.H4)
    candles = candles.create(since, until)

    analyser_buffer = AnalyserBuffer(since)
    analyser_buffer.reserve(tf.H4, OPEN, quantity)
    analyser_buffer.reserve(tf.H4, HIGH, quantity)
    analyser_buffer.reserve(tf.H4, LOW, quantity)
    analyser_buffer.reserve(tf.H4, CLOSE, quantity)
    analyser = Analyser(analyser_buffer)
    '''
    TODO: test for the settings listed in the technical specification
        might also consider to include this in provided sources
          params validation for indicators?
          change version convention to release dates instead of numbers
          implement one strategy, backtest on available amount on data,
          benchmark time elapsed

    TradingView settings used to get the reference values:
        - Length        20
        - Multiplier    2
        - Source        close
        - Use EMA 
        - Bands Style   Average True Range
        - ATR Length    10
    '''
    expected_upper_band = Decimal('17094.44')
    expected_middle_band = Decimal('16654.21')
    expected_lower_band = Decimal('16213.98')
    expected_precision = Decimal('0.01')

    for candle in candles:
        analyser_buffer.update(candle)

    keltner_channel = analyser.keltner_channel(tf.H4)[-1]

    upper_band = keltner_channel.upper_band
    upper_band_diff = (Decimal(upper_band) - expected_upper_band).copy_abs()
    upper_band_diff = upper_band_diff.quantize(expected_precision, ROUND_HALF_UP)

    mid_band = keltner_channel.middle_band
    mid_band_diff = (Decimal(mid_band) - expected_middle_band).copy_abs()
    mid_band_diff = mid_band_diff.quantize(expected_precision, ROUND_HALF_UP)

    lower_band = keltner_channel.lower_band
    lower_band_diff = (Decimal(lower_band) - expected_lower_band).copy_abs()
    lower_band_diff = lower_band_diff.quantize(expected_precision, ROUND_HALF_UP)

    assert upper_band_diff <= expected_precision
    assert mid_band_diff <= expected_precision
    assert lower_band_diff <= expected_precision


def test_keltner_channel_len():
    """
    Ensure that computing Keltner Channel results in a sequence 
    with expected length.
    """
    params = keltner_channel_params(tf.H4)
    quantity = params[0].quantity
    expected_len = quantity

    dirname = os.path.dirname(__file__)
    test_file = os.path.join(dirname, "test_h4.csv")
    until = datetime.fromisoformat('2022-12-01 00:00+00:00')
    since = estimate_open_time(until, tf.H4, -quantity)
    candles = CSVCandlesFactory(test_file, 'BTCUSDT', tf.H4)
    candles = candles.create(since, until)

    analyser_buffer = AnalyserBuffer(since)
    analyser_buffer.reserve(tf.H4, HIGH, quantity)
    analyser_buffer.reserve(tf.H4, LOW, quantity)
    analyser_buffer.reserve(tf.H4, CLOSE, quantity)
    analyser = Analyser(analyser_buffer)

    for candle in candles:
        analyser_buffer.update(candle)

    keltner_channel = analyser.keltner_channel(tf.H4)
    assert len(keltner_channel) == expected_len
