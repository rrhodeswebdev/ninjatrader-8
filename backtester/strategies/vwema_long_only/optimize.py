import time
import itertools
from decimal import Decimal
from datetime import datetime
import multiprocessing as mp

from backintime.analyser.indicators.constants import OPEN, CLOSE

from strategy import run_with_params


def main():
    since = datetime.fromisoformat("2024-03-10 18:00:00-04:00")
    until = datetime.fromisoformat("2024-03-12 18:00:00-04:00")

    tp_ratios = []
    sl_ratios = []

    for x in range(10, 13): # 10, 11, 12
        tp_ratios.append(Decimal(x)/10)     # 0.1, 0.1, 0.12

    for x in range(4, 7):   # 4,5,6
        sl_ratios.append(Decimal(x)/10)     # 0.4, 0.5, 0.6

    short_ema_periods = [ x for x in range(5, 7) ]
    long_ema_periods = [ x for x in range(21, 23) ]
    sources = [OPEN, CLOSE]

    inputs = [ 
        short_ema_periods,
        long_ema_periods, 
        sources, tp_ratios, sl_ratios
    ]

    product = list(itertools.product(*inputs))
    start_time = time.time()

    with mp.Pool() as pool:     # os.cpu_count() of workers
        for idx, params in enumerate(product):
            print(f"Running {idx + 1}/{len(product)}...")
            short_ema_period, long_ema_period, source, tp_ratio, sl_ratio = params
            pool.apply_async(run_with_params, 
                             (since, until, short_ema_period, 
                              long_ema_period, source, tp_ratio, sl_ratio))
            time.sleep(1)
        # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.join
        pool.close()    # required before join()
        pool.join()     # wait for the tasks to complete

    end_time = time.time()
    print(f"Elapsed {end_time - start_time}")

if __name__ == '__main__':
	main()