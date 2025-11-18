import time
import itertools
import multiprocessing as mp
from datetime import datetime

from backintime.analyser.indicators.constants import OPEN, CLOSE

from strategy import run_with_params


def main():
    since = datetime.fromisoformat("2024-03-10 18:00:00-04:00")
    until = datetime.fromisoformat("2024-03-12 18:00:00-04:00")

    short_ema_periods = [ x for x in range(5, 7) ]
    long_ema_periods = [ x for x in range(21, 23) ]
    sources = [OPEN, CLOSE]

    product = list(itertools.product(*[short_ema_periods, long_ema_periods, sources]))
    start_time = time.time()

    with mp.Pool() as pool:     # os.cpu_count() of workers
        for idx, params in enumerate(product):
            print(f"Running {idx + 1}/{len(product)}...")
            short_ema_period, long_ema_period, source = params
            pool.apply_async(run_with_params,
                             (since, until, short_ema_period, 
                             long_ema_period, source))
            time.sleep(1)
        # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.join
        pool.close()    # required before join()
        pool.join()     # wait for the tasks to complete

    end_time = time.time()
    print(f"Elapsed {end_time - start_time}")

if __name__ == '__main__':
	main()