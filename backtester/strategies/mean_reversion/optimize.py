import os
import time
import itertools
from decimal import Decimal
from datetime import datetime
import multiprocessing as mp

from backintime.analyser.indicators.constants import OPEN, CLOSE
from backintime.result import rank_by_stat

from strategy import run_with_params


def main():
    since = datetime.fromisoformat("2024-03-10 18:00:00-04:00")
    until = datetime.fromisoformat("2024-03-12 18:00:00-04:00")

    tp_ratios = []
    sl_ratios = []
    multipliers = []
    additional_info = {}
    filename = ''

    for x in range(1, 2): # 20, 21, 22
        tp_ratios.append(Decimal(x)/4)     # 2, 2.1, 2.2
    tp_ratios.append(Decimal(1)/3) 
    
    for x in [10,15,20]:   # 15, 16, 17
        sl_ratios.append(Decimal(x)/10)     # 1.5, 1.6, 1.7

    for x in range(15, 16):
        multipliers.append(x/10)

    ema_periods = [ x for x in [5,21] ]
    atr_periods = [ x for x in [5,14] ]
    sources = [OPEN, CLOSE]

    inputs = [ 
        ema_periods,
        atr_periods,
        multipliers,
        sources, tp_ratios, sl_ratios
    ]

    # Configure dirs
    dirname = os.path.dirname(os.path.abspath(__file__))
    rootdir = os.path.dirname(os.path.dirname(dirname))
    results_dir = os.path.join(rootdir, 'results')
    with open(os.path.join(results_dir, 'timestamp.txt'), 'w', newline='') as tsfile:
        tsfile.writelines(datetime.strftime(datetime.now(), "_%Y%m%d%H%M%S"))
        
    product = list(itertools.product(*inputs))
    start_time = time.time()
    lock = mp.Lock()
    processes = []
    # with mp.Pool(1) as pool:     # os.cpu_count of workers
    for idx, params in enumerate(product):
        print(f"Running {idx + 1}/{len(product)}...")
        ema_period, atr_period, multiplier, source, tp_ratio, sl_ratio = params
        optimization = True
        p = mp.Process(target=run_with_params, args=(since, until, ema_period, 
                            source, atr_period, multiplier,  
                            tp_ratio, sl_ratio, optimization, lock))
        processes.append(p)
        p.start()
        # pool.apply_async(run_with_params, 
        #                 (since, until, ema_period, 
        #                     source, atr_period, multiplier,  
        #                     tp_ratio, sl_ratio, optimization, lock))
        time.sleep(1)
        # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.join
        # pool.close()    # required before join()
        # pool.join()     # wait for the tasks to complete
    for p in processes:
        p.join()

    end_time = time.time()
    print(f"Elapsed {end_time - start_time}")
    
    additional_info = {
                    'date'              : datetime.now(),
                    'data_title'        : 'local CSV file E:\Coding\Backtester-Pro\data\mnq_1m_20240310_fixed.csv',
                    'data_timeframe'    : 'M1',
                    'data_since'        : since,
                    'data_until'        : until,
                    'symbol'            : 'MNQUSD',
                    'strategy_title'    : 'Mean Reversion'
    }
    with open(os.path.join(results_dir, 'timestamp.txt'),'r') as tsfile:
        filename = os.path.join(results_dir, 'mean_reversion_stats' + tsfile.readline() + '.csv')
    rank_by_stat(filename, 'Expectancy', 10, additional_info)


if __name__ == '__main__':
	main()