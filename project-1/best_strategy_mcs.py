import statistics
from concurrent.futures import ThreadPoolExecutor
from mcs.monte_carlo_twice import main_twice
from mcs.monte_carlo_split import main_split

ATTEMPT = 10000
succes_twice = []
succes_split = []


with ThreadPoolExecutor() as pool:
    results_twice = [x for x in pool.map(main_twice, range(1, ATTEMPT))]
    results_split = [x for x in pool.map(main_split, range(1, ATTEMPT))]
    print('Avg successful approach for "twice" mission:', round(statistics.mean(results_twice), 2))
    print('Avg successful approach for "split" mission:', round(statistics.mean(results_split), 2))