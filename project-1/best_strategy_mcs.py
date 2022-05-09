# from concurrent.futures import ThreadPoolExecutor
import statistics
from mcs.monte_carlo_twice import main_twice
from mcs.monte_carlo_split import main_split

ATTEMPT = 1000
succes_twice = []
succes_split = []


for x in range(ATTEMPT):
    succes_twice.append(main_twice(1))

print('Avg successful approach for "twice" mission:', statistics.mean(succes_twice))

for x in range(ATTEMPT):
    succes_split.append(main_split())

print('Avg successful approach for "split" mission:', statistics.mean(succes_split))

# with ThreadPoolExecutor() as pool:
#     results = pool.map(main, range(1, ATTEMPT))
#     for result in results:
#         succes.append(result)
#         # succes += main()
#     # print(list(results))

# print('Average number of a successful rescue mission', statistics.mean(succes))
# # print('Average number of a successful rescue mission is:', succes / ATTEMPT)
