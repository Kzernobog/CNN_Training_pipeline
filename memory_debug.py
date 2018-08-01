import time
import tracemalloc

snapshot = None

def trace_print():
    global snapshot
    snapshot2 = tracemalloc.take_snapshot()
    snapshot2 = snapshot2.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
        tracemalloc.Filter(False, tracemalloc.__file__)
    ))

    if snapshot is not None:
        print("===================================================== Begin Trace:")
        top_stats = snapshot2.compare_to(snapshot, 'lineno', cumulative=True)
        for stat in top_stats[:10]:
            print(stat)

    snapshot = snapshot2

l = []

def leaky_func(x):
    global l
    l.append(x)

if __name__ == "__main__":
    i =0
    tracemalloc.start()
    while True:
        leaky_func(i)
        i+=1
        time.sleep(1)
        if i % 10 == 0:
            trace_print()
