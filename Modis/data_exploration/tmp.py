import sys
import gc
from time import sleep
from pathlib import Path
from dask.bag import from_sequence
from collections import Counter
from dask.distributed import Client, LocalCluster
import dask_memusage


def calculate_top_10(file_path: Path):
    gc.collect()  # See notes below

    # Load the file
    with open(file_path) as f:
        data = f.read()

    # Count the words
    counts = Counter()
    for word in data.split():
        counts[word.strip(".,'\"").lower()] += 1

    # Choose the top 10:
    by_count = sorted(counts.items(), key=lambda x: x[1])
    sleep(0.1)  # See notes below
    return (file_path.name, by_count[-10:])


def main(directory):
    # Setup the calculation:

    # Create a 4-process cluster (running locally). Note only one thread
    # per-worker: because polling is per-process, you can't run multiple
    # threads per worker, otherwise you'll get results that combine memory
    # usage of multiple tasks.
    cluster = LocalCluster(n_workers=4, threads_per_worker=1,
                           memory_limit=None)
    # Install dask-memusage:
    dask_memusage.install(cluster.scheduler, "memusage.csv")
    client = Client(cluster)

    # Create the task graph:
    files = from_sequence(Path(directory).iterdir())
    graph = files.map(calculate_top_10)
    graph.visualize(filename="example2.png", rankdir="TD")

    # Run the calculations:
    for result in graph.compute():
        print(result)
    # ... do something with results ...


if __name__ == '__main__':
    main(sys.argv[1])