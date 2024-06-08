from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark import StorageLevel
import threading
import sys
import math
import random

n = -1  # To be set via command line
streamLength = [-1]
m = -1
reservoir_t = [-1]


def process_batch(time, batch):
    global streamLength, exactHistogram
    batch_size = batch.count()

    if streamLength[0] >= n:
        return
    elif batch_size + streamLength[0] > n:
        elements_to_take = n - streamLength[0]
    else:
        elements_to_take = batch_size

    streamLength[0] += elements_to_take

    # Extract the distinct items from the batch
    batch_items = batch.map(lambda elem: int(elem)).take(elements_to_take)
    # batch_items = batch.map(lambda elem: int(elem)).collect()[0:elements_to_take]

    for i in batch_items:
        if i not in exactHistogram:
            exactHistogram[i] = 1
        else:
            exactHistogram[i] += 1

    for i in batch_items:
        reservoir_t[0] += 1
        if len(reservoir) < m:
            reservoir.append(i)
        else:
            chance = m / reservoir_t[0]
            rand = random.random()
            if rand <= chance:
                reservoir[random.randint(0, m-1)] = i

    for i in batch_items:

    if streamLength[0] >= n:
        stopping_condition.set()


if __name__ == '__main__':
    assert len(sys.argv) == 6, "Usage: python G040HW3.py <n> <phi> <epsilon> <delta> <portExp>"

    # PARSE n
    n = sys.argv[1]
    assert n.isdigit(), "K is invalid"
    n = int(n)

    # PARSE phi
    phi = sys.argv[2]
    try:
        phi = float(phi)
    except ValueError:
        print("phi is invalid")
        raise
    assert 0 < phi < 1, "phi should be a float number in range (0, 1)"

    # PARSE epsilon
    epsilon = sys.argv[3]
    try:
        epsilon = float(epsilon)
    except ValueError:
        print("epsilon is invalid")
        raise
    assert 0 < epsilon < 1, "epsilon should be a float number in range (0, 1)"

    # PARSE delta
    delta = sys.argv[4]
    try:
        delta = float(delta)
    except ValueError:
        print("delta is invalid")
        raise
    assert 0 < delta < 1, "delta should be a float number in range (0, 1)"

    # PARSE portExp
    portExp = sys.argv[5]
    assert portExp.isdigit(), "portExp is invalid"
    portExp = int(portExp)

    # SET UP CONFIG
    conf = SparkConf().setMaster("local[*]").setAppName("G040HW3")

    # SET STREAMING
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 0.01)  # Batch duration of 0.01 seconds
    ssc.sparkContext.setLogLevel("ERROR")

    # DEFINING STOPPING CONDITION
    stopping_condition = threading.Event()

    # DECLARE GLOBAL VARIABLES
    streamLength = [0]  # Stream length (an array to be passed by reference)
    exactHistogram = {}

    reservoir = []
    m = math.ceil(1/phi)
    reservoir_t = [0]

    sticky_hashmap = {}
    r = math.log(1)

    # DEFINE STREAM
    stream = ssc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevel.MEMORY_AND_DISK)
    stream.foreachRDD(lambda time, batch: process_batch(time, batch))

    # print("Starting streaming engine")
    ssc.start()
    # print("Waiting for shutdown condition")
    stopping_condition.wait()
    # print("Stopping the streaming engine")
    ssc.stop(False, False)
    # print("Streaming engine stopped")

    # PRINT CONSOLE ARGUMENTS
    print("INPUT PROPERTIES")
    print(f"n = {n} phi = {phi} epsilon = {epsilon} delta = {delta} port = {portExp}")

    print("EXACT ALGORITHM")
    exactHistogramLength = len(exactHistogram)
    exactFrequentItems = [i for i in exactHistogram.items() if i[1]/n >= phi]
    exactFrequentItems = [i[0] for i in exactFrequentItems]
    exactFrequentItems.sort()
    exactFrequentItemsSet = set(exactFrequentItems)

    print(f"Number of items in the data structure = {exactHistogramLength}")
    print(f"Number of true frequent items = {len(exactFrequentItems)}")
    print("True frequent items:")
    for i in exactFrequentItems:
        print(i)

    unique_reservoir_items = set(reservoir)
    unique_reservoir_items_list = list(unique_reservoir_items)
    unique_reservoir_items_list.sort()
    print("RESERVOIR SAMPLING")
    print(f"Size m of the sample = {m}")
    print(f"Number of estimated frequent items = {len(unique_reservoir_items)}")
    print(f"Estimated frequent items:")
    for i in unique_reservoir_items_list:
        print(f"{i} {'+' if i in exactFrequentItemsSet else '-'}")
