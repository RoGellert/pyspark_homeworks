from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark import StorageLevel
import threading
import sys
import os

n = -1  # To be set via command line


def process_batch(time, batch):
    global streamLength, exactHistogram

    batch_size = batch.count()
    print(batch_size)

    if streamLength[0] >= n:
        print("boop")
        return
    streamLength[0] += batch_size

    # Extract the distinct items from the batch
    batch_items = (batch
                   .map(lambda elem: (int(elem), 1))
                   .reduceByKey(lambda value1, value2: value1 + value2)
                   .collectAsMap())
    print(batch_items.keys())

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

    # PRINT CONSOLE ARGUMENTS
    print(f"n={n} phi={phi} L={epsilon} delta={delta} portExp={portExp}")

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

    # DEFINE STREAM
    stream = ssc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevel.MEMORY_AND_DISK)
    stream.foreachRDD(lambda time, batch: process_batch(time, batch))

    print("Starting streaming engine")
    ssc.start()
    print("Waiting for shutdown condition")
    stopping_condition.wait()
    print("Stopping the streaming engine")
    ssc.stop(False, True)
    print("Streaming engine stopped")

    print(exactHistogram)

