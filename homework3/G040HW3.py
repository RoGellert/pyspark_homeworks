from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark import StorageLevel
import threading
import sys
import os

# After how many items should we stop?
THRESHOLD = -1  # To be set via command line

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

    # PARSE epsilon
    epsilon = sys.argv[3]
    try:
        epsilon = float(epsilon)
    except ValueError:
        print("epsilon is invalid")
        raise

    # PARSE delta
    delta = sys.argv[4]
    try:
        delta = float(delta)
    except ValueError:
        print("delta is invalid")
        raise

    # PARSE portExp
    portExp = sys.argv[5]
    assert portExp.isdigit(), "portExp is invalid"
    portExp = int(portExp)

    # PRINT CONSOLE ARGUMENTS
    print(f"n={n} phi={phi} L={epsilon} delta={delta} portExp={portExp}")

    # SET UP CONFIG
    conf = SparkConf().setMaster("local[*]").setAppName("G040HW3")

