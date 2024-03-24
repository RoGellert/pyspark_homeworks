import numpy
from pyspark import SparkContext, SparkConf
import numpy as np
import sys
import os
import time


def string_to_coordinates(doc):
    doc_list = doc.split(',')
    return [float(doc_list[0]), float(doc_list[1])]


def euclidian_distance(point1, point2):
    return np.linalg.norm(point1 - point2)


def ExactOutliers(points, d, m, k):
    # GET NUMBER OF POINTS
    points = np.array(points)
    n = points.shape[0]

    # INITIALIZE AND ARRAY TO STORE THE NUMBER OF NEIGHBORING POINTS
    neighbour_count = numpy.zeros(n)

    # COUNT THE NUMBER OF NEIGHBORING POINTS
    for i in range(n):
        for j in range(i+1, n):
            if euclidian_distance(points[i], points[j]) <= d:
                neighbour_count[i] += 1
                neighbour_count[j] += 1

    # ZIP POINTS AND COUNTS TO SORT
    points_zip = zip(points, neighbour_count)
    points_zip = sorted(points_zip, key=lambda x: x[1])
    # for i in points_zip:
    #     print(i[1])

    # COUNT THE NUMBER OF OUTLIERS
    outlier_count = 0
    i = 0
    while i < n and points_zip[i][1] <= m:
        outlier_count += 1
        i += 1
    print(f"Number of outliers: {outlier_count}")

    # PRINT K FIRST OUTLIERS
    i = 0
    while i < outlier_count and i < k:
        print(points_zip[i][0])
        i += 1


def main():
    # SPARK SETUP
    conf = SparkConf().setAppName('G040HW1')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("OFF")

    # CHECK THE NUMBER OF CMD LINE PARAMETERS
    assert len(sys.argv) == 6, "Usage: python G04HW1.py <file_name> <D> <M> <K> <L>"

    # PARSE FILE NAME
    path = sys.argv[1]
    assert os.path.isfile(path), "File or folder not found"

    # PARSE D
    D = sys.argv[2]
    try:
        D = float(D)
    except ValueError:
        print("D is invalid")
        raise

    # PARSE M
    M = sys.argv[3]
    assert M.isdigit(), "M is invalid"
    M = int(M)

    # PARSE K
    K = sys.argv[4]
    assert K.isdigit(), "K is invalid"
    K = int(K)

    # PARSE D
    L = sys.argv[5]
    assert L.isdigit(), "L is invalid"
    L = int(L)

    # READ INPUT FILE
    inputPoints = (sc.textFile(path, minPartitions=L)
                   .map(string_to_coordinates)
                   .repartition(numPartitions=L).cache())

    # COUNT THE NUMBER OF POINTS
    count = inputPoints.count()
    print(inputPoints.collect())

    # CONVERT RDD INTO LIST
    listOfPoints = inputPoints.collect()

    # RUN BRUTE FORCE ALGORITHM
    start = time.time()
    if count <= 200000:
        ExactOutliers(listOfPoints, D, M, K)
    end = time.time()
    print(f"Execution time: {end - start} seconds")


if __name__ == "__main__":
    main()
