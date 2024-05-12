from pyspark import SparkContext, SparkConf
import numpy as np
import sys
import os
import time
import math


def points_to_cells(point, cell_side):
    i = math.floor(point[0]/cell_side)
    j = math.floor(point[1]/cell_side)
    return i, j


def cell_count(points):
    count = {}
    for point in points:
        if point not in count:
            count[point] = 1
        else:
            count[point] += 1

    return [(i, count[i]) for i in count]


def string_to_coordinates(point):
    doc_list = point.split(',')
    return [float(doc_list[0]), float(doc_list[1])]


def euclidian_distance_squared(point1, point2):
    return np.sum(np.square(point1 - point2))


def MRApproxOutliers(inputPoints, D, M):
    # STEP A
    cell_side = D/(2*math.sqrt(2))
    cells = (inputPoints
             .map(lambda x: points_to_cells(x, cell_side))
             .mapPartitions(cell_count)
             .reduceByKey(lambda x, y: x + y)
             .cache())

    # STEP B
    cells_list = cells.collect()

    # COUNTING NUMBER OF POINTS IN N_3 AND N_7 REGIONS OF CELLS
    n = len(cells_list)
    n_3_neighbour_count = np.zeros(n)
    n_7_neighbour_count = np.zeros(n)
    for i in range(n):
        n_3_neighbour_count[i] += cells_list[i][1]
        n_7_neighbour_count[i] += cells_list[i][1]
        for j in range(i+1, n):
            dist_x = abs(cells_list[j][0][0] - cells_list[i][0][0])
            dist_y = abs(cells_list[j][0][1] - cells_list[i][0][1])
            if dist_x <= 1 and dist_y <= 1:
                n_3_neighbour_count[i] += cells_list[j][1]
                n_3_neighbour_count[j] += cells_list[i][1]
            if dist_x <= 3 and dist_y <= 3:
                n_7_neighbour_count[i] += cells_list[j][1]
                n_7_neighbour_count[j] += cells_list[i][1]

    # COUNTING SURE AND UNCERTAIN POINTS
    sure_outlier_count = 0
    uncertain_point_count = 0

    for i in range(n):
        if n_3_neighbour_count[i] > M:
            continue
        if n_7_neighbour_count[i] <= M:
            sure_outlier_count += cells_list[i][1]
        elif n_3_neighbour_count[i] <= M < n_7_neighbour_count[i]:
            uncertain_point_count += cells_list[i][1]

    # PRINTING SURE OUTLIERS AND UNCERTAIN POINTS
    print(f"Number of sure outliers = {sure_outlier_count}")
    print(f"Number of uncertain points = {uncertain_point_count}")


def SequentialFFT(P, K):
    """Implements Sequential FFT algorithm to find k clusters in points."""
    P = np.array(list(P))
    n = len(P)
    C = np.zeros((K, 2))
    distances = np.full(n, np.inf)
    cluster_idx_rand = np.random.randint(n, size=1).item()
    C[0] = P[cluster_idx_rand]

    for i in range(1, K):
        for j in range(n):
            distances[j] = np.min([euclidian_distance_squared(C[i-1], P[j]), distances[j]])

        cluster_idx = np.argmax(distances)
        C[i] = P[cluster_idx]

    return C


def distance_to_cluster(point, global_centers):
    return np.min([euclidian_distance_squared(point, i) for i in global_centers])


def MRFFT(P, K):
    # ROUND 1
    start = time.time() * 1000
    partition_centers = P.mapPartitions(lambda x: SequentialFFT(x, K)).collect()
    end = time.time() * 1000
    print(f"Running time of MRFFT Round 1 = {round(end - start)} ms")

    # ROUND 2
    start = time.time() * 1000
    global_centers = SequentialFFT(partition_centers, K)
    end = time.time() * 1000
    print(f"Running time of MRFFT Round 2 = {round(end - start)} ms")

    # ROUND 3
    start = time.time() * 1000
    R = P.map(lambda x: distance_to_cluster(x, global_centers)).max()
    end = time.time() * 1000
    print(f"Running time of MRFFT Round 3 = {round(end - start)} ms")

    return np.sqrt(R)


def main():
    # SPARK SETUP
    conf = SparkConf().setAppName('G040HW2').set("spark.locality.wait", "0s")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("OFF")

    # CHECK THE NUMBER OF CMD LINE PARAMETERS
    assert len(sys.argv) == 5, "Usage: python G04HW1.py <file_name> <M> <K> <L>"

    # PARSE FILE NAME
    path = sys.argv[1]
    # assert os.path.isfile(path), "File or folder not found"

    # PARSE M
    M = sys.argv[2]
    assert M.isdigit(), "M is invalid"
    M = int(M)

    # PARSE K
    K = sys.argv[3]
    assert K.isdigit(), "K is invalid"
    K = int(K)

    # PARSE L
    L = sys.argv[4]
    assert L.isdigit(), "L is invalid"
    L = int(L)

    # PRINT CONSOLE ARGUMENTS
    print(f"{path} M={M} K={K} L={L}")

    # READ INPUT FILE
    rawData = sc.textFile(path, minPartitions=L)

    # P
    inputPoints = (rawData
                   .map(string_to_coordinates)
                   .repartition(numPartitions=L).cache())

    # COUNT THE NUMBER OF POINTS
    count = inputPoints.count()

    # PRINTING THE NUMBER OF POINTS
    print(f"Number of points = {count}")

    # GETTING DISTANCE SQUARED
    R = MRFFT(inputPoints, K)

    print(f"Radius = {R:.8f}")

    # RUN APPROXIMATE ALGORITHM
    start = time.time() * 1000
    MRApproxOutliers(inputPoints, R, M)
    end = time.time() * 1000

    print(f"Running time of MRApproxOutliers = {round(end - start)} ms")


if __name__ == "__main__":
    main()
