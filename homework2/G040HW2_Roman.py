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


def SequentialFFT(points, K):
    C = [points[0]]


def main():
    # SPARK SETUP
    conf = SparkConf().setAppName('G040HW2')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("OFF")

    # CHECK THE NUMBER OF CMD LINE PARAMETERS
    assert len(sys.argv) == 5, "Usage: python G04HW1.py <file_name> <M> <K> <L>"

    # PARSE FILE NAME
    path = sys.argv[1]
    assert os.path.isfile(path), "File or folder not found"

    # PARSE D
    M = sys.argv[2]
    try:
        M = float(M)
    except ValueError:
        print("D is invalid")
        raise

    # PARSE M
    K = sys.argv[3]
    assert K.isdigit(), "M is invalid"
    K = int(K)

    # PARSE K
    L = sys.argv[4]
    assert L.isdigit(), "K is invalid"
    L = int(L)

    # PRINT CONSOLE ARGUMENTS
    print(f"{path} M={M} K={K} L={L}")

    # READ INPUT FILE
    rawData = sc.textFile(path, minPartitions=L)
    inputPoints = (rawData
                   .map(string_to_coordinates)
                   .repartition(numPartitions=L).cache())

    # COUNT THE NUMBER OF POINTS
    count = inputPoints.count()

    # PRINTING THE NUMBER OF POINTS
    print(f"Number of points = {count}")

    # RUN APPROXIMATE ALGORITHM
    # start = time.time() * 1000
    # MRApproxOutliers(inputPoints, D, M, K)
    # end = time.time() * 1000

    # print(f"Running time of MRApproxOutliers = {round(end - start)} ms")


if __name__ == "__main__":
    main()
