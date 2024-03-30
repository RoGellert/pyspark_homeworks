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


def ExactOutliers(inputPoints, D, M, K):
    # GET NUMBER OF POINTS
    inputPoints = np.array(inputPoints)
    n = inputPoints.shape[0]

    # INITIALIZE AN ARRAY TO STORE THE NUMBER OF NEIGHBORING POINTS
    neighbour_count = np.ones(n)

    # COUNT THE NUMBER OF NEIGHBORING POINTS
    distance_squared = D**2
    for i in range(n):
        for j in range(i+1, n):
            if euclidian_distance_squared(inputPoints[i], inputPoints[j]) <= distance_squared:
                neighbour_count[i] += 1
                neighbour_count[j] += 1

    # ZIP POINTS AND COUNTS TO SORT
    points_zip = zip(inputPoints, neighbour_count)
    points_zip = sorted(points_zip, key=lambda x: x[1])

    # COUNT THE NUMBER OF OUTLIERS
    outlier_count = 0
    i = 0
    while i < n and points_zip[i][1] <= M:
        outlier_count += 1
        i += 1

    print(f"Number of outliers: {outlier_count}")

    # PRINT K FIRST OUTLIERS
    i = 0
    while i < outlier_count and i < K:
        print(f"Point: ({points_zip[i][0][0]},{points_zip[i][0][1]})")
        i += 1


def MRApproxOutliers(inputPoints, D, M, K):
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

    # SORTING THE CELLS AND PRINTING K CELLS WITH THE LEAST NUMBER OF POINTS
    sorted_cells = cells.map(lambda x: (x[1], x[0])).sortByKey().take(K)
    for i in sorted_cells:
        print(f"Cell: ({i[1][0]},{i[1][1]})  Size = {i[0]}")


def main():
    # SPARK SETUP
    conf = SparkConf().setAppName('G040HW1')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("OFF")

    # SET UP THE MAX NUMBER OF POINTS TO RUN BRUTE FORCE ALGORITHM ON
    max_brute_force_num = 200000

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

    # PARSE L
    L = sys.argv[5]
    assert L.isdigit(), "L is invalid"
    L = int(L)

    # PRINT CONSOLE ARGUMENTS
    print(f"{path} D={D} M={M} K={K} L={L}")

    # READ INPUT FILE
    rawData = sc.textFile(path, minPartitions=L)
    inputPoints = (rawData
                   .map(string_to_coordinates)
                   .repartition(numPartitions=L).cache())

    # COUNT THE NUMBER OF POINTS
    count = inputPoints.count()

    # PRINTING THE NUMBER OF POINTS
    print(f"Number of points = {count}")

    # RUN BRUTE FORCE ALGORITHM
    if count <= max_brute_force_num:
        listOfPoints = inputPoints.collect()

        start = time.time() * 1000
        ExactOutliers(listOfPoints, D, M, K)
        end = time.time() * 1000

        print(f"Running time of ExactOutliers = {round(end - start)} ms")

    # RUN APPROXIMATE ALGORITHM
    start = time.time() * 1000
    MRApproxOutliers(inputPoints, D, M, K)
    end = time.time() * 1000

    print(f"Running time of MRApproxOutliers = {round(end - start)} ms")


if __name__ == "__main__":
    main()
