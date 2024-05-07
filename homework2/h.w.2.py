import numpy as np

def euclidean_distance_squared(point1, point2):
    return np.sum(np.square(point1 - point2))

def SequentialFFT(points, k):
    """Implements Sequential FFT algorithm to find k clusters in points."""
    n = len(points)
    clusters = np.zeros((k, 2))
    distances = np.full(n, np.inf)

    for i in range(k):
        cluster_idx = np.argmax(distances)
        clusters[i] = points[cluster_idx]

        for j in range(n):
            distances[j] = min(euclidean_distance_squared(clusters[i], points[j]), distances[j])

    return clusters

def test_SequentialFFT():
    inputPoints = np.array([[1, 2], [3, 4], [5, 6]])
    K = 2
    expected_output = np.array([[1, 2], [5, 6]])
    assert np.array_equal(SequentialFFT(inputPoints, K), expected_output)

if __name__ == '__main__':
    test_SequentialFFT()