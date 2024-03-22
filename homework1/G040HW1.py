from pyspark import SparkContext, SparkConf
import sys
import os


def main():
    # CHECKING NUMBER OF CMD LINE PARAMETERS
    assert len(sys.argv) == 6, "Usage: python G04HW1.py <file_name> <D> <M> <K> <L>"

    # PARSE FILE NAME
    #path = sys.argv[1]
    #assert os.path.isfile(path), "File or folder not found"

    # PARSE D
    d_distance = sys.argv[2]
    try:
        d_distance = float(d_distance)
    except ValueError:
        print("D is invalid")
        raise

    # PARSE M
    m_points = sys.argv[3]
    assert m_points.isdigit(), "M is invalid"
    m_points = int(m_points)

    # PARSE K
    k_points = sys.argv[4]
    assert k_points.isdigit(), "K is invalid"
    k_points = int(k_points)

    # PARSE D
    l_partitions = sys.argv[5]
    assert l_partitions.isdigit(), "L is invalid"
    l_partitions = int(l_partitions)

    print(f"{d_distance}, {m_points}, {k_points}, {l_partitions}")


if __name__ == "__main__":
    main()
