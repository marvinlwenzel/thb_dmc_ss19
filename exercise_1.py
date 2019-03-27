import sys

from davies_bouldin_index import davies_bouldin_index
from kmeans import KmeanClusterer

if sys.version_info[0] < 3 or sys.version_info[1] < 6:
    raise Exception("Must be using Python 3.6 or higher")

if __name__ == '__main__':
    POINTS = ((1, 3), (3, 3), (3, 4), (4, 2), (5, 2), (5, 8), (8, 3), (8, 7))
    INITIAL_CENTERS = (('c1', (3, 2),), ('c2', (6, 2)))

    MEANER = KmeanClusterer(POINTS)

    for clustering in MEANER:
        clusters = clustering.values()
        dbi = davies_bouldin_index(clusters)
        print("Clustering: {}\nDBI: {}".format(clustering, dbi))

    MEANER = KmeanClusterer(POINTS, INITIAL_CENTERS)

    for clustering in MEANER:
        clusters = clustering.values()
        dbi = davies_bouldin_index(clusters)
        print("Clustering: {}\nDBI: {}".format(clustering, dbi))
