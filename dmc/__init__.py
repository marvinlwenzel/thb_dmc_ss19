import sys
if not (sys.version_info[0] == 3 and sys.version_info[1] >= 6):
    raise Exception("Must be using Python 3.6 or higher")

from davies_bouldin_index import davies_bouldin_index
from kmeans import KmeanClusterer, kmean_clusterize as kmeans