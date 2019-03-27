"""
Module implements the Davies-Bouldin-Index as relative measure of quality for clusterings.

Based on:
D. L. Davies and D. W. Bouldin.
A cluster separation measure. IEEE Trans. on Pattern Analysis and Machine Intelligence, 1(2):224–227, 1979.
DOI: 10.1109/TPAMI.1979.4766909

"""
import sys
from typing import Collection

from vector_util import Vector, Distance_Function, euclidean_distance, simple_centroid, Centroid_Function

if sys.version_info[0] < 3 or sys.version_info[1] < 6:
    raise Exception("Must be using Python 3.6 or higher")


def db_similarity(si, sj, mij: float) -> float:
    """
    Implements the proposed similarity score of the paper of Davies and Bouldin. Definition 4

    Calculates positive scalar representing a relative measure indicating the similarity of two clusters. Big
    numbers imply similar clusters. "Big" depends on the context, it's all relative. ¯\_(ツ)_/¯

    :param si: Dispersion measure of the cluster i
    :param sj: Dispersion measure of the cluster j
    :param mij: Distance between the centroids representing the clusters i and j
    :return: A positive float. Measure of similarity.
    """
    return (si + sj) / mij


def db_dispersion(elements: Collection[Vector],
                  centroid_function: Centroid_Function,
                  distance_function: Distance_Function,
                  q: int) -> float:
    """
    Implements the proposed dispersion measure of the paper of Davies and Bouldin. Below definition 5, S_i

    Please just check the paper... p.225 left side for the formular, p.224 right for the theory behind it.

    :param elements: Collection of Vectors that are in one cluster.
    :param centroid_function: Function used to calculate the centroid of the cluster.
    :param distance_function: Function used to calculate the distance between to points.
    :param q: Parameter q of the formula in Davies' and Bouldins paper.
    :return: A positive float. The dispersion of the cluster.
    """
    centroid = centroid_function(elements)

    dist_sum = 0.0
    for element in elements:
        dist_sum += distance_function(element, centroid) ** q

    return (dist_sum / float(len(elements))) ** (1.0 / q)


def davies_bouldin_index(clusters: Collection[Collection[Vector]],
                         centroid_func: Centroid_Function = simple_centroid,
                         dispersion_distance_func: Distance_Function = euclidean_distance,
                         cluster_distance_func: Distance_Function = euclidean_distance,
                         q: int = 1) -> float:
    """
    Calculates the Davies-Bouldin-Index of a given clustering.

    See the paper for what that is.
    D. L. Davies and D. W. Bouldin.
    A cluster separation measure. IEEE Trans. on Pattern Analysis and Machine Intelligence, 1(2):224–227, 1979.
    DOI: 10.1109/TPAMI.1979.4766909

    :param clusters: The clustered points. Collection of collection of Vectors.
    :param centroid_func: Function used to calculate the centroid of the cluster.
    :param dispersion_distance_func: Function used to calculate the distance for the dispersion.
    :param cluster_distance_func: Function used to calculate the distance between two cluster centroids.
    :param q: exponent q of the dispersion function.
    :return:
    """
    cluster_list = list(clusters)
    max_similarity_sum = 0
    for i in range(0, len(cluster_list)):
        max_similarity = 0
        for j in range(0, len(cluster_list)):
            if i == j:
                continue
            a_i = simple_centroid(cluster_list[i])
            a_j = simple_centroid(cluster_list[j])
            s_i = db_dispersion(elements=cluster_list[i],
                                centroid_function=centroid_func,
                                distance_function=dispersion_distance_func,
                                q=q)
            s_j = db_dispersion(elements=cluster_list[j],
                                centroid_function=centroid_func,
                                distance_function=dispersion_distance_func,
                                q=q)
            m_ij = cluster_distance_func(a_i, a_j)

            r_ij = db_similarity(s_i, s_j, m_ij)

            max_similarity = max(max_similarity, r_ij)
        max_similarity_sum += max_similarity

    return float(max_similarity_sum) / float(len(cluster_list))
