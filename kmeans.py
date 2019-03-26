"""
Module implementing the k-means algorithm for simple vectors.
"""
import copy
from typing import Dict, Collection

from vector_util import Vector, NamedVector, Distance_Function, euclidean_distance, simple_centroid, Centroid_Function


def assign_points_to_clusters(points: Collection[Vector],
                              centroids: Collection[NamedVector],
                              distance_function: Distance_Function = euclidean_distance
                              ) -> Dict[NamedVector, Collection[Vector]]:
    """
    Assigns each given point to its closest cluster/centroid.

    For collections of points and (named) centroids, it returns a dictionary such that
    each given named centroid as a key that has an entry in that dictionary. The value
    is the collection of points that have no centroid closer to them than that one.

    If there are two or more centroids that have the same distance to a point, the point
    will be assigned to one and only one of them. The is NO GUARANTEE for which of them a
    point will be assigned to or that this assignment is deterministic.

    :param points: Collection of points that will be clustered / assigned to centroids
    :param centroids: Centroid used for clustering
    :param distance_function: Function used to calculate the distance between a point and
                        centroid
    :return:
    """
    result = {}
    distance = distance_function
    for point in points:
        # Starts with no cluster associated and an positive infinite distance to the clusters center
        associated_cluster = None
        min_dist = float("inf")
        for named_centroid in centroids:
            d = distance(point, named_centroid[1])
            if d < min_dist:
                min_dist = d
                associated_cluster = named_centroid
        if associated_cluster not in result:
            result[associated_cluster] = {point}
        else:
            result[associated_cluster].add(point)

    # Add Clusters to result that do not have any points associated with them
    for named_centroid in centroids:
        if named_centroid not in result:
            result[named_centroid] = {}
    return result


class KmeanClusterer:
    """
    This class encapsulates the k-means algorithm.

    It implements an iterator to allow the user to access the (non-final)
    result after each iteration of the k-means algorithm if she/he so desires.
    """

    def __copy__(self):
        return KmeanClusterer(self.points,
                              initial_centroids=self._initial_clusters,
                              distance_function=self._distance_function,
                              centroid_function=self._centroid_function)

    def __deepcopy__(self, memodict={}):
        """
        Deeply copies its points and initial clusters, shallowly copies all function pointers.
        :param memodict: Currently ignored
        :return:
        """
        return KmeanClusterer(copy.deepcopy(self.points),
                              initial_centroids=copy.deepcopy(self._initial_clusters),
                              distance_function=self._distance_function,
                              centroid_function=self._centroid_function)

    def __init__(self, points: Collection[Vector],
                 initial_centroids: Collection[NamedVector] = (("default", (0.0, 0.0)), ),
                 distance_function: Distance_Function = euclidean_distance,
                 centroid_function: Centroid_Function = simple_centroid):
        """

        :param points: Collection of points that shall be clustered
        :param initial_centroids: Collection of initial centroids, also implicitly stating the k of k-means
        :param distance_function: Function to calculate the distance between vectors.
                            Defaults to the euclidean distance.
        :param centroid_function: Function to calculate a centroid of a set of vectors.
                            Defaults to the simple centroid function.
        """
        self.points = points
        self._print_steps = False
        self._is_converged = False
        self._initial_clusters = initial_centroids
        self._current_clusters = self._initial_clusters
        self._current_clustered_points = None
        self._distance_function = distance_function
        self._centroid_function = centroid_function

    def __iter__(self):
        """

        :return: A deep copy of itself reset to its initial state.
        """
        return copy.deepcopy(self)

    def __next__(self) -> Dict[NamedVector, Collection[Vector]]:
        """
        Calculates the next step of the k-means algorithm. Stops iteration on convergence.
        :return: Dictionary of clusters. Keys are NamedVectors, Values are sets of Vectors. Empty
                        Clusters are contained as key with an empty Collection as value.
        """
        if self._is_converged:
            raise StopIteration

        old_clusters = copy.deepcopy(self._current_clusters)

        # Assign points to current cluster
        self._current_clustered_points = assign_points_to_clusters(self.points, self._current_clusters,
                                                                   distance_function=self._distance_function)
        result = copy.deepcopy(self._current_clustered_points)

        # Calc new clusters based on previous assignment
        new_clusters = set()
        for cluster in self._current_clustered_points:
            new_clusters.add((cluster[0], self._centroid_function(self._current_clustered_points[cluster])))

        # Check if newly calculated clusters are the same as the old ones
        # if so, they converged and the previously returned clustering was already the final one
        self._is_converged = True
        for cluster in new_clusters:
            if cluster not in old_clusters:
                self._is_converged = False
                break

        # After calculating the new centroid, we see that we converged.
        # However, we still want to show the previous solution, so we return the result of this iteration
        # so that the next iteration shows the clustering with the final centroids
        # We do no want the same result twice, therefor we stop the iteration.

        self._current_clusters = new_clusters
        return result

    def final_result(self) -> Dict[NamedVector, Collection[Vector]]:
        """
        Calculates the final clustering of the k-means algorithm based on its initial configuration.
        :return: Dictionary of clusters. Keys are NamedVectors, Values are sets of Vectors. Empty
                        Clusters are contained as key with an empty Collection as value.
        """
        result = None
        for r in self:
            result = r
        return result
