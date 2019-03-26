from hamcrest import *

from kmeans import KmeanClusterer


def test_kmeans_final():
    ueb_1_2_c_points = ((1, 3), (3, 3), (3, 4), (4, 2), (5, 2), (5, 8), (8, 3), (8, 7))
    ueb_1_2_c_start = (('c1', (3, 2),), ('c2', (6, 2)))
    ueb_1_2_c_solution = {('c1', (2.75, 3.0)): {(1, 3), (3, 3), (5, 2), (4, 2), (3, 4)},
                          ('c2', (6.5, 5.0)): {(8, 3), (8, 7), (5, 8)}}
    clusterer = KmeanClusterer(ueb_1_2_c_points, ueb_1_2_c_start)
    result = clusterer.final_result()
    assert_that(result, equal_to(ueb_1_2_c_solution))


if __name__ == '__main__':
    test_kmeans_final()
    print("Looks good for k-means")
