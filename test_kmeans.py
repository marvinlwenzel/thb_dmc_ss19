import sys

from hamcrest import *

from kmeans import KmeanClusterer

if sys.version_info[0] < 3 or sys.version_info[1] < 6:
    raise Exception("Must be using Python 3.6 or higher")


def test_kmeans_final_ueb_1_2_c():
    points = ((1, 3), (3, 3), (3, 4), (4, 2), (5, 2), (5, 8), (8, 3), (8, 7))
    start = (('c1', (3, 2),), ('c2', (6, 2)))
    solution = {('c1', (3.2, 2.8)): {(1, 3), (3, 3), (5, 2), (4, 2), (3, 4)},
                ('c2', (7.0, 6.0)): {(8, 3), (8, 7), (5, 8)}}
    clusterer = KmeanClusterer(points, start)
    result = clusterer.final_result()
    assert_that(result, equal_to(solution))


def test_kmeans_final_1():
    scalars = {(2,), (4,), (10,), (12,), (3,), (20,), (30,), (11,), (25,)}
    start = (("M1", (4,)), ("M2", (11,)))

    solution = {('M1', (7.0,)): {(2,), (3,), (4,), (10,), (11,), (12,)}, ('M2', (25.0,)): {(30,), (25,), (20,)}}
    clusterer = KmeanClusterer(scalars, start)
    result = clusterer.final_result()
    assert_that(result, equal_to(solution))


def test_no_double_solution():
    points = ((1, 3), (3, 3), (3, 4), (4, 2), (5, 2), (5, 8), (8, 3), (8, 7))
    start = (('c1', (3, 2),), ('c2', (6, 2)))

    old_sol = None

    clusterer = KmeanClusterer(points, start)

    for sol in clusterer:
        assert_that(old_sol, not_(equal_to(sol)))
        old_sol = sol


if __name__ == '__main__':
    test_no_double_solution()
    test_kmeans_final_ueb_1_2_c()
    test_kmeans_final_1()
    print("Looks good for k-means")
