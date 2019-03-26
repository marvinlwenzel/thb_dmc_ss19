from hamcrest import *

from davies_bouldin_index import davies_bouldin_index


def test_dbi_final():
    ueb_1_2_c_in = (((1, 3), (3, 3), (5, 2), (4, 2), (3, 4)),
                    ((8, 3), (8, 7), (5, 8)))
    ueb_1_2_c_solution = 0.7709959

    result = davies_bouldin_index(ueb_1_2_c_in)
    assert_that(result, close_to(ueb_1_2_c_solution, 0.0000001))


if __name__ == '__main__':
    test_dbi_final()
    print("Looks good for the Davies Bouldin Index")
