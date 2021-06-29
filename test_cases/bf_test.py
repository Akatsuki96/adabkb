import unittest as ut
from .adabkb_test_cases import GenericBFTestCase, BraninBFTestCase

def build_suite(tests):
    suite = ut.TestSuite()
    for test_tuple in tests:
        test_class = test_tuple[0]
        for test in test_tuple[1]:
            suite.addTest(test_class(test))
    return suite

def generic_bf_suite():
    generic_tests = [
        'test_bidimensional_bf',\
        'test_single_element_in_bounds_search_space',\
        'test_empty_search_space_dimension',\
        'test_negative_noise_std']
    generic_bf_tests = [
        (GenericBFTestCase, generic_tests)
    ]
    return build_suite(generic_bf_tests)


def bidimensional_bf_suite():
    bidim_tests = ['test_global_minima']
    bidim_bf_tests = [
        (BraninBFTestCase, bidim_tests)
    ]
    return build_suite(bidim_bf_tests)


if __name__ == '__main__':
    runner = ut.TextTestRunner()
    runner.run(generic_bf_suite())
    runner.run(bidimensional_bf_suite())
