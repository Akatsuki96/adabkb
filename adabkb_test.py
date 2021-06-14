import unittest as ut

from test_cases import AdaBKBTestCase


def adabkb_test_suite():
    suite = ut.TestSuite()
    suite.addTest(AdaBKBTestCase('initialization_test_case'))
    return suite


if __name__ == "__main__":
    runner = ut.TextTestRunner()
    runner.run(adabkb_test_suite())
    