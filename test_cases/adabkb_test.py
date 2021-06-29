import unittest as ut

from adabkb_test_cases import AdaBKBTestCase, AdaBBKBTestCase

function_test = [
    'branin_test_case',
    'booth_test_case',
    'six_hump_camel_test_case',
    'rosenbrock_2_test_case',
    'hartmann_3_test_case',
    'ackley_3_test_case',
    'ackley_5_test_case',
    'hartmann6_test_case'
]

def adabkb_test_suite():
    suite = ut.TestSuite()

    suite.addTest(AdaBKBTestCase('initialization_test_case'))
    suite.addTest(AdaBKBTestCase('single_step_test_case'))
    for fun in function_test:
        suite.addTest(AdaBKBTestCase(fun))
    return suite


def adabbkb_test_suite():
    suite = ut.TestSuite()
    suite.addTest(AdaBBKBTestCase('temp_test'))
    suite.addTest(AdaBBKBTestCase('branin_test_case'))
    suite.addTest(AdaBBKBTestCase('six_hump_camel_test_case'))
    suite.addTest(AdaBBKBTestCase('hartmann_3_test_case'))
    suite.addTest(AdaBBKBTestCase('hartmann6_test_case'))
    return suite


if __name__ == "__main__":
    runner = ut.TextTestRunner()
    runner.run(adabkb_test_suite())
    runner.run(adabbkb_test_suite())
    