import unittest as ut

from test_cases.adabkb_test_cases import AdaBKBTestCase, AdaBBKBTestCase, AdaBBKBWPTestCase, AdaBBKBWP_2TestCase

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


def adabbkbwp_test_suite():
    suite = ut.TestSuite()
    #suite.addTest(AdaBBKBWPTestCase('temp_test'))
    #suite.addTest(AdaBBKBWPTestCase('branin_test_case'))
    suite.addTest(AdaBBKBWPTestCase('hartmann_3_test_case'))
   # suite.addTest(AdaBBKBWPTestCase('hartmann_6_test_case'))
   # suite.addTest(AdaBBKBWPTestCase('levy_8_test_case'))
    return suite


def adabbkbwp2_test_suite():
    suite = ut.TestSuite()
    suite.addTest(AdaBBKBWP_2TestCase('temp_test'))
    #suite.addTest(AdaBBKBWP_2TestCase('temp_test_no_hmax'))
    return suite

if __name__ == "__main__":
    runner = ut.TextTestRunner()
    #runner.run(adabkb_test_suite())
    #runner.run(adabbkb_test_suite())
    runner.run(adabbkbwp_test_suite())
    runner.run(adabbkbwp2_test_suite())
    