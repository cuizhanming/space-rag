"""
Test runner for RAG chatbot debugging.
Runs all tests and provides detailed output about failures.
"""

import unittest
import sys
import os


def run_all_tests():
    """Run all test suites and provide detailed results"""
    print("=" * 60)
    print("RAG CHATBOT DEBUG TEST SUITE")
    print("=" * 60)

    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Import test modules
    try:
        from test_course_search_tool import TestCourseSearchTool
        from test_ai_generator import TestAIGenerator
        from test_rag_integration import TestRAGIntegration, TestRAGSystemRealScenarios

        print("✓ All test modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import test modules: {e}")
        return False

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add all test cases
    test_classes = [
        TestCourseSearchTool,
        TestAIGenerator,
        TestRAGIntegration,
        TestRAGSystemRealScenarios,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
        print(f"✓ Added {test_class.__name__} tests")

    print(f"\nRunning {test_suite.countTestCases()} total tests...\n")

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=True)

    result = runner.run(test_suite)

    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")

    # Detailed failure analysis
    if result.failures:
        print(f"\n{'FAILURES':-^60}")
        for test, traceback in result.failures:
            print(f"\n{test}:")
            print(traceback)

    if result.errors:
        print(f"\n{'ERRORS':-^60}")
        for test, traceback in result.errors:
            print(f"\n{test}:")
            print(traceback)

    # Success status
    if result.wasSuccessful():
        print(f"\n✓ ALL TESTS PASSED")
        return True
    else:
        print(f"\n✗ SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
