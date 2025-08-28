#!/bin/bash

# Comprehensive quality check script for the RAG system

echo "🚀 Running comprehensive quality checks..."
echo ""

# Check code formatting
echo "1. Checking code format..."
./scripts/check-format.sh
format_result=$?

echo ""
echo "2. Running tests..."
cd backend && uv run python tests/run_tests.py
test_result=$?
cd ..

echo ""
echo "📊 Quality Check Summary:"
echo "========================"

if [ $format_result -eq 0 ]; then
    echo "✅ Code formatting: PASS"
else
    echo "❌ Code formatting: FAIL"
fi

if [ $test_result -eq 0 ]; then
    echo "✅ Tests: PASS"
else
    echo "❌ Tests: FAIL"
fi

# Overall result
overall_result=$((format_result + test_result))
if [ $overall_result -eq 0 ]; then
    echo ""
    echo "🎉 All quality checks passed!"
    exit 0
else
    echo ""
    echo "💥 Some quality checks failed. Please fix the issues above."
    exit 1
fi