#!/bin/bash

# Code format checking script for the RAG system

echo "🔍 Checking code format with Black..."
uv run black backend/ main.py --check --diff

if [ $? -eq 0 ]; then
    echo "✅ All files are properly formatted!"
else
    echo "❌ Some files need formatting. Run './scripts/format.sh' to fix."
    exit 1
fi