#!/bin/bash

# Code formatting script for the RAG system

echo "🎨 Running Black code formatter..."
uv run black backend/ main.py

echo "✅ Code formatting completed!"