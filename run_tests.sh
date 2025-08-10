#!/bin/bash

# Test runner script for the RAG chatbot system

set -e

echo "🧪 Running RAG Chatbot Test Suite"
echo "================================="

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Install test dependencies if not already installed
echo "📥 Installing test dependencies..."
pip install -q pytest pytest-cov pytest-mock httpx

# Run tests with coverage
echo "🏃 Running tests with coverage report..."
pytest \
    --cov=backend \
    --cov-report=html:htmlcov \
    --cov-report=term-missing \
    --cov-report=xml \
    -v \
    --tb=short

echo ""
echo "✅ Tests completed!"
echo "📊 Coverage report generated in: htmlcov/index.html"
echo "📄 XML coverage report: coverage.xml"