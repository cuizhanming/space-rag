#!/bin/bash

# LangChain Agentic RAG Startup Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🤖 LangChain Agentic RAG System${NC}"
echo -e "${BLUE}================================${NC}"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}❌ Error: pyproject.toml not found. Please run this script from the langchain-agentic-rag directory.${NC}"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  Warning: .env file not found. Please copy .env.example to .env and configure it.${NC}"
    echo -e "${BLUE}Copying .env.example to .env...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}Please edit .env file with your API keys and configuration before starting the server.${NC}"
    echo -e "${YELLOW}Minimum required: GEMINI_API_KEY${NC}"
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${RED}❌ Error: uv is not installed. Please install it first:${NC}"
    echo -e "${BLUE}curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Environment checks passed${NC}"

# Install dependencies
echo -e "${BLUE}📦 Installing dependencies...${NC}"
uv sync

# Check if GEMINI_API_KEY is set
if grep -q "your_gemini_api_key_here" .env; then
    echo -e "${YELLOW}⚠️  Warning: GEMINI_API_KEY is not configured in .env file${NC}"
    echo -e "${YELLOW}Please set your Gemini API key before starting the server.${NC}"
    exit 1
fi

# Create necessary directories
echo -e "${BLUE}📁 Creating directories...${NC}"
mkdir -p chroma_db
mkdir -p docs
mkdir -p logs

# Load documents if docs directory has files
if [ "$(ls -A docs 2>/dev/null)" ]; then
    echo -e "${GREEN}📚 Documents found in docs/ directory - will be loaded on startup${NC}"
else
    echo -e "${YELLOW}📚 No documents found in docs/ directory${NC}"
    echo -e "${BLUE}You can add .txt, .md, .py, or .json files to the docs/ directory${NC}"
fi

echo -e "${GREEN}🚀 Starting LangChain Agentic RAG server...${NC}"
echo -e "${BLUE}Server will be available at: http://localhost:8001${NC}"
echo -e "${BLUE}API documentation: http://localhost:8001/docs${NC}"
echo -e "${BLUE}Press Ctrl+C to stop the server${NC}"
echo ""

# Start the server
cd backend
uv run uvicorn app:app --reload --host 0.0.0.0 --port 8001