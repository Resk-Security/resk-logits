#!/bin/bash
# Script de test complet pour ReskLogits

set -e

echo "🧪 Running all tests for ReskLogits..."
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to run command and check result
run_check() {
    local name=$1
    shift
    echo -e "${YELLOW}▶ $name...${NC}"
    if "$@" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ $name passed${NC}"
        return 0
    else
        echo -e "${RED}✗ $name failed${NC}"
        "$@"  # Run again to show output
        return 1
    fi
}

# 1. Formatting check
run_check "Formatting check (black)" black --check src/ tests/ examples/

# 2. Linting
run_check "Linting (ruff)" ruff check src/ tests/ examples/

# 3. Type checking (non-blocking)
echo -e "${YELLOW}▶ Type checking (mypy)...${NC}"
mypy src/ || echo -e "${YELLOW}⚠ Type checking has warnings (non-blocking)${NC}"

# 4. Unit tests
run_check "Unit tests (pytest)" pytest tests/ -v

# 5. Build
run_check "Build package" uv build

# 6. Package check
run_check "Package validation (twine)" twine check dist/*

echo ""
echo -e "${GREEN}✅ All checks passed!${NC}"

