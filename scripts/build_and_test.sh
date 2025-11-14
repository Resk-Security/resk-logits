#!/bin/bash
# Script de build et test d'installation

set -e

echo "🔨 Building ReskLogits package..."
echo ""

# Build
uv build

echo ""
echo "🧪 Testing installation in clean environment..."
echo ""

# Create test environment
TEST_ENV="test-install-$$"
python -m venv "$TEST_ENV"
source "$TEST_ENV/bin/activate"

# Install from wheel
echo "Installing from wheel..."
pip install dist/resklogits-*.whl --quiet

# Test imports
echo "Testing imports..."
python -c "from resklogits import ShadowBanProcessor; print('✓ ShadowBanProcessor imported')"
python -c "from resklogits import ConfigParser; print('✓ ConfigParser imported')"
python -c "from resklogits import RuleCache; print('✓ RuleCache imported')"
python -c "import resklogits; print(f'✓ Version: {resklogits.__version__}')"

# Test CLI
echo "Testing CLI..."
resklogits --help > /dev/null && echo "✓ CLI works"

# Cleanup
deactivate
rm -rf "$TEST_ENV"

echo ""
echo "✅ Build and test successful!"

