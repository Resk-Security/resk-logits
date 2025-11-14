@echo off
REM Script de test complet pour ReskLogits (Windows)

echo 🧪 Running all tests for ReskLogits...
echo.

set ERRORLEVEL=0

REM 1. Formatting check
echo ▶ Formatting check (black)...
black --check src/ tests/ examples/
if %ERRORLEVEL% NEQ 0 (
    echo ✗ Formatting check failed
    exit /b 1
)
echo ✓ Formatting check passed
echo.

REM 2. Linting
echo ▶ Linting (ruff)...
ruff check src/ tests/ examples/
if %ERRORLEVEL% NEQ 0 (
    echo ✗ Linting failed
    exit /b 1
)
echo ✓ Linting passed
echo.

REM 3. Type checking (non-blocking)
echo ▶ Type checking (mypy)...
mypy src/
if %ERRORLEVEL% NEQ 0 (
    echo ⚠ Type checking has warnings (non-blocking)
)
echo.

REM 4. Unit tests
echo ▶ Unit tests (pytest)...
pytest tests/ -v
if %ERRORLEVEL% NEQ 0 (
    echo ✗ Tests failed
    exit /b 1
)
echo ✓ Tests passed
echo.

REM 5. Build
echo ▶ Build package...
uv build
if %ERRORLEVEL% NEQ 0 (
    echo ✗ Build failed
    exit /b 1
)
echo ✓ Build successful
echo.

REM 6. Package check
echo ▶ Package validation (twine)...
twine check dist/*
if %ERRORLEVEL% NEQ 0 (
    echo ✗ Package validation failed
    exit /b 1
)
echo ✓ Package validation passed
echo.

echo ✅ All checks passed!

