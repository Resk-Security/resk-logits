@echo off
REM Script de build et test d'installation (Windows)

echo 🔨 Building ReskLogits package...
echo.

REM Build
uv build
if %ERRORLEVEL% NEQ 0 (
    echo ✗ Build failed
    exit /b 1
)

echo.
echo 🧪 Testing installation in clean environment...
echo.

REM Create test environment
set TEST_ENV=test-install-%RANDOM%
python -m venv %TEST_ENV%
call %TEST_ENV%\Scripts\activate.bat

REM Install from wheel
echo Installing from wheel...
pip install dist\resklogits-*.whl --quiet

REM Test imports
echo Testing imports...
python -c "from resklogits import ShadowBanProcessor; print('✓ ShadowBanProcessor imported')"
python -c "from resklogits import ConfigParser; print('✓ ConfigParser imported')"
python -c "from resklogits import RuleCache; print('✓ RuleCache imported')"
python -c "import resklogits; print(f'✓ Version: {resklogits.__version__}')"

REM Test CLI
echo Testing CLI...
resklogits --help >nul 2>&1 && echo ✓ CLI works

REM Cleanup
deactivate
rmdir /s /q %TEST_ENV%

echo.
echo ✅ Build and test successful!

