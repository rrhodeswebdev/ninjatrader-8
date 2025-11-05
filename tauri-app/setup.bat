@echo off
echo ========================================
echo RNN Trading Server Desktop App Setup
echo ========================================
echo.

echo [1/4] Checking prerequisites...
echo.

where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Node.js is not installed!
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)
echo ✓ Node.js found

where cargo >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Rust is not installed!
    echo Please install Rust from https://rustup.rs/
    pause
    exit /b 1
)
echo ✓ Rust found

where uv >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: uv is not installed!
    echo The app needs uv to run the Python server.
    echo Install it with: pip install uv
    echo.
    echo Continue anyway? (Y/N)
    set /p continue=
    if /i not "%continue%"=="Y" exit /b 1
) else (
    echo ✓ uv found
)

echo.
echo [2/4] Installing Node.js dependencies...
call npm install
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install Node.js dependencies
    pause
    exit /b 1
)

echo.
echo [3/4] Generating default icons...
call npx @tauri-apps/cli icon
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Failed to generate icons
    echo You may need to create icons manually
    echo See generate-icons.md for instructions
)

echo.
echo [4/4] Setup complete!
echo.
echo ========================================
echo Next steps:
echo ========================================
echo.
echo Development mode:
echo   npm run dev
echo.
echo Build for production:
echo   npm run build
echo.
echo ========================================
pause

