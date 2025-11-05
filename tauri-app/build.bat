@echo off
echo ========================================
echo Building RNN Trading Server Desktop App
echo ========================================
echo.

echo [1/3] Checking prerequisites...
where cargo >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Rust is not installed!
    echo Please run setup.bat first
    pause
    exit /b 1
)
echo ✓ Rust found

echo.
echo [2/3] Building application...
echo This may take several minutes...
echo.
call npm run build
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Build failed!
    echo.
    echo Common fixes:
    echo 1. Run setup.bat first
    echo 2. Make sure icons exist in src-tauri/icons/
    echo 3. Try: cd src-tauri ^&^& cargo clean ^&^& cd ..
    echo.
    pause
    exit /b 1
)

echo.
echo [3/3] Build complete!
echo.
echo ========================================
echo Installer location:
echo ========================================
echo.
echo src-tauri\target\release\bundle\msi\
echo.
echo Look for: RNN Trading Server_1.0.0_x64_en-US.msi
echo.
echo ========================================
echo.
pause

