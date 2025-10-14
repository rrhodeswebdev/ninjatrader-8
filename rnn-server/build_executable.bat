@echo off
REM Build script for creating RNN Server executable with Nuitka on Windows

echo Building RNN Server Executable with Nuitka...
echo ==============================================
echo.

REM Clean previous builds
echo Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist server_app.dist rmdir /s /q server_app.dist
if exist server_app.build rmdir /s /q server_app.build

REM Run Nuitka compilation
echo Starting Nuitka compilation...
echo This will create a standalone directory with the executable.
echo.

uv run python -m nuitka ^
    --standalone ^
    --output-filename=rnn-server.exe ^
    --output-dir=dist ^
    --include-data-dir=models=models ^
    --nofollow-import-to=matplotlib ^
    --nofollow-import-to=PIL ^
    --nofollow-import-to=IPython ^
    --nofollow-import-to=jupyter ^
    --nofollow-import-to=pytest ^
    --nofollow-import-to=setuptools ^
    --assume-yes-for-downloads ^
    --show-progress ^
    --show-memory ^
    --python-flag=no_site ^
    server_app.py

echo.
echo ==============================================
echo Build complete!
echo.
echo Executable location: dist\server_app.dist\rnn-server.exe
echo.
echo The standalone folder contains:
echo   - rnn-server.exe (executable)
echo   - All required dependencies
echo   - models\ directory with trained models
echo.
echo To run the server:
echo   cd dist\server_app.dist
echo   rnn-server.exe
echo.
echo The server will start on http://127.0.0.1:8000
echo Health check: http://127.0.0.1:8000/health-check
echo.
