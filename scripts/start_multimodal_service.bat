
@echo off
echo Starting Multimodal Service (Port 8002)...
echo Please ensure you have activated the correct environment or set it below.

REM Set your Multimodal environment name here
set MULTIMODAL_ENV_NAME=multimodal_env

REM Check if conda is available
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo Conda not found in PATH. Please run from Anaconda Prompt.
    pause
    exit /b
)

REM Activate environment
call conda activate %MULTIMODAL_ENV_NAME%
if %errorlevel% neq 0 (
    echo Failed to activate environment %MULTIMODAL_ENV_NAME%.
    echo Please edit this script and set MULTIMODAL_ENV_NAME to your actual environment name.
    pause
    exit /b
)

REM Set PYTHONPATH
set PYTHONPATH=%~dp0..
cd %~dp0..

REM Run service
python src/services/multimodal/main.py

pause
