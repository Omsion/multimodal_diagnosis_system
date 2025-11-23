
@echo off
echo Starting API Gateway (Port 8000)...
echo This service orchestrates the other two microservices.

REM Set your Gateway environment name here (can be same as DR or base)
set GATEWAY_ENV_NAME=base

REM Check if conda is available
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo Conda not found in PATH. Please run from Anaconda Prompt.
    pause
    exit /b
)

REM Activate environment
call conda activate %GATEWAY_ENV_NAME%
if %errorlevel% neq 0 (
    echo Failed to activate environment %GATEWAY_ENV_NAME%.
    echo Please edit this script and set GATEWAY_ENV_NAME to your actual environment name.
    pause
    exit /b
)

REM Set PYTHONPATH
set PYTHONPATH=%~dp0..
cd %~dp0..

REM Run service
python src/api/main.py

pause
