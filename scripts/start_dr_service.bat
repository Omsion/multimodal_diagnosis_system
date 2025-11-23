
@echo off
echo Starting DR Grading Service (Port 8001)...
echo Please ensure you have activated the correct environment or set it below.

REM Set your DR environment name here
set DR_ENV_NAME=dr_env

REM Check if conda is available
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo Conda not found in PATH. Please run from Anaconda Prompt.
    pause
    exit /b
)

REM Activate environment
call conda activate %DR_ENV_NAME%
if %errorlevel% neq 0 (
    echo Failed to activate environment %DR_ENV_NAME%.
    echo Please edit this script and set DR_ENV_NAME to your actual environment name.
    pause
    exit /b
)

REM Set PYTHONPATH
set PYTHONPATH=%~dp0..
cd %~dp0..

REM Run service
python src/services/dr_grading/main.py

pause
