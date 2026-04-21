@echo off
title FaceGuard Server Launcher
echo =========================================
echo       Starting FaceGuard Server...
echo =========================================
echo.

:: Get the directory of this script
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo Current Directory: %CD%

:: Automatically open the browser to the dashboard
start http://localhost:5001

:: Run the application using the stabilized Conda environment
:: Using the directory discovered during our session
C:\Users\sansk\miniconda3\envs\faceguard_env\python.exe src2/app.py

echo.
echo Server has stopped or crashed. See any errors above.
pause
