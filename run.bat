@echo off
title Engineering Career Analytics Platform
cd /d "%~dp0"
echo ======================================================================
echo           Starting Engineering Career Analytics Platform
echo ======================================================================
echo Opening executive dashboard in your default browser on port 6699...
start "" http://127.0.0.1:6699/
echo Starting backend server on port 6699...
python app.py
pause
