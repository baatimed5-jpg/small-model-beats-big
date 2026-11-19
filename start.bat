@echo off
cd /d "%~dp0"
echo Loading the AI model (about 20 seconds)...
start "" cmd /c "timeout /t 20 >nul & start http://127.0.0.1:8002"
.venv\Scripts\python api.py
