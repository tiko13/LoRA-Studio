@echo off
setlocal
cd /d "%~dp0" || exit /b 1
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if errorlevel 1 goto failed
)
.venv\Scripts\python.exe -c "import PIL" >nul 2>&1
if errorlevel 1 (
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  if errorlevel 1 goto failed
)
.venv\Scripts\python.exe server.py --open %*
if errorlevel 1 goto failed
exit /b 0
:failed
set "launch_exit_code=%errorlevel%"
echo.
echo LoRA Studio could not start. Review the error above.
pause
exit /b %launch_exit_code%
