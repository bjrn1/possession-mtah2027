@echo off
setlocal
cd /d "%~dp0"
py -3.10 --version >nul 2>&1
if errorlevel 1 (
  echo FAIL: Python 3.10 is not installed
  exit /b 1
)
if not exist .venv py -3.10 -m venv .venv
if errorlevel 1 exit /b 1
.venv\Scripts\python -m pip install --quiet --upgrade pip
.venv\Scripts\python -m pip install --quiet -r requirements-dev.lock
if errorlevel 1 (
  echo FAIL: dependency installation
  exit /b 1
)
.venv\Scripts\python -m pip check
if errorlevel 1 exit /b 1
set PYTHONPATH=%~dp0src
set OMP_NUM_THREADS=1
.venv\Scripts\python -m pytest -q tests
if errorlevel 1 (
  echo FAIL: tests
  exit /b 1
)
echo OK: environment ready
