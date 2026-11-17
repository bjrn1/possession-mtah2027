@echo off
setlocal
cd /d "%~dp0"
if "%~3"=="" (
  echo usage: run_windows.bat DATA_DIR OUT_DIR VALIDATOR_SCRIPT
  echo   DATA_DIR          folder that contains tactical\BAR-ATH_tactical_atd.xml and .txt
  echo   OUT_DIR           folder for the submission file
  echo   VALIDATOR_SCRIPT  path to evaluation\validate_submission.py of the official repository
  exit /b 2
)
set DATA=%~1
set OUT=%~2
set VALIDATOR=%~3
set PYTHONPATH=%~dp0src
set OMP_NUM_THREADS=1
set PYTHONHASHSEED=0
if not exist "%OUT%" mkdir "%OUT%"
.venv\Scripts\python -m possession.run --data-dir "%DATA%" --out "%OUT%\possession_run1.csv"
if errorlevel 1 (
  echo FAIL: run 1
  exit /b 1
)
.venv\Scripts\python -m possession.run --data-dir "%DATA%" --out "%OUT%\possession_run2.csv"
if errorlevel 1 (
  echo FAIL: run 2
  exit /b 1
)
fc /b "%OUT%\possession_run1.csv" "%OUT%\possession_run2.csv" >nul
if errorlevel 1 (
  echo FAIL: the two runs differ
  exit /b 1
)
echo OK: the two runs are identical
.venv\Scripts\python "%VALIDATOR%" "%OUT%\possession_run1.csv" > "%OUT%\validator_report.txt" 2>&1
if errorlevel 1 (
  echo FAIL: official validator, see %OUT%\validator_report.txt
  exit /b 1
)
copy /y "%OUT%\possession_run1.csv" "%OUT%\possession.csv" >nul
echo OK: file is valid, validator report in %OUT%\validator_report.txt
