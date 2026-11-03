@echo off
cd /d "%~dp0"
if exist "%USERPROFILE%\miniconda3\envs\lstm-py312\python.exe" (
  "%USERPROFILE%\miniconda3\envs\lstm-py312\python.exe" run_all.py %*
) else (
  call conda run --no-capture-output -n lstm-py312 python run_all.py %*
)
pause
