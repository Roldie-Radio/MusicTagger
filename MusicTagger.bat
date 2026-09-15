@echo off
REM Launch MusicTagger. First run installs dependencies into a local venv.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo First run: creating a virtual environment...
  py -3 -m venv .venv || python -m venv .venv || goto :nopython
  echo Installing dependencies, this takes a minute...
  ".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
  ".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt || goto :nodeps
)

".venv\Scripts\pythonw.exe" -m musictag ui %*
goto :eof

:nopython
echo Could not find Python 3.10+. Install it from https://python.org and try again.
pause
goto :eof

:nodeps
echo Dependency installation failed. Run this from a terminal to see the error:
echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
pause
