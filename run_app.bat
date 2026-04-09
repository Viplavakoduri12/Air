@echo off
setlocal

pushd "%~dp0"

set "PYTHON_EXE="
if exist "%~dp0.deps" if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python312\python.exe"
if not defined PYTHON_EXE if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%~dp0.venv312\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv312\Scripts\python.exe"
if not defined PYTHON_EXE set "PYTHON_EXE=python"

"%PYTHON_EXE%" "%~dp0main.py" %*

set EXIT_CODE=%ERRORLEVEL%
popd
exit /b %EXIT_CODE%
