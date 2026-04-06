@echo off
setlocal

pushd "%~dp0"

set "PYTHON_EXE="
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%~dp0venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python312\python.exe"
if not defined PYTHON_EXE set "PYTHON_EXE=python"

if not exist "xgboost_aqi_model.pkl" (
    echo ERROR: xgboost_aqi_model.pkl was not found in "%~dp0"
    echo Place the model file next to app.py, then run this script again.
    popd
    exit /b 1
)

"%PYTHON_EXE%" -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --onedir ^
  --name AirQualityPrediction ^
  --collect-all streamlit ^
  --collect-all matplotlib ^
  --collect-all xgboost ^
  --collect-all altair ^
  --collect-all pydeck ^
  --hidden-import streamlit.web.cli ^
  --add-data "app.py;." ^
  --add-data "xgboost_aqi_model.pkl;." ^
  main.py

set EXIT_CODE=%ERRORLEVEL%
popd
exit /b %EXIT_CODE%
