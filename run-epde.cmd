@echo off
setlocal
set "ROOT=%~dp0"
set "PROJECT=%ROOT%projects\pic"
if not exist "%PROJECT%\uv.lock" (
  echo [EPDE] Missing projects\pic\uv.lock. Use a complete repository checkout.
  exit /b 1
)
if not exist "%PROJECT%\app\Home.py" (
  echo [EPDE] Missing Streamlit application.
  exit /b 1
)
if /I "%~1"=="--check" (
  echo EPDE launcher paths OK: "%PROJECT%"
  exit /b 0
)
set "UV="
for %%U in (uv.exe) do set "UV=%%~$PATH:U"
if not defined UV if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV=%USERPROFILE%\.local\bin\uv.exe"
if not defined UV if exist "%USERPROFILE%\.cargo\bin\uv.exe" set "UV=%USERPROFILE%\.cargo\bin\uv.exe"
if not defined UV (
  echo [EPDE] Installing the official uv Python manager - one time setup...
  powershell.exe -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
  if errorlevel 1 goto :failure
  if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV=%USERPROFILE%\.local\bin\uv.exe"
)
if not defined UV (
  echo [EPDE] uv was not installed. See https://docs.astral.sh/uv/getting-started/installation/
  goto :failure
)
echo [EPDE] Starting full Streamlit UI at http://127.0.0.1:8501
echo [EPDE] Initial setup downloads Python 3.12, PyTorch, and dependencies.
set "EPDE_LOCAL_FULL_RUN=1"
if not defined OMP_NUM_THREADS set "OMP_NUM_THREADS=1"
if not defined OPENBLAS_NUM_THREADS set "OPENBLAS_NUM_THREADS=1"
"%UV%" run --project "%PROJECT%" --python 3.12 --locked --extra cpu --extra app streamlit run "%PROJECT%\app\Home.py" --server.address=127.0.0.1 --server.port=8501 %*
if errorlevel 1 goto :failure
exit /b 0
:failure
echo.
echo [EPDE] Startup failed. Read the error above.
if not defined CI (
  pause
)
exit /b 1
