@echo off
setlocal
set "ROOT=%~dp0"
set "PROJECT=%ROOT%projects\pic"
set "BUNDLE=%ROOT%experiment.zip"
if not exist "%BUNDLE%" (
  echo Copy the downloaded experiment ZIP here and name it experiment.zip.
  pause
  exit /b 1
)
set "UV="
for %%U in (uv.exe) do set "UV=%%~$PATH:U"
if not defined UV if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV=%USERPROFILE%\.local\bin\uv.exe"
if not defined UV (
  echo Installing official uv...
  powershell.exe -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
  if errorlevel 1 goto :error
  if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV=%USERPROFILE%\.local\bin\uv.exe"
)
if not defined UV goto :error
set "OMP_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
echo The first run installs Python 3.12 and scientific dependencies.
"%UV%" run --project "%PROJECT%" --python 3.12 --locked --extra cpu --extra app python "%PROJECT%\portable_experiment.py" --bundle "%BUNDLE%"
if errorlevel 1 goto :error
echo.
echo Upload experiment-results.zip on the website's Results page.
pause
exit /b 0
:error
echo Experiment failed; see the error above.
pause
exit /b 1
