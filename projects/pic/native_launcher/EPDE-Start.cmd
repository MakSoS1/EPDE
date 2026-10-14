@echo off
setlocal
title EPDE Explorer - first run may download Python and scientific libraries
echo EPDE Explorer native launcher - real EPDE and PySINDy
echo.
echo This OPTIONAL program runs searches on YOUR CPU. It contacts GitHub and
echo the official uv package service to install the exact pinned Python environment.
echo Nothing is executed on your computer unless you explicitly open this file.
echo First setup needs internet and may take a long time because PyTorch and data are large.
echo.
powershell.exe -NoProfile -Command "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; $root=Join-Path $env:LOCALAPPDATA 'EPDE-Explorer'; New-Item -ItemType Directory -Path $root -Force | Out-Null; $uvdir=Join-Path $root 'uv'; $uv=Join-Path $uvdir 'uv.exe'; if(!(Test-Path $uv)){ $zip=Join-Path $root 'uv.zip'; Write-Host 'Downloading the official uv executable...'; Invoke-WebRequest 'https://github.com/astral-sh/uv/releases/download/0.13.0/uv-x86_64-pc-windows-msvc.zip' -OutFile $zip; $hash=(Get-FileHash $zip -Algorithm SHA256).Hash.ToLowerInvariant(); if($hash -ne '088962f9e7b7bd9ea740c04c650b2a21c8928c345bd99ac24350dc924dba656c'){throw 'Official uv archive hash mismatch'}; Expand-Archive -Path $zip -DestinationPath $uvdir -Force; Remove-Item $zip -Force; if(!(Test-Path $uv)){throw 'uv.exe missing from verified official release archive'} }; $source=Join-Path $root 'checkout'; $marker=Join-Path $source '.ready'; if(!(Test-Path $marker) -and !$env:EPDE_SOURCE_OVERRIDE){ $zip=Join-Path $root 'repo.zip'; Write-Host 'Downloading the versioned EPDE scientific source...'; Invoke-WebRequest 'https://github.com/MakSoS1/EPDE/archive/2a338d356cea510972e6057619e3365264baddb3.zip' -OutFile $zip; New-Item -ItemType Directory -Path $source -Force | Out-Null; Expand-Archive -Path $zip -DestinationPath $source -Force; Remove-Item $zip -Force; $base=Get-ChildItem $source -Directory | Select-Object -First 1; if(!$base){throw 'Missing EPDE project source'}; Set-Content -Path $marker -Value $base.FullName -Encoding UTF8 }; $checkout=if($env:EPDE_SOURCE_OVERRIDE){$env:EPDE_SOURCE_OVERRIDE}else{(Get-Content -Path $marker -Raw).Trim()}; $pic=Join-Path $checkout 'projects\pic'; if(!(Test-Path (Join-Path $pic 'uv.lock'))){throw 'The pinned EPDE lockfile is missing'}; Write-Host 'Installing Python 3.12 and the pinned EPDE + PySINDy dependencies...'; & $uv python install 3.12; if($LASTEXITCODE -ne 0){throw 'Python 3.12 bootstrap failed'}; & $uv sync --project $pic --python 3.12 --locked --extra cpu --extra app; if($LASTEXITCODE -ne 0){throw 'Pinned scientific dependencies failed to install'}; if($env:EPDE_SETUP_ONLY -eq '1'){ & $uv run --project $pic --no-sync python -c 'import epde,pysindy,torch; print(1)'; exit $LASTEXITCODE }; Write-Host 'Starting the local CPU worker and opening the existing website...'; Start-Process 'https://epde-explorer-demo.streamlit.app/'; & $uv run --project $pic --no-sync python (Join-Path $pic 'app\local_worker_service.py'); exit $LASTEXITCODE"
if errorlevel 1 (
  echo.
  echo EPDE setup or launch failed. Check the message above.
  echo This program never needs Administrator permissions.
  if "%EPDE_SETUP_ONLY%"=="1" exit /b 1
  pause
  exit /b 1
)
endlocal
exit /b 0
