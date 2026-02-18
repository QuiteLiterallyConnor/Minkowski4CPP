@echo off
REM Usage: create-release-tarball.bat <version> [staging-dir]

SETLOCAL ENABLEDELAYEDEXPANSION
set "VERSION=%~1"
if "%VERSION%"=="" set "VERSION=v1.0.0"
set "STAGING_DIR=%~2"
if "%STAGING_DIR%"=="" (
  set "STAGING_DIR=%TEMP%\mink_stage_%RANDOM%"
)
set "INSTALL_PREFIX=C:\Program Files\MinkowskiEngine"

echo Version: %VERSION%
echo Staging dir: %STAGING_DIR%

if not exist build mkdir build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="%INSTALL_PREFIX%"
cmake --build build --config Release

cmake --install build --prefix "%INSTALL_PREFIX%" --destdir "%STAGING_DIR%" --config Release

set "TARBALL=minkowskiengine-%VERSION%-windows.zip"

REM Prefer tar if available (Windows 10+), otherwise use PowerShell Compress-Archive
where tar >nul 2>&1
if %ERRORLEVEL%==0 (
  tar -C "%STAGING_DIR%" -czf "%TARBALL%" .
) else (
  powershell -Command "Compress-Archive -Path '%STAGING_DIR%\*' -DestinationPath '%TARBALL%'"
)

echo Created %TARBALL%
ENDLOCAL
