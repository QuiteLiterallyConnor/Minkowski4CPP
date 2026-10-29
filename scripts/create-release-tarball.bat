@echo off
REM Usage: create-release-tarball.bat <version> [staging-dir]

SETLOCAL ENABLEDELAYEDEXPANSION
set "VERSION=%~1"
if "%VERSION%"=="" set "VERSION=v1.1.0"
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
if errorlevel 1 exit /b 1

REM cmake --install has no --destdir option; install straight into the staging dir.
cmake --install build --prefix "%STAGING_DIR%" --config Release
if errorlevel 1 exit /b 1

set "TARBALL=minkowskiengine-%VERSION%-windows.zip"
set "OUT_DIR=%CD%\"

REM Prefer tar if available (Windows 10+), otherwise use PowerShell Compress-Archive
where tar >nul 2>&1
if %ERRORLEVEL%==0 (
  REM Name the top-level entries explicitly: a "./" prefix makes Explorer show the zip as empty.
  pushd "%STAGING_DIR%"
  tar -a -cf "%OUT_DIR%%TARBALL%" *
  popd
) else (
  powershell -Command "Compress-Archive -Path '%STAGING_DIR%\*' -DestinationPath '%TARBALL%'"
)

echo Created %TARBALL%
ENDLOCAL
