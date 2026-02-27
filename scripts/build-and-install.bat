@echo off
REM Usage: build-and-install.bat [build-dir] [install-prefix] [build-type] [extra-cmake-flags]

SETLOCAL ENABLEDELAYEDEXPANSION
set "BUILD_DIR=%~1"
if "%BUILD_DIR%"=="" set "BUILD_DIR=build"
set "INSTALL_PREFIX=%~2"
if "%INSTALL_PREFIX%"=="" set "INSTALL_PREFIX=C:\Program Files\MinkowskiEngine"
set "BUILD_TYPE=%~3"
if "%BUILD_TYPE%"=="" set "BUILD_TYPE=Release"
set "EXTRA_FLAGS=%~4"

echo Build dir: %BUILD_DIR%
echo Install prefix: %INSTALL_PREFIX%
echo Build type: %BUILD_TYPE%

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
pushd "%BUILD_DIR%"

cmake -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DCMAKE_INSTALL_PREFIX="%INSTALL_PREFIX%" %EXTRA_FLAGS% ..
cmake --build . --config %BUILD_TYPE%
cmake --install . --config %BUILD_TYPE%

popd
echo Build and install complete.
ENDLOCAL
