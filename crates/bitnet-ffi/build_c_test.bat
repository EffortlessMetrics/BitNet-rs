@echo off
REM Build script for C integration test on Windows
REM This script compiles the C integration test and links it with the Rust library

echo Building BitNet C API integration test...

REM Build the Rust library first
echo Building Rust library...
cargo build --release -p bitnet-ffi
if %ERRORLEVEL% neq 0 (
    echo Failed to build Rust library
    exit /b 1
)

REM Set paths
set TARGET_DIR=..\..\target\release
set LIB_NAME=bitnet_ffi

REM Check if we have a C compiler available
where cl >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo Using MSVC compiler...
    REM Compile with MSVC
    cl /std:c11 /W4 /I.\include tests\c_integration_test.c /link %TARGET_DIR%\%LIB_NAME%.lib /OUT:c_integration_test.exe
) else (
    where gcc >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        echo Using GCC compiler...
        REM Compile with GCC (MinGW)
        gcc -std=c99 -Wall -Wextra -I.\include -L%TARGET_DIR% -l%LIB_NAME% tests\c_integration_test.c -o c_integration_test.exe
    ) else (
        echo No C compiler found. Please install either MSVC or MinGW-w64.
        exit /b 1
    )
)

if %ERRORLEVEL% neq 0 (
    echo Failed to compile C integration test
    exit /b 1
)

echo C integration test compiled successfully!

REM Add the target directory to PATH so the DLL can be found
set PATH=%TARGET_DIR%;%PATH%

echo Running C integration test...
c_integration_test.exe
if %ERRORLEVEL% neq 0 (
    echo C integration test failed
    exit /b 1
)

echo C integration test completed successfully!

REM Clean up
del c_integration_test.exe 2>nul

echo Build and test completed!