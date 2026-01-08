@echo off
setlocal EnableDelayedExpansion

echo ==========================================
echo Python Virtual Environment Setup Script
echo ==========================================
echo.

echo Detecting installed Python versions...
where python > "%TEMP%\py_list.txt" 2> nul

set INDEX=0

for /f "delims=" %%A in ("%TEMP%\py_list.txt") do (
    set /a INDEX+=1
    set "PYTHON_!INDEX!=%%A"
    echo   !INDEX!. %%A
)

del "%TEMP%\py_list.txt" 2> nul

if %INDEX%==0 (
    echo No Python installation found.
    echo Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

echo.
set /p PY_CHOICE=Select Python version number ^>

if "%PY_CHOICE%"=="" (
    echo No selection provided.
    pause 
    exit /b 1
)

if not defined PYTHON_%PY_CHOICE% (
    echo Invalid selection.
    pause
    exit /b 1

)

set "PYTHON_EXEC=!PYTHON_%PY_CHOICE%!"
echo Selected: %PYTHON_EXEC%
echo.

%PYTHON_EXEC% --version

if errorlevel 1 (
    echo Failed to run selected Python executable.
    pause
    exit /b 1
)

if exist .venv\Scripts\activate (
    echo Virtual environment already exists.
) else (
    echo Creating virtual environment...
    %PYTHON_EXEC% -m venv .venv
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause 
        exit /b 1
    )

)

echo.

if exist requirements.txt (
    echo Upgrading pip...
    .venv\Scripts\python.exe -m pip install --upgrade pip

    echo Installing dependencies...
    .venv\Scripts\python.exe -m pip install -r requirements.txt

    if errorlevel 1 (
        echo Dependency installation failed.
        pause
        exit /b 1
    ) 
) else (
    echo requirements.txt not found. Skipping dependency installation.
)

echo.
echo ==========================================
echo Setup complete!
echo ==========================================
echo.
echo To activate the virtual environment, run:
echo.
echo     call .venv\Scripts\activate
echo.

pause
endlocal
