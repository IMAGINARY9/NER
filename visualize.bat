@echo off
setlocal enabledelayedexpansion

REM ======================================================
REM   NER Project - Unified Visualization Script              
REM ======================================================

echo +----------------------------------------------------+
echo ^|  Named Entity Recognition (NER) - Visualization   ^|
echo ^|  Model Performance Analysis & Entity Visualizer   ^|
echo +----------------------------------------------------+

REM Activate the virtual environment
call .\venv\Scripts\activate.bat 2>nul
if %ERRORLEVEL% NEQ 0 (
    call .\env\Scripts\activate.bat 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Virtual environment not found. Please create and activate it first.
        echo Hint: Run setup.bat to create a virtual environment
        exit /b 1
    ) else (
        echo Using env virtual environment
    )
) else (
    echo Using venv virtual environment
)

REM Set environment variables to suppress TensorFlow and TFA warnings
echo Configuring environment to suppress common TensorFlow warnings...
set TF_CPP_MIN_LOG_LEVEL=2
set PYTHONWARNINGS=ignore::UserWarning,ignore::DeprecationWarning,ignore::FutureWarning
set TF_USE_LEGACY_KERAS=1
set CUDA_VISIBLE_DEVICES=-1

echo Enabling TensorFlow compatibility mode for newer versions...


REM Set environment variables
set PYTHONPATH=%PYTHONPATH%;%cd%

REM Default paths and settings
set MODEL_PATH=
set DATA_PATH=data\ner.csv
set INTERACTIVE=false
set OUTPUT_DIR=visualizations
set VISUALIZATION_TYPE=all

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :end_parse_args

if /i "%~1"=="--help" (
    echo Usage: visualize_unified.bat [options]
    echo.
    echo Options:
    echo   --model-path PATH    Path to the model file (default: latest model)
    echo   --data-path PATH     Path to the data file (default: data\ner.csv)
    echo   --interactive        Run in interactive mode with UI controls
    echo   --output-dir DIR     Directory to save visualizations (default: visualizations)
    echo   --type TYPE          Visualization type: confusion, entities, token_distribution, all
    echo   --help               Display this help message
    exit /b 0
)

if /i "%~1"=="--model-path" (
    set MODEL_PATH=%~2
    shift
    shift
    goto :parse_args
)

if /i "%~1"=="--data-path" (
    set DATA_PATH=%~2
    shift
    shift
    goto :parse_args
)

if /i "%~1"=="--output-dir" (
    set OUTPUT_DIR=%~2
    shift
    shift
    goto :parse_args
)

if /i "%~1"=="--type" (
    set VISUALIZATION_TYPE=%~2
    shift
    shift
    goto :parse_args
)

if /i "%~1"=="--interactive" (
    set INTERACTIVE=true
    shift
    goto :parse_args
)

REM Unknown argument, just skip it
shift
goto :parse_args
:end_parse_args

REM Find latest model if not specified
if "%MODEL_PATH%"=="" (
    echo Model path not specified, searching for latest model...
    
    for /f "tokens=*" %%F in ('dir /b /o-d /a-d "models\*.keras" 2^>nul') do (
        set "MODEL_PATH=models\%%F"
        goto :found_model
    )
    echo No model files found in models directory.
    exit /b 1
    :found_model
    echo Found latest model: %MODEL_PATH%
)

REM Check if the data file exists
if not exist "%DATA_PATH%" (
    echo Data file not found: %DATA_PATH%
    exit /b 1
)

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%"
    echo Created output directory: %OUTPUT_DIR%
)

REM Show configuration
echo Running NER model visualization with:
echo Model Path: %MODEL_PATH%
echo Data Path: %DATA_PATH%
echo Output Directory: %OUTPUT_DIR%
echo Visualization Type: %VISUALIZATION_TYPE%
if "%INTERACTIVE%"=="true" (
    echo Mode: Interactive
) else (
    echo Mode: Static (non-interactive)
)
echo.

REM Build command arguments
set VISUALIZE_ARGS=--model-path "%MODEL_PATH%" --data-path "%DATA_PATH%" --output-dir "%OUTPUT_DIR%" --type %VISUALIZATION_TYPE%
if "%INTERACTIVE%"=="true" (
    set VISUALIZE_ARGS=%VISUALIZE_ARGS% --interactive
)

REM Run visualization
echo Starting visualization process...
echo Attempting to load model with enhanced compatibility for TensorFlow 2.13+...
python src\visualize.py %VISUALIZE_ARGS%

if %ERRORLEVEL% NEQ 0 (
    echo Error visualizing the model. Check the logs for details.
    echo You may try running with visualize_enhanced.bat which has additional compatibility settings.
    exit /b 1
) else (
    echo Visualization completed successfully.
    echo Results saved in: %OUTPUT_DIR%
)

REM Reset environment variables
set TF_CPP_MIN_LOG_LEVEL=
set PYTHONWARNINGS=
set TF_USE_LEGACY_KERAS=
set CUDA_VISIBLE_DEVICES=

endlocal
