@echo off
setlocal enabledelayedexpansion

REM ======================================================
REM   NER Project - Unified Run Script              
REM ======================================================

echo +----------------------------------------------------+
echo ^|  Named Entity Recognition (NER) - Unified Runner   ^|
echo ^|  Deep Learning Text Analysis Project               ^|
echo +----------------------------------------------------+

REM Default configuration values
set BATCH_SIZE=32
set EPOCHS=15
set LEARNING_RATE=0.001
set MODEL_TYPE=bilstm-crf
set MAX_SEQ_LENGTH=128
set EMBEDDING_DIM=100
set HIDDEN_DIM=200
set ENHANCEMENTS=default
set VERBOSITY=normal
set USE_GPU=true

REM Activate the virtual environment
call .\venv\Scripts\activate.bat 2>nul
if %ERRORLEVEL% NEQ 0 (
    call .\env\Scripts\activate.bat 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo WARNING: Could not activate virtual environment. Using system Python installation.
        echo Hint: Run setup.bat to create a virtual environment
    ) else (
        echo Virtual environment activated
    )
) else (
    echo Virtual environment activated
)

REM Set environment variables
set PYTHONPATH=%PYTHONPATH%;%cd%
set MODEL_TYPE=bilstm-crf
set MAX_SEQ_LENGTH=128
set EMBEDDING_DIM=100
set HIDDEN_DIM=200
set USE_GPU=yes
set USE_JUPYTER=no
set USE_CHAR_FEATURES=yes
set USE_PRETRAINED_EMBEDDINGS=no
set ENHANCEMENTS=default

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :end_parse_args

REM Core parameters
if /i "%~1"=="--batch-size" (
    set BATCH_SIZE=%~2
    shift
) else if /i "%~1"=="--epochs" (
    set EPOCHS=%~2
    shift
) else if /i "%~1"=="--learning-rate" (
    set LEARNING_RATE=%~2
    shift
) else if /i "%~1"=="--model-type" (
    set MODEL_TYPE=%~2
    shift
) else if /i "%~1"=="--max-seq-length" (
    set MAX_SEQ_LENGTH=%~2
    shift

REM Enhancement options
) else if /i "%~1"=="--basic" (
    set ENHANCEMENTS=basic
) else if /i "%~1"=="--full" (
    set ENHANCEMENTS=full
) else if /i "%~1"=="--embeddings" (
    set ENHANCEMENTS=embeddings
) else if /i "%~1"=="--char-features" (
    set ENHANCEMENTS=char_features

REM Environment options
) else if /i "%~1"=="--no-gpu" (
    set USE_GPU=no
) else if /i "%~1"=="--jupyter" (
    set USE_JUPYTER=yes
) else if /i "%~1"=="--help" (
    goto :show_help
)

REM Add support for missing arguments
) else if /i "%~1"=="--mode" (
    set MODE=%~2
    shift
) else if /i "%~1"=="--predict-text" (
    set PREDICT_TEXT=%~2
    shift
) else if /i "%~1"=="--model-dir" (
    set MODEL_DIR=%~2
    shift
) else if /i "%~1"=="--use-char-features" (
    set USE_CHAR_FEATURES=yes
) else if /i "%~1"=="--no-pretrained-embeddings" (
    set USE_PRETRAINED_EMBEDDINGS=no

REM Pass all arguments to main.py
set MODEL_ARGS=--mode %MODE% --predict-text "%PREDICT_TEXT%" --model-dir "%MODEL_DIR%" --batch-size %BATCH_SIZE% --epochs %EPOCHS% --learning-rate %LEARNING_RATE% --model-type %MODEL_TYPE% --max-seq-length %MAX_SEQ_LENGTH% --use-char-features %USE_CHAR_FEATURES% --no-pretrained-embeddings %USE_PRETRAINED_EMBEDDINGS%
)

shift
goto :parse_args
:end_parse_args

REM Show help if requested
if "%1"=="--help" goto :show_help

REM Configure based on enhancement level
if "%ENHANCEMENTS%"=="basic" (
    echo Running with basic configuration (no enhancements)
    set MODEL_ARGS=--model-type %MODEL_TYPE% --batch-size %BATCH_SIZE% --epochs %EPOCHS% --learning-rate %LEARNING_RATE% --no-char-features --no-pretrained-embeddings
) else if "%ENHANCEMENTS%"=="full" (
    echo Running with all enhancements (pre-trained embeddings + character features)
    set MODEL_ARGS=--model-type %MODEL_TYPE% --batch-size %BATCH_SIZE% --epochs %EPOCHS% --learning-rate %LEARNING_RATE% --use-char-features --use-pretrained-embeddings --embedding-dim %EMBEDDING_DIM% --hidden-dim %HIDDEN_DIM%
) else if "%ENHANCEMENTS%"=="embeddings" (
    echo Running with pre-trained embeddings
    set MODEL_ARGS=--model-type %MODEL_TYPE% --batch-size %BATCH_SIZE% --epochs %EPOCHS% --learning-rate %LEARNING_RATE% --no-char-features --use-pretrained-embeddings --embedding-dim %EMBEDDING_DIM% --hidden-dim %HIDDEN_DIM%
) else if "%ENHANCEMENTS%"=="char_features" (
    echo Running with character features
    set MODEL_ARGS=--model-type %MODEL_TYPE% --batch-size %BATCH_SIZE% --epochs %EPOCHS% --learning-rate %LEARNING_RATE% --use-char-features --no-pretrained-embeddings --embedding-dim %EMBEDDING_DIM% --hidden-dim %HIDDEN_DIM%
) else (
    echo Running with default configuration (char features, no pre-trained embeddings)
    set MODEL_ARGS=--model-type %MODEL_TYPE% --batch-size %BATCH_SIZE% --epochs %EPOCHS% --learning-rate %LEARNING_RATE% --use-char-features --no-pretrained-embeddings
)

REM Add GPU flag if needed
if "%USE_GPU%"=="no" (
    set MODEL_ARGS=%MODEL_ARGS% --no-gpu
)

REM Display configuration
echo.
echo ======================================
echo      NER MODEL CONFIGURATION
echo ======================================
echo Model Type: %MODEL_TYPE%
echo Batch Size: %BATCH_SIZE%
echo Epochs: %EPOCHS%
echo Learning Rate: %LEARNING_RATE%
echo Use GPU: %USE_GPU%
echo Enhancement Profile: %ENHANCEMENTS%
echo.

REM Run Jupyter if requested
if "%USE_JUPYTER%"=="yes" (
    echo Starting Jupyter notebook...
    jupyter notebook notebooks\ner_training_example.ipynb
    goto :eof
)

REM Run the model
echo Starting NER model training...
python src\main.py --batch-size %BATCH_SIZE% --epochs %EPOCHS% --learning-rate %LEARNING_RATE% --model-type %MODEL_TYPE% --max-seq-length %MAX_SEQ_LENGTH% --use-gpu %USE_GPU% --use-xla %USE_XLA% --use-mixed-precision %USE_MIXED_PRECISION% %ENHANCEMENTS%

if %ERRORLEVEL% NEQ 0 (
    echo Error running the model. Check the logs for details.
    exit /b 1
) else (
    echo Training completed successfully.
)

goto :eof

:show_help
echo.
echo Usage: run.bat [options]
echo.
echo Options:
echo   --batch-size N      Set batch size (default: 32)
echo   --epochs N          Set number of epochs (default: 15)
echo   --learning-rate N   Set learning rate (default: 0.001)
echo   --model-type TYPE   Set model type: bilstm, bilstm-crf, transformer (default: bilstm-crf)
echo   --max-seq-length N  Set maximum sequence length (default: 128)
echo   --no-gpu            Disable GPU acceleration
echo   --jupyter           Launch Jupyter notebook instead of training
echo.
echo Enhancement profiles:
echo   --basic             Use basic configuration without enhancements
echo   --full              Use all enhancements (pre-trained embeddings + char features)
echo   --embeddings        Use pre-trained embeddings only
echo   --char-features     Use character features only
echo.
echo Example:
echo   run.bat --full --batch-size 64 --epochs 20
echo.
exit /b 0

:eof
endlocal
