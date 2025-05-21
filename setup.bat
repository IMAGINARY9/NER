@echo off
REM Setup script for Entity Extraction Project

echo ======================================
echo    NER Project Environment Setup
echo ======================================

REM Create Python virtual environment
if not exist venv (
    echo Creating Python virtual environment...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to create Python virtual environment. Please ensure Python 3.8+ is installed.
        exit /b 1
    )
)

REM Activate virtual environment and install dependencies
echo Activating virtual environment and installing dependencies...
call venv\Scripts\activate
pip install -r requirements.txt

if %ERRORLEVEL% NEQ 0 (
    echo Failed to install required packages. Please check requirements.txt and try again.
    exit /b 1
)

REM Set up Jupyter kernel
echo Setting up Jupyter notebook kernel...
pip install jupyter ipykernel
python -m ipykernel install --user --name=ner-venv --display-name="NER Project (venv)"

if %ERRORLEVEL% NEQ 0 (
    echo Warning: Failed to create Jupyter kernel, but setup will continue.
) else (
    echo Jupyter kernel "NER Project (venv)" created successfully!
)

REM Download dataset if it doesn't exist
if not exist data\ner.csv (
    echo Downloading NER dataset...
    mkdir data
    echo You'll need to download the NER dataset from Kaggle:
    echo 1. Visit https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus
    echo 2. Download the 'ner.csv' file
    echo 3. Place it in the 'data' directory of this project
) else (
    echo Dataset already exists.
)

echo.
echo Setup completed successfully!
echo.
echo To train a model, run:
echo     python src\main.py --data-path data\ner.csv --model-type bilstm-crf
echo.
echo To open a Jupyter notebook:
echo     jupyter notebook notebooks\ner_training_example.ipynb
echo.
echo To deactivate the virtual environment when finished:
echo     deactivate
