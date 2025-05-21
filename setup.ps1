# PowerShell script for setting up the Entity Extraction project on Windows

param(
    [switch]$UseCPUOnly,
    [switch]$SkipGPUCheck,
    [switch]$Force
)

# Set up error handling
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # Makes downloads faster

# Print banner
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  Entity Extraction Project - Windows Setup Script     " -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

# Check if Python is installed
try {
    $pythonVersion = python --version
    Write-Host "✓ Found $pythonVersion" -ForegroundColor Green
    
    # Check Python version
    $versionString = $pythonVersion -replace "Python ", ""
    $versionParts = $versionString.Split(".")
    $majorVersion = [int]$versionParts[0]
    $minorVersion = [int]$versionParts[1]
    
    if ($majorVersion -lt 3 -or ($majorVersion -eq 3 -and $minorVersion -lt 8)) {
        Write-Host "⚠️ Warning: Python 3.8+ is recommended (found $versionString)" -ForegroundColor Yellow
        
        if (-Not $Force) {
            $confirmation = Read-Host "Continue anyway? (y/n)"
            if ($confirmation -ne 'y') {
                exit 1
            }
        }
    }
} catch {
    Write-Host "✕ Python not found. Please install Python 3.8 or newer." -ForegroundColor Red
    Write-Host "You can download Python from: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Check if virtual environment exists, create if not
if (-Not (Test-Path -Path ".\venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    
    if (-Not (Test-Path -Path ".\venv")) {
        Write-Host "✕ Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "✓ Virtual environment exists" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
try {
    & .\venv\Scripts\Activate.ps1
} catch {
    Write-Host "✕ Failed to activate virtual environment due to PowerShell execution policy." -ForegroundColor Red
    Write-Host "Attempting to set execution policy for current user..." -ForegroundColor Yellow
    
    try {
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
        & .\venv\Scripts\Activate.ps1
    } catch {
        Write-Host "✕ Still unable to activate virtual environment." -ForegroundColor Red
        Write-Host "Please run the following command manually and try again:" -ForegroundColor Yellow
        Write-Host "    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor White
        exit 1
    }
}

# Check if activation was successful
if (-Not $env:VIRTUAL_ENV) {
    Write-Host "✕ Failed to activate virtual environment." -ForegroundColor Red
    Write-Host "You may need to run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Yellow
    exit 1
}
Write-Host "✓ Virtual environment activated" -ForegroundColor Green

# Install pip tools first for better dependency handling
Write-Host "Upgrading pip and installing key tools..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow

if ($UseCPUOnly) {
    # Explicitly disable GPU to avoid any detection attempts
    Write-Host "Installing for CPU-only operation..." -ForegroundColor Cyan
    $env:CUDA_VISIBLE_DEVICES = "-1"
    python -m pip install -r requirements.txt
} else {
    # Standard installation
    python -m pip install -r requirements.txt
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "✕ Failed to install dependencies." -ForegroundColor Red
    exit 1
}
Write-Host "✓ Dependencies installed" -ForegroundColor Green

# Install Jupyter kernel for the virtual environment
Write-Host "Installing Jupyter kernel for this environment..." -ForegroundColor Yellow
python -m ipykernel install --user --name=ner-env --display-name="Python (Entity Extraction)"
Write-Host "✓ Jupyter kernel installed" -ForegroundColor Green

# Create necessary directories if they don't exist
$directories = @("data", "logs\fit", "models\checkpoints", "notebooks")
foreach ($dir in $directories) {
    if (-Not (Test-Path -Path ".\$dir")) {
        New-Item -Path ".\$dir" -ItemType Directory | Out-Null
        Write-Host "Created directory: $dir" -ForegroundColor Yellow
    }
}

# Create a .pth file in the site-packages directory to automatically add the project to Python path
$sitePackagesDir = Join-Path -Path ".\venv\Lib\site-packages" -ChildPath "entity_extraction.pth"
$projectPath = (Get-Item -Path ".").FullName
$projectPath | Out-File -FilePath $sitePackagesDir -Encoding ascii
Write-Host "✓ Created .pth file for automatic Python path configuration" -ForegroundColor Green

# Download dataset if it doesn't exist
if (-Not (Test-Path -Path ".\data\ner.csv")) {
    Write-Host "NER dataset not found in data directory." -ForegroundColor Yellow
    Write-Host "You'll need to download the dataset from Kaggle manually:" -ForegroundColor Yellow
    Write-Host "1. Visit: https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus" -ForegroundColor Cyan
    Write-Host "2. Download the 'ner.csv' file" -ForegroundColor Cyan
    Write-Host "3. Place it in the 'data' directory of this project" -ForegroundColor Cyan
} else {
    Write-Host "✓ NER dataset found in data directory" -ForegroundColor Green
}

# Check for TensorFlow and GPU availability (skip if requested)
if (-Not $SkipGPUCheck) {
    Write-Host "Checking TensorFlow and GPU availability..." -ForegroundColor Yellow

    $gpuCheckScript = @"
import os
import sys
import platform

# Try to configure TensorFlow to show minimal logs but still show errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def check_tensorflow_gpu():
    """Perform comprehensive TensorFlow GPU check"""
    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.system()} {platform.release()}")
    
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        
        # Check for GPU devices
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPU(s) detected: {len(gpus)}")
            for gpu in gpus:
                print(f"  {gpu.name}")
            return True
        else:
            print("No GPU detected. The model will run on CPU only.")
            return False
    except ImportError:
        print("TensorFlow not installed. Install it with: pip install tensorflow")
        return False
    except Exception as e:
        print(f"Error checking TensorFlow: {str(e)}")
        return False

if __name__ == "__main__":
    check_tensorflow_gpu()
"@

    $gpuCheckScript | Out-File -FilePath ".\gpu_check.py" -Encoding utf8
    python .\gpu_check.py
    Remove-Item -Path ".\gpu_check.py"
}

# Create config.bat for easier environment activation
$configBatContent = @"
@echo off
echo Activating Entity Extraction environment...
call %~dp0venv\Scripts\activate.bat
echo.
echo Environment ready! You can now run:
echo   - .\run.bat                     (Run the main application)
echo   - python src\main.py            (Run the main script directly)
echo   - jupyter notebook              (Start Jupyter notebook server)
echo   - jupyter notebook notebooks\ner_training_example.ipynb (Open example notebook)
echo.
set PYTHONPATH=%~dp0;%PYTHONPATH%
"@

$configBatContent | Out-File -FilePath ".\activate.bat" -Encoding ascii
Write-Host "✓ Created activate.bat for easy environment activation" -ForegroundColor Green

# Write summary
Write-Host "`n=======================================================" -ForegroundColor Cyan
Write-Host " Setup Complete! Environment is ready!" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

Write-Host "`nQuick Start Guide:" -ForegroundColor White
Write-Host "1. Activate the environment:" -ForegroundColor White
Write-Host "   - Using PowerShell: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "   - Using CMD:        .\activate.bat" -ForegroundColor Yellow

Write-Host "`n2. Run the application:" -ForegroundColor White
Write-Host "   python src\main.py --data-path data\ner.csv --model-type bilstm-crf" -ForegroundColor Yellow

Write-Host "`n3. For Jupyter notebook:" -ForegroundColor White
Write-Host "   jupyter notebook notebooks\ner_training_example.ipynb" -ForegroundColor Yellow

if ($UseCPUOnly) {
    Write-Host "`n4. Using CPU-only mode" -ForegroundColor White
    Write-Host "   You've configured the environment for CPU-only operation" -ForegroundColor Yellow
} else {
    Write-Host "`n4. For GPU/CPU options:" -ForegroundColor White
    Write-Host "   - Force CPU mode:  .\run.bat --no-gpu" -ForegroundColor Yellow
}

Write-Host "`nTroubleshooting:" -ForegroundColor White
Write-Host "If you encounter import errors, check the following:" -ForegroundColor Yellow
Write-Host "1. Make sure the virtual environment is activated" -ForegroundColor Yellow
Write-Host "2. Use the run.bat script which sets up PYTHONPATH correctly" -ForegroundColor Yellow
Write-Host "3. Add the project directory to PYTHONPATH manually:" -ForegroundColor Yellow
Write-Host "   `$env:PYTHONPATH = '$((Get-Item -Path '.').FullName);`$env:PYTHONPATH'" -ForegroundColor White
