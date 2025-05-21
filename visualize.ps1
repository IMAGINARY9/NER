# Named Entity Recognition (NER) - Unified Visualization Script
# This script provides comprehensive visualization for NER model performance

# Banner
Write-Host "┌────────────────────────────────────────────────────────┐" -ForegroundColor Cyan
Write-Host "│ Named Entity Recognition (NER) - Visualization Tool    │" -ForegroundColor Cyan 
Write-Host "│ Model Performance Analysis & Entity Visualizer         │" -ForegroundColor Cyan
Write-Host "└────────────────────────────────────────────────────────┘" -ForegroundColor Cyan
Write-Host ""

# Activate the virtual environment (trying both possible paths)
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    & ".\venv\Scripts\Activate.ps1"
    Write-Host "Virtual environment activated" -ForegroundColor Green
} elseif (Test-Path ".\env\Scripts\Activate.ps1") {
    & ".\env\Scripts\Activate.ps1" 
    Write-Host "Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "Virtual environment not found. Please create and activate it first." -ForegroundColor Red
    Write-Host "Hint: Run setup.ps1 to create a virtual environment" -ForegroundColor Yellow
    exit 1
}

# Set environment variables to suppress TensorFlow and TFA warnings
Write-Host "Configuring environment to suppress common TensorFlow warnings..." -ForegroundColor Blue
$env:TF_CPP_MIN_LOG_LEVEL = "2"
$env:PYTHONWARNINGS = "ignore::UserWarning,ignore::DeprecationWarning,ignore::FutureWarning"
$env:TF_USE_LEGACY_KERAS = "1"  # For TensorFlow 2.13+ compatibility
$env:CUDA_VISIBLE_DEVICES = "-1" # Use CPU only to avoid GPU-related errors

Write-Host "TensorFlow compatibility mode enabled for newer versions" -ForegroundColor Blue

# Set environment variables
$env:PYTHONPATH = "$env:PYTHONPATH;$(Get-Location)"

# Default paths and settings
$MODEL_PATH = ""
$DATA_PATH = "data\ner.csv"
$INTERACTIVE = $false
$OUTPUT_DIR = "visualizations"
$VISUALIZATION_TYPE = "all"  # Options: confusion, entities, token_distribution, all

# Help message function
function Show-Help {
    Write-Host "Usage: .\visualize_unified.ps1 [options]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Green
    Write-Host "  --model-path PATH    Path to the model file (default: latest model)"
    Write-Host "  --data-path PATH     Path to the data file (default: data\ner.csv)"
    Write-Host "  --interactive        Run in interactive mode with UI controls"
    Write-Host "  --output-dir DIR     Directory to save visualizations (default: visualizations)"
    Write-Host "  --type TYPE          Visualization type: confusion, entities, token_distribution, all"
    Write-Host "  --help               Display this help message"
}

# Parse command line arguments
$i = 0
$skipNext = $false

for ($i = 0; $i -lt $args.Count; $i++) {
    if ($skipNext) {
        $skipNext = $false
        continue
    }

    switch ($args[$i]) {
        # Help command
        "--help" {
            Show-Help
            exit 0
        }

        # Options with values
        "--model-path" {
            $MODEL_PATH = $args[$i+1]
            $skipNext = $true
        }
        "--data-path" {
            $DATA_PATH = $args[$i+1]
            $skipNext = $true
        }
        "--output-dir" {
            $OUTPUT_DIR = $args[$i+1]
            $skipNext = $true
        }
        "--type" {
            $VISUALIZATION_TYPE = $args[$i+1]
            $skipNext = $true
        }
        
        # Flag options
        "--interactive" {
            $INTERACTIVE = $true
        }
    }
}

# Find latest model if not specified
if ([string]::IsNullOrEmpty($MODEL_PATH)) {
    Write-Host "Model path not specified, searching for latest model..." -ForegroundColor Cyan
    
    $latestModel = Get-ChildItem "models\*.keras" -ErrorAction SilentlyContinue | 
                   Sort-Object -Property LastWriteTime -Descending | 
                   Select-Object -First 1
    
    if ($latestModel) {
        $MODEL_PATH = $latestModel.FullName
        Write-Host "Found latest model: $MODEL_PATH" -ForegroundColor Green
    } else {
        Write-Host "No model files found in models directory." -ForegroundColor Red
        exit 1
    }
}

# Check if the data file exists
if (-not (Test-Path $DATA_PATH)) {
    Write-Host "Data file not found: $DATA_PATH" -ForegroundColor Red
    exit 1
}

# Create output directory if it doesn't exist
if (-not (Test-Path $OUTPUT_DIR)) {
    New-Item -Path $OUTPUT_DIR -ItemType Directory | Out-Null
    Write-Host "Created output directory: $OUTPUT_DIR" -ForegroundColor Green
}

# Show configuration
Write-Host "Running NER model visualization with:" -ForegroundColor Green
Write-Host "Model Path: $MODEL_PATH"
Write-Host "Data Path: $DATA_PATH" 
Write-Host "Output Directory: $OUTPUT_DIR"
Write-Host "Visualization Type: $VISUALIZATION_TYPE"
if ($INTERACTIVE) {
    Write-Host "Mode: Interactive"
} else {
    Write-Host "Mode: Static (non-interactive)"
}
Write-Host ""

# Build command arguments
$visualizeArgs = "--model-path `"$MODEL_PATH`" --data-path `"$DATA_PATH`" --output-dir `"$OUTPUT_DIR`" --type $VISUALIZATION_TYPE"
if ($INTERACTIVE) {
    $visualizeArgs += " --interactive"
}

# Run visualization
Write-Host "Starting visualization process..." -ForegroundColor Cyan
Write-Host "Attempting to load model with enhanced compatibility for TensorFlow 2.13+..." -ForegroundColor Yellow
python src\visualize.py $visualizeArgs.Split(" ")

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error visualizing the model. Check the logs for details." -ForegroundColor Red
    exit 1
} else {
    Write-Host "Visualization completed successfully." -ForegroundColor Green
    Write-Host "Results saved in: $OUTPUT_DIR" -ForegroundColor Green
}

# Reset environment variables
$env:TF_CPP_MIN_LOG_LEVEL = ""
$env:PYTHONWARNINGS = ""
$env:TF_USE_LEGACY_KERAS = ""
$env:CUDA_VISIBLE_DEVICES = ""

Write-Host "Environment restored to default settings." -ForegroundColor DarkGray
Write-Host "Exiting visualization script." -ForegroundColor Cyan
