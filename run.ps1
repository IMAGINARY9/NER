# Entity Extraction (NER) Project - Unified Runner Script for PowerShell
# This script combines functionality from run.ps1 and run_improved.ps1

# Banner
Write-Host "┌────────────────────────────────────────────────────────┐" -ForegroundColor Cyan
Write-Host "│ Named Entity Recognition (NER) - Unified Runner        │" -ForegroundColor Cyan 
Write-Host "│ Deep Learning Text Analysis Project                    │" -ForegroundColor Cyan
Write-Host "└────────────────────────────────────────────────────────┘" -ForegroundColor Cyan
Write-Host ""

# Help message function
function Show-Help {
    Write-Host "Usage: .\run_unified.ps1 [options]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Common Options:" -ForegroundColor Green
    Write-Host "  --help             Display this help message"
    Write-Host ""
    Write-Host "Environment Options:" -ForegroundColor Green
    Write-Host "  --verbose          Enable verbose logging"
    Write-Host "  --quiet            Minimize output"
    Write-Host "  --no-gpu           Disable GPU acceleration"
    Write-Host "  --test-gpu         Test if GPU is properly configured"
    Write-Host "  --jupyter          Launch Jupyter Notebook server"
    Write-Host ""
    Write-Host "Model Configuration Options:" -ForegroundColor Green
    Write-Host "  --batch-size N     Set batch size (default: 32)"
    Write-Host "  --epochs N         Set number of epochs (default: 15)"
    Write-Host "  --learning-rate N  Set learning rate (default: 0.001)"
    Write-Host "  --model-type TYPE  Set model type: bilstm, bilstm-crf, transformer (default: bilstm-crf)"
    Write-Host ""
    Write-Host "Enhancement Profiles:" -ForegroundColor Green
    Write-Host "  --basic            Use basic configuration without enhancements"
    Write-Host "  --full             Use all enhancements (pre-trained embeddings + character features)"
    Write-Host "  --embeddings       Use pre-trained embeddings only"
    Write-Host "  --char-features    Use character features only"
    Write-Host "  --default          Use default enhancements (char features, no pre-trained embeddings)"
}

# Set default configuration values
$BATCH_SIZE = 32
$EPOCHS = 15
$LEARNING_RATE = 0.001
$MODEL_TYPE = "bilstm-crf"
$MAX_SEQ_LENGTH = 128
$EMBEDDING_DIM = 100
$HIDDEN_DIM = 200
$ENHANCEMENTS = "default"
$verbosityLevel = "normal"
$env:NER_VERBOSITY = $verbosityLevel
$env:TF_CPP_MIN_LOG_LEVEL = "2"  # Suppress TensorFlow warnings

# Activate the virtual environment if it exists
if (Test-Path "$PSScriptRoot\venv\Scripts\Activate.ps1") {
    & "$PSScriptRoot\venv\Scripts\Activate.ps1"
    Write-Host "Virtual environment activated" -ForegroundColor Green
} elseif (Test-Path "$PSScriptRoot\env\Scripts\Activate.ps1") {
    & "$PSScriptRoot\env\Scripts\Activate.ps1"
    Write-Host "Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "No virtual environment found. Please create one first." -ForegroundColor Yellow
    Write-Host "Hint: Run setup.ps1 to create a virtual environment" -ForegroundColor Yellow
}

# Set PYTHONPATH to include the project root directory
$env:PYTHONPATH = "$PSScriptRoot;$env:PYTHONPATH"

# Check if Python is available
try {
    $pythonVersion = (python --version) 2>&1
    Write-Host "Using $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found in PATH" -ForegroundColor Red
    Write-Host "Please install Python or add it to your PATH" -ForegroundColor Red
    Write-Host "Hint: Run setup.ps1 to create a virtual environment" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Parse command line arguments
$i = 0
$skipNext = $false
$remainingArgs = @()

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

        # Environment options
        "--verbose" {
            $env:NER_VERBOSITY = "verbose"
            Write-Host "Verbose logging enabled" -ForegroundColor DarkGray
        }
        "--quiet" {
            $env:NER_VERBOSITY = "quiet" 
            Write-Host "Quiet mode enabled" -ForegroundColor DarkGray
        }
        "--no-gpu" {
            $env:CUDA_VISIBLE_DEVICES = "-1"
            Write-Host "GPU disabled, using CPU only" -ForegroundColor Yellow
        }
        "--test-gpu" {
            Write-Host "Running GPU test..." -ForegroundColor Yellow
            python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU'))); print('GPU Devices: ', tf.config.list_physical_devices('GPU'))"
            if ($LASTEXITCODE -ne 0) {
                Write-Host "GPU test failed! Please check your TensorFlow installation." -ForegroundColor Red
                Write-Host "Continuing with the main application..." -ForegroundColor Yellow
            } else {
                Write-Host "GPU test completed successfully!" -ForegroundColor Green
            }
        }
        "--jupyter" {
            Write-Host "Starting Jupyter Notebook server..." -ForegroundColor Yellow
            jupyter notebook --notebook-dir="$PSScriptRoot/notebooks"
            exit $LASTEXITCODE
        }

        # Model configuration options
        "--batch-size" {
            $BATCH_SIZE = $args[$i+1]
            $skipNext = $true
        }
        "--epochs" {
            $EPOCHS = $args[$i+1]
            $skipNext = $true
        }
        "--learning-rate" {
            $LEARNING_RATE = $args[$i+1]
            $skipNext = $true
        }
        "--model-type" {
            $MODEL_TYPE = $args[$i+1]
            $skipNext = $true
        }

        # Enhancement profiles
        "--basic" {
            $ENHANCEMENTS = "basic"
        }
        "--full" {
            $ENHANCEMENTS = "full"
        }
        "--embeddings" {
            $ENHANCEMENTS = "embeddings"
        }
        "--char-features" {
            $ENHANCEMENTS = "char_features"
        }
        "--default" {
            $ENHANCEMENTS = "default"
        }

        # Any other arguments
        default {
            $remainingArgs += $args[$i]
        }
    }
}

# Configure based on enhancement level
switch ($ENHANCEMENTS) {
    "basic" {
        Write-Host "Running with basic configuration (no enhancements)" -ForegroundColor Cyan
        $MODEL_ARGS = "--model-type $MODEL_TYPE --batch-size $BATCH_SIZE --epochs $EPOCHS --learning-rate $LEARNING_RATE --no-char-features --no-pretrained-embeddings"
    }
    "full" {
        Write-Host "Running with all enhancements (pre-trained embeddings + character features)" -ForegroundColor Cyan
        $MODEL_ARGS = "--model-type $MODEL_TYPE --batch-size $BATCH_SIZE --epochs $EPOCHS --learning-rate $LEARNING_RATE --use-char-features --use-pretrained-embeddings --embedding-dim $EMBEDDING_DIM --hidden-dim $HIDDEN_DIM"
    }
    "embeddings" {
        Write-Host "Running with pre-trained embeddings" -ForegroundColor Cyan
        $MODEL_ARGS = "--model-type $MODEL_TYPE --batch-size $BATCH_SIZE --epochs $EPOCHS --learning-rate $LEARNING_RATE --no-char-features --use-pretrained-embeddings --embedding-dim $EMBEDDING_DIM --hidden-dim $HIDDEN_DIM"
    }
    "char_features" {
        Write-Host "Running with character features" -ForegroundColor Cyan
        $MODEL_ARGS = "--model-type $MODEL_TYPE --batch-size $BATCH_SIZE --epochs $EPOCHS --learning-rate $LEARNING_RATE --use-char-features --no-pretrained-embeddings --embedding-dim $EMBEDDING_DIM --hidden-dim $HIDDEN_DIM"
    }
    default {
        Write-Host "Running with default configuration (char features, no pre-trained embeddings)" -ForegroundColor Cyan
        $MODEL_ARGS = "--model-type $MODEL_TYPE --batch-size $BATCH_SIZE --epochs $EPOCHS --learning-rate $LEARNING_RATE --use-char-features --no-pretrained-embeddings"
    }
}

# Check for required packages
try {
    # Check for TensorFlow
    python -c "import tensorflow" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: TensorFlow not found in the current Python environment" -ForegroundColor Yellow
        Write-Host "NER model training and inference may not work correctly" -ForegroundColor Yellow
        Write-Host "Hint: Run setup.ps1 to install required packages" -ForegroundColor Yellow
    }
} catch {
    # Silently continue if the import check fails
}

# Show configuration
Write-Host "Running NER model with the following configuration:" -ForegroundColor Green
Write-Host "Model Type: $MODEL_TYPE"
Write-Host "Batch Size: $BATCH_SIZE"
Write-Host "Epochs: $EPOCHS"
Write-Host "Learning Rate: $LEARNING_RATE" 
Write-Host "Enhancement Profile: $ENHANCEMENTS"
Write-Host ""
Write-Host "Starting training..." -ForegroundColor Green

# Run the model
if ($remainingArgs.Count -gt 0) {
    # If there are remaining args, pass them along
    Write-Host "Running: python -m src.main $MODEL_ARGS $remainingArgs" -ForegroundColor DarkGray
    $argsList = $MODEL_ARGS.Split(" ") + $remainingArgs
    python -m src.main $argsList
} else {
    # Otherwise just use the model args
    Write-Host "Running: python -m src.main $MODEL_ARGS" -ForegroundColor DarkGray
    python -m src.main $MODEL_ARGS.Split(" ")
}

# Check for errors
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "An error occurred while running the application." -ForegroundColor Red
    exit $LASTEXITCODE
} else {
    Write-Host ""
    Write-Host "Training completed successfully." -ForegroundColor Green
}
