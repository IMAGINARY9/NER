#!/bin/bash
# Setup script for Entity Extraction Project

# Print banner
echo "======================================================="
echo "  Entity Extraction Project - Setup Script              "
echo "======================================================="

# Check if script was called with special flags
CPU_ONLY=false
SKIP_GPU_CHECK=false

# Parse arguments
for arg in "$@"
do
    case $arg in
        --cpu-only)
        CPU_ONLY=true
        shift
        ;;
        --skip-gpu-check)
        SKIP_GPU_CHECK=true
        shift
        ;;
        --force)
        FORCE=true
        shift
        ;;
    esac
done

# Check if Python is installed
if command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
    echo "✅ Python 3 found"
elif command -v python &>/dev/null; then
    PYTHON_CMD=python
    echo "✅ Python found"
else
    echo "❌ Python not found. Please install Python 3.8 or newer."
    echo "You can download Python from: https://www.python.org/downloads/"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))")
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "✅ Using Python $PYTHON_VERSION"

if [ $PYTHON_MAJOR -lt 3 ] || ([ $PYTHON_MAJOR -eq 3 ] && [ $PYTHON_MINOR -lt 8 ]); then
    echo "⚠️ Warning: Python 3.8+ is recommended (found $PYTHON_VERSION)"
    
    if [ "$FORCE" != true ]; then
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Create Python virtual environment
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    $PYTHON_CMD -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create Python virtual environment. Please ensure Python 3.8+ is installed."
        exit 1
    fi
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate virtual environment."
    exit 1
fi
echo "✅ Virtual environment activated"

# Upgrade pip and essential tools
echo "Upgrading pip and installing key tools..."
pip install --upgrade pip setuptools wheel

# Install dependencies based on selected options
echo "Installing dependencies..."

if [ "$CPU_ONLY" = true ]; then
    # Explicitly disable GPU
    echo "Installing for CPU-only operation..."
    export CUDA_VISIBLE_DEVICES="-1"
    pip install -r requirements.txt
else
    # Standard installation
    pip install -r requirements.txt
fi

if [ $? -ne 0 ]; then
    echo "❌ Failed to install required packages. Please check requirements.txt and try again."
    exit 1
fi
echo "✅ Dependencies installed successfully"

# Install Jupyter kernel for the virtual environment
echo "Installing Jupyter kernel for this environment..."
pip install ipykernel
python -m ipykernel install --user --name=ner-env --display-name="Python (Entity Extraction)"
echo "✅ Jupyter kernel installed"

# Create necessary directories
mkdir -p data logs/fit models/checkpoints notebooks

# Download dataset if it doesn't exist
if [ ! -f "data/ner.csv" ]; then
    echo "NER dataset not found in data directory."
    echo "You'll need to download the dataset from Kaggle manually:"
    echo "1. Visit: https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus"
    echo "2. Download the 'ner.csv' file"
    echo "3. Place it in the 'data' directory of this project"
else
    echo "✅ NER dataset found in data directory"
fi

# Check for TensorFlow and GPU availability (unless skipped)
if [ "$SKIP_GPU_CHECK" != true ]; then
    echo "Checking TensorFlow and GPU availability..."
    
    # Create a temporary Python script for checking GPU
    cat > gpu_check.py << 'EOL'
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
                print(f"  {gpu}")
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
EOL

    # Execute the script
    $PYTHON_CMD gpu_check.py
    rm gpu_check.py
fi

# Create an activation script
cat > activate.sh << 'EOL'
#!/bin/bash
# This script activates the virtual environment

echo "Activating Entity Extraction environment..."
source venv/bin/activate

echo
echo "Environment ready! You can now run:"
echo "  - ./run.sh                     (Run the main application)"
echo "  - python src/main.py           (Run the main script directly)"
echo "  - jupyter notebook             (Start Jupyter notebook server)"
echo
export PYTHONPATH="$PWD:$PYTHONPATH"
EOL

# Make activation script executable
chmod +x activate.sh
echo "✅ Created activate.sh for easy environment activation"

# Print summary
echo
echo "======================================================="
echo " Setup Complete! Environment is ready!"
echo "======================================================="
echo
echo "Quick Start Guide:"
echo "1. Activate the environment:"
echo "   source venv/bin/activate"
echo "   or"
echo "   source ./activate.sh"
echo
echo "2. Run the application:"
echo "   python src/main.py --data-path data/ner.csv --model-type bilstm-crf"
echo
echo "3. For Jupyter notebook:"
echo "   jupyter notebook notebooks/ner_training_example.ipynb"
echo
if [ "$CPU_ONLY" = true ]; then
    echo "4. Using CPU-only mode"
    echo "   You've configured the environment for CPU-only operation"
else
    echo "4. For GPU/CPU options:"
    echo "   - Force CPU mode: ./run.sh --no-gpu"
fi
echo
echo "To deactivate the virtual environment when finished:"
echo "    deactivate"
