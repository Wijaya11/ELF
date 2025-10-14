#!/bin/bash

# ELF Setup Script
# Quick setup for the Electrical Load Forecasting tool

set -e  # Exit on error

echo "=========================================="
echo "ELF - Electrical Load Forecasting Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Check if Python 3.8+
if ! python3 -c "import sys; assert sys.version_info >= (3, 8)" 2>/dev/null; then
    echo "Error: Python 3.8 or higher is required"
    exit 1
fi

echo "✓ Python version is compatible"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip -q
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies..."
echo "This may take 5-10 minutes on first install..."
pip install -r requirements.txt
echo "✓ All dependencies installed"
echo ""

# Verify installation
echo "Verifying installation..."
python3 test_imports.py
echo ""

# Success message
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "To run the application:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run: streamlit run app.py"
echo ""
echo "Or test the core functionality:"
echo "  python example.py"
echo ""
echo "For more information, see:"
echo "  - QUICKSTART.md for quick start"
echo "  - USER_GUIDE.md for detailed usage"
echo "  - INSTALL.md for troubleshooting"
echo ""
