# Installation Guide for ELF

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation Methods

### Method 1: Using pip (Recommended)

1. **Clone the repository**
```bash
git clone https://github.com/Wijaya11/ELF.git
cd ELF
```

2. **Create a virtual environment (optional but recommended)**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open your browser**
The app should automatically open at `http://localhost:8501`

### Method 2: Using conda

1. **Clone the repository**
```bash
git clone https://github.com/Wijaya11/ELF.git
cd ELF
```

2. **Create conda environment**
```bash
conda create -n elf python=3.10
conda activate elf
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

## Testing the Installation

After installing dependencies, you can test the core functionality without the UI:

```bash
python example.py
```

This will run a complete forecasting workflow and display the results in the terminal.

## Troubleshooting

### Issue: Dependencies fail to install

**Solution 1:** Try installing packages individually
```bash
pip install streamlit
pip install pandas numpy
pip install scikit-learn
pip install xgboost
pip install tensorflow
pip install matplotlib seaborn plotly
```

**Solution 2:** Update pip
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Issue: TensorFlow installation fails

**Solution:** TensorFlow can be tricky. Try:
```bash
# For CPU version
pip install tensorflow-cpu

# Or use a compatible version
pip install tensorflow==2.13.0
```

### Issue: Streamlit doesn't open in browser

**Solution:** 
1. Check the terminal for the URL (usually http://localhost:8501)
2. Manually open the URL in your browser
3. Try a different port:
```bash
streamlit run app.py --server.port 8502
```

### Issue: LSTM model training is slow

**Solution:** 
1. Reduce the number of epochs in the sidebar (default is 30)
2. Reduce the amount of data generated
3. Consider using a GPU-enabled version of TensorFlow if available

## System Requirements

### Minimum
- RAM: 4GB
- CPU: 2 cores
- Disk: 2GB free space

### Recommended
- RAM: 8GB or more
- CPU: 4 cores or more
- Disk: 5GB free space
- GPU: Optional, but speeds up LSTM training

## Next Steps

After successful installation:
1. Read the README.md for feature overview
2. Run `streamlit run app.py` to start the interactive UI
3. Try `python example.py` to test the core functionality
4. Explore the code in `models.py` and `data_utils.py` to understand the implementation

## Support

If you encounter issues not covered here, please:
1. Check the GitHub Issues page
2. Create a new issue with details about your environment and error messages
3. Include Python version (`python --version`) and OS information
