**Project Overview**

# Project Documentation

## Overview
This project is a data analysis and profiling tool built with Python 3.9. It's designed to analyze and profile various datasets, particularly focusing on data provided by Behavio: credits, payments, and Atlas Cechu data.

## Project Structure

### Root Directory
- `data_profiling.py`: Main script for data profiling functionality
- `requirements.txt`: Project dependencies
- `README.md`: Basic project information
- `.gitignore`: Git ignore rules
- `.DS_Store`: macOS system file (can be ignored)

### Key Directories
1. `data/`: Contains the input datasets
   - `Atlas Cechu Student Access.csv` (93MB): Atlas Cechu student access data
   - `User Credits Student Access.csv` (2.2MB): Student credits information
   - `Payments Student Access.csv` (2.9MB): Student payment records

2. `profiling_output/`: Directory where generated profiling reports using ydata_profiling package are stored

3. `data_output/`: Directory for processed data outputs

4. `sandbox/`: Contains work-in-progress (WIP) and testing code
   - `data_profiling.ipynb`: Data profiling script
   - `data-prep&modeling.ipynb`: Data prep and machine learning model training code

## Dependencies
The project uses the following key packages:
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `ydata-profiling`: Data profiling and analysis
- `catboost`: Machine learning library
- `scikit-learn`: Machine learning tools
- `plotly`: Interactive data visualization
- `ipywidgets`: Interactive widgets for Jupyter notebooks
- `openpyxl`: Excel file handling

## Core Components

### Data Profiling (`data_profiling.py`)
The `DataProfiler` class provides functionality to:
- Load CSV data files
- Generate comprehensive profiling reports using ydata-profiling
- Save reports to the profiling_output directory
- Display reports interactively

## Usage

### Data Profiling
```python
from data_profiling import DataProfiler

# Create a profiler instance
profiler = DataProfiler('your_data_file.csv')

# Generate and save the report
profiler.save_report()

# Display the report
profiler.show_report()
```

## Notes
- The project requires Python 3.9 due to compatibility requirements with ydata-profiling
- All experimental code should be placed in the sandbox directory
- Generated reports are stored in the profiling_output directory
- Processed data outputs are saved in the data_output directory

## Best Practices
1. Always work with experimental code in the sandbox directory
2. Keep the data directory organized and maintain data versioning
3. Document any changes to the data processing pipeline
4. Use the virtual environment (dp_env or venv as they are already in gitignore. Or use conda env) for development
5. Follow the existing project structure for consistency