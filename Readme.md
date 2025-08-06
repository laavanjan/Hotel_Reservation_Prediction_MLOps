# Hotel Reservation Prediction - MLOps Project

A machine learning project for predicting hotel reservation outcomes using MLOps best practices. This project implements a complete ML pipeline with proper logging, configuration management, and modular code structure.

## Project Overview

This project predicts hotel reservation patterns using machine learning techniques. It's structured as an MLOps pipeline with proper data ingestion, preprocessing, model training, and deployment capabilities.

## Project Structure

```
Hotel_Reservation_Prediction/
├── artifacts/
│   └── raw/                    # Raw data storage
├── config/
│   ├── __init__.py
│   ├── config.yaml            # Configuration parameters
│   └── paths_config.py        # File paths configuration
├── logs/                      # Application logs
├── notebook/
│   ├── notebook.ipynb         # Jupyter notebook for exploration
│   ├── random_forest.pkl      # Trained model
│   └── train.csv             # Training data
├── src/
│   ├── __init__.py
│   ├── custom_exception.py    # Custom exception handling
│   ├── data_ingestion.py      # Data ingestion pipeline
│   └── logger.py             # Logging configuration
├── static/                    # Static files for web interface
├── templates/                 # HTML templates
├── utils/
│   ├── __init__.py
│   └── common_functions.py    # Utility functions
├── requirements.txt           # Python dependencies
├── setup.py                  # Package setup
└── test.py                   # Test file
```

## Key Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for different functionalities
- **Comprehensive Logging**: Centralized logging system using [`get_logger`](src/logger.py) from [src/logger.py](src/logger.py)
- **Configuration Management**: YAML-based configuration with [`read_yaml`](utils/common_functions.py) function
- **Custom Exception Handling**: Robust error handling with custom exceptions
- **Data Pipeline**: Automated data ingestion and preprocessing using [`DataIngestion`](src/data_ingestion.py) class
- **Google Cloud Integration**: Support for Google Cloud Storage for data management

## Core Components

### Data Ingestion
The [`DataIngestion`](src/data_ingestion.py) class in [src/data_ingestion.py](src/data_ingestion.py) handles:
- Loading data from various sources including Google Cloud Storage
- Train-test split using scikit-learn
- Data validation and preprocessing

### Configuration Management
- [config/paths_config.py](config/paths_config.py) defines all file paths including:
  - `RAW_DIR`: Raw data directory
  - `TRAIN_FILE_PATH`: Training data path
  - `TEST_FILE_PATH`: Test data path
  - `CONFIG_PATH`: Configuration file path

### Logging System
The [src/logger.py](src/logger.py) module provides:
- Daily log file rotation
- Structured logging with timestamps and levels
- Centralized logger configuration

### Utility Functions
[utils/common_functions.py](utils/common_functions.py) contains helper functions:
- [`read_yaml`](utils/common_functions.py): Safe YAML file reading with error handling

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Hotel_Reservation_Prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### Running Data Ingestion
```python
from src.data_ingestion import DataIngestion
from utils.common_functions import read_yaml
from config.paths_config import CONFIG_PATH

# Initialize and run data ingestion
data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
data_ingestion.run()
```

### Configuration
Update [config/config.yaml](config/config.yaml) with your specific parameters and paths.

## Requirements

The project dependencies are listed in [requirements.txt](requirements.txt). Key requirements include:
- pandas for data manipulation
- scikit-learn for machine learning
- google-cloud-storage for cloud integration
- PyYAML for configuration management

## Logging

Logs are automatically generated in the [logs/](logs/) directory with daily rotation. Each log file follows the format `log_YYYY-MM-DD.log`.

## Development

For development and testing, you can use the [notebook/notebook.ipynb](notebook/notebook.ipynb) for data exploration and model experimentation.

## Model

The trained Random Forest model is saved as [notebook/random_forest.pkl](notebook/random_forest.pkl).

## Testing

Run tests using:
```bash
python test.py
```

## Contributing

1. Follow the existing code structure and naming conventions
2. Add appropriate logging using the [`get_logger`](src/logger.py) function
3. Handle exceptions using the custom exception system
4. Update configuration files as needed

## License

[Add your license information here]

## Contact

[Add your contact information here]