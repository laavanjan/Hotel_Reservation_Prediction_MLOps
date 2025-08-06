# Hotel Reservation Prediction - MLOps Project

A machine learning project for predicting hotel reservation outcomes using MLOps best practices. This project implements a complete ML pipeline with proper logging, configuration management, data preprocessing, feature selection, and model training capabilities.

## Project Overview

This project predicts hotel reservation patterns (Canceled vs Not_Canceled) using machine learning techniques. It's structured as an MLOps pipeline with comprehensive data ingestion, preprocessing, feature engineering, model training, and evaluation capabilities.

## Project Structure

```
Hotel_Reservation_Prediction/
├── artifacts/
│   ├── raw/                       # Raw data storage
│   └── processed/                 # Processed data storage
├── config/
│   ├── __init__.py
│   ├── config.yaml               # Configuration parameters
│   └── paths_config.py           # File paths configuration
├── logs/                         # Application logs (daily rotation)
├── mlruns/                       # MLflow experiment tracking (gitignored)
├── mlartifacts/                  # MLflow artifacts (gitignored)
├── notebook/
│   ├── notebook.ipynb            # Complete EDA and model development
│   ├── random_forest.pkl         # Trained Random Forest model
│   └── train.csv                 # Training data
├── src/
│   ├── __init__.py
│   ├── custom_exception.py       # Custom exception handling
│   ├── data_ingestion.py         # Data ingestion pipeline
│   ├── data_preprocessing.py     # Data preprocessing pipeline
│   └── logger.py                 # Logging configuration
├── static/                       # Static files for web interface
├── templates/                    # HTML templates
├── utils/
│   ├── __init__.py
│   └── common_functions.py       # Utility functions
├── venv/                         # Virtual environment (gitignored)
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
├── setup.py                     # Package setup
└── test.py                      # Test file
```

## Key Features

- **Complete ML Pipeline**: End-to-end pipeline from data ingestion to model deployment
- **Data Preprocessing**: Comprehensive preprocessing including:
  - Label encoding for categorical variables
  - Skewness handling with log transformation
  - Data balancing using SMOTE
  - Feature selection using Random Forest importance
- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Comprehensive Logging**: Centralized logging system with daily rotation
- **Configuration Management**: YAML-based configuration for easy parameter tuning
- **Custom Exception Handling**: Robust error handling with detailed error messages
- **MLflow Integration**: Experiment tracking and model versioning support
- **Version Control**: Proper .gitignore for ML projects

## Core Components

### Data Ingestion
The [`DataIngestion`](src/data_ingestion.py) class handles:
- Loading data from various sources including Google Cloud Storage
- Train-test split using scikit-learn
- Data validation and storage in artifacts directory

### Data Preprocessing
The [`DataPreprocessor`](src/data_preprocessing.py) class provides:
- **Label Encoding**: Converts categorical variables to numerical
- **Skewness Handling**: Applies log transformation for highly skewed features
- **Data Balancing**: Uses SMOTE to handle class imbalance
- **Feature Selection**: Selects top features using Random Forest importance
- **Data Validation**: Ensures data quality throughout the pipeline

### Configuration Management
- [config/paths_config.py](config/paths_config.py) defines all file paths
- [config/config.yaml](config/config.yaml) contains:
  - Categorical and numerical column lists
  - Skewness threshold for transformation
  - Number of features to select
  - Processing parameters

### Logging System
The [src/logger.py](src/logger.py) module provides:
- Daily log file rotation (`log_YYYY-MM-DD.log`)
- Structured logging with timestamps and levels
- Error tracking and debugging capabilities

### Utility Functions
[utils/common_functions.py](utils/common_functions.py) contains:
- [`read_yaml`](utils/common_functions.py): Safe YAML file reading
- [`load_data`](utils/common_functions.py): CSV data loading with validation

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MLOPS-PROJECT-1
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### Running the Complete Pipeline

#### Data Ingestion
```python
from src.data_ingestion import DataIngestion
from utils.common_functions import read_yaml
from config.paths_config import CONFIG_PATH

# Initialize and run data ingestion
data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
data_ingestion.run()
```

#### Data Preprocessing
```python
from src.data_preprocessing import DataPreprocessor
from config.paths_config import TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH

# Initialize and run preprocessing
processor = DataPreprocessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
processor.process()
```

### Configuration

Update [config/config.yaml](config/config.yaml) with your parameters:

```yaml
data_processing:
  categorical_columns:
    - type_of_meal_plan
    - required_car_parking_space
    - room_type_reserved
    - market_segment_type
    - repeated_guest
    - booking_status
  numerical_columns:
    - no_of_adults
    - no_of_children
    - no_of_weekend_nights
    - no_of_week_nights
    - lead_time
    - arrival_year
    - arrival_month
    - arrival_date
    - no_of_previous_cancellations
    - avg_price_per_room
    - no_of_previous_bookings_not_canceled
    - no_of_special_requests
  skewness_threshold: 5
  num_features: 10
```

## Data Processing Pipeline

1. **Data Loading**: Load train and test datasets
2. **Data Cleaning**: Remove duplicates and unnecessary columns
3. **Label Encoding**: Convert categorical variables to numerical
4. **Skewness Handling**: Apply log transformation for highly skewed features
5. **Data Balancing**: Use SMOTE to balance the target variable
6. **Feature Selection**: Select top 10 features based on importance
7. **Data Saving**: Save processed data for model training

## Model Development

The project includes comprehensive model evaluation with multiple algorithms:
- Random Forest (Best performer)
- XGBoost
- LightGBM
- Logistic Regression
- SVM
- Decision Tree
- AdaBoost
- KNN
- Naive Bayes

Model evaluation metrics include:
- Accuracy
- Precision
- Recall
- F1 Score

## Requirements

Key dependencies include:
- `pandas`: Data manipulation
- `scikit-learn`: Machine learning algorithms
- `imbalanced-learn`: SMOTE for data balancing
- `xgboost`, `lightgbm`: Gradient boosting algorithms
- `PyYAML`: Configuration management
- `google-cloud-storage`: Cloud integration
- `mlflow`: Experiment tracking

## Logging

- Logs are automatically generated in the `logs/` directory
- Daily rotation with format: `log_YYYY-MM-DD.log`
- Comprehensive error tracking and pipeline monitoring

## Git Configuration

The project includes a comprehensive `.gitignore` that excludes:
- Python cache files (`__pycache__/`, `*.pyc`)
- Data files (`*.csv`, `*.pkl`)
- Virtual environments (`venv/`)
- Logs (`logs/`, `*.log`)
- MLflow artifacts (`mlruns/`, `mlartifacts/`)
- Jupyter checkpoints

## Development

- Use [notebook/notebook.ipynb](notebook/notebook.ipynb) for EDA and experimentation
- Follow modular architecture for new components
- Add comprehensive logging for debugging
- Update configuration files for new parameters

## Testing

Run tests using:
```bash
python test.py
```

## Contributing

1. Follow the existing modular code structure
2. Use the centralized logging system with `get_logger(__name__)`
3. Handle exceptions using the `CustomException` class with `sys.exc_info()`
4. Update configuration files in `config/` for new parameters
5. Add appropriate documentation and type hints
6. Ensure proper error handling and logging

## Troubleshooting

### Common Issues:

1. **YAML Configuration Errors**: Ensure proper YAML syntax in `config.yaml`
2. **Column Name Mismatches**: Verify column names match between code and data
3. **CustomException Errors**: Always pass `sys.exc_info()` to CustomException
4. **Missing Dependencies**: Install all requirements from `requirements.txt`
5. **Path Issues**: Use paths defined in `config/paths_config.py`

## License

[Add your license information here]

## Contact

[Add your contact information here]