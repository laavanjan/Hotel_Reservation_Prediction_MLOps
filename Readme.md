# Hotel Reservation Prediction - MLOps Project

A machine learning project for predicting hotel reservation outcomes using MLOps best practices. This project implements a complete ML pipeline with proper logging, configuration management, data preprocessing, feature selection, model training, and automated pipeline execution capabilities.

## Project Overview

This project predicts hotel reservation patterns (Canceled vs Not_Canceled) using machine learning techniques. It's structured as an MLOps pipeline with comprehensive data ingestion, preprocessing, feature engineering, model training, and evaluation capabilities with automated pipeline execution.

## Project Structure

```
Hotel_Reservation_Prediction/
├── artifacts/
│   ├── raw/                       # Raw data storage
│   ├── processed/                 # Processed data storage
│   └── models/                    # Trained model storage
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
├── pipeline/
│   └── training_pipeline.py      # Complete automated training pipeline
├── src/
│   ├── __init__.py
│   ├── custom_exception.py       # Custom exception handling
│   ├── data_ingestion.py         # Data ingestion pipeline
│   ├── data_preprocessing.py     # Data preprocessing pipeline
│   ├── model_training.py         # Model training with MLflow
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

- **Complete Automated ML Pipeline**: End-to-end pipeline execution with single command
- **Data Preprocessing**: Comprehensive preprocessing including:
  - Label encoding for categorical variables
  - Skewness handling with log transformation
  - Data balancing using SMOTE
  - Feature selection using Random Forest importance
- **Model Training**: LightGBM with hyperparameter optimization and MLflow tracking
- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Comprehensive Logging**: Centralized logging system with daily rotation
- **Configuration Management**: YAML-based configuration for easy parameter tuning
- **Custom Exception Handling**: Robust error handling with detailed error messages
- **MLflow Integration**: Experiment tracking and model versioning support
- **Version Control**: Proper .gitignore for ML projects
- **Automated Pipeline Execution**: Single script to run entire pipeline

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

### Model Training
The [`ModelTraining`](src/model_training.py) class provides:
- **LightGBM Implementation**: Advanced gradient boosting algorithm
- **Hyperparameter Optimization**: RandomizedSearchCV for optimal parameter selection
- **MLflow Integration**: Complete experiment tracking and model versioning
- **Comprehensive Evaluation**: Multiple performance metrics (accuracy, precision, recall, F1-score)
- **Automated Model Persistence**: Saves trained models with joblib

### Training Pipeline
The [`training_pipeline.py`](pipeline/training_pipeline.py) provides:
- **Automated Pipeline Execution**: Single script to run entire ML pipeline
- **Sequential Processing**: Runs data ingestion → preprocessing → model training
- **Error Handling**: Comprehensive error management across all pipeline stages
- **Logging Integration**: Complete pipeline monitoring and debugging

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

### Option 1: Automated Complete Pipeline (Recommended)

Run the entire ML pipeline with a single command:

```bash
python pipeline/training_pipeline.py
```

This will automatically execute:
1. **Data Ingestion**: Load and validate raw data
2. **Data Preprocessing**: Clean, encode, balance, and select features
3. **Model Training**: Train LightGBM model with hyperparameter tuning and MLflow tracking

### Option 2: Individual Component Execution

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

#### Model Training
```python
from src.model_training import ModelTraining
from config.paths_config import PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH

# Initialize and run model training
trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH)
trainer.run()
```

### Option 3: Individual Module Execution

```bash
# Run individual components
python src/data_preprocessing.py
python src/model_training.py
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

## Model Training Pipeline

### Algorithm: LightGBM Classifier
**Why LightGBM?**
- Fast training speed and high efficiency
- Lower memory usage compared to other boosting algorithms
- Better accuracy than traditional algorithms
- Handles categorical features automatically
- Supports parallel and GPU learning

### Training Process:
1. **Data Loading**: Loads processed train/test datasets
2. **Data Splitting**: Separates features (X) and target variable (y)
3. **Model Training**: LightGBM with hyperparameter tuning via RandomizedSearchCV
4. **Model Evaluation**: Calculates accuracy, precision, recall, and F1-score
5. **Model Saving**: Persists trained model for deployment
6. **MLflow Logging**: Tracks experiments, parameters, metrics, and artifacts

### MLflow Integration
The model training automatically logs:
- **Datasets**: Training and testing data files
- **Model**: Trained LightGBM model
- **Parameters**: All model hyperparameters
- **Metrics**: Performance evaluation metrics

Access MLflow UI:
```bash
mlflow ui
# Access at: http://localhost:5000
```

## Model Development

The project includes comprehensive model evaluation with multiple algorithms:
- **LightGBM** (Current implementation)
- Random Forest (Available in notebook)
- XGBoost (Available in notebook)
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
- `lightgbm`: LightGBM classifier
- `imbalanced-learn`: SMOTE for data balancing
- `mlflow`: Experiment tracking
- `PyYAML`: Configuration management
- `google-cloud-storage`: Cloud integration
- `joblib`: Model persistence

## Logging

- Logs are automatically generated in the `logs/` directory
- Daily rotation with format: `log_YYYY-MM-DD.log`
- Comprehensive error tracking and pipeline monitoring
- All pipeline stages are logged for debugging and monitoring

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
- Use the automated pipeline for consistent execution

## Testing

Run tests using:
```bash
python test.py
```

## Pipeline Execution Examples

### Quick Start
```bash
# Run complete pipeline
python pipeline/training_pipeline.py
```

### Development Mode
```python
# Step-by-step execution for debugging
python src/data_ingestion.py      # Step 1
python src/data_preprocessing.py  # Step 2
python src/model_training.py      # Step 3
```

### Monitoring
```bash
# Check logs
tail -f logs/log_$(date +%Y-%m-%d).log

# View MLflow experiments
mlflow ui
```

## Contributing

1. Follow the existing modular code structure
2. Use the centralized logging system with `get_logger(__name__)`
3. Handle exceptions using the `CustomException` class with `sys.exc_info()`
4. Update configuration files in `config/` for new parameters
5. Add appropriate documentation and type hints
6. Ensure proper error handling and logging
7. Test changes with the automated pipeline

## Troubleshooting

### Common Issues:

1. **YAML Configuration Errors**: Ensure proper YAML syntax in `config.yaml`
2. **Column Name Mismatches**: Verify column names match between code and data
3. **CustomException Errors**: Always pass `sys.exc_info()` to CustomException
4. **Missing Dependencies**: Install all requirements from `requirements.txt`
5. **Path Issues**: Use paths defined in `config/paths_config.py`
6. **Pipeline Execution Errors**: Check individual component logs for detailed error information
7. **MLflow Issues**: Ensure MLflow server is accessible and artifacts directory has proper permissions

### Debugging Pipeline Issues
```bash
# Check if all paths exist
python -c "from config.paths_config import *; print('All paths configured')"

# Validate configuration
python -c "from utils.common_functions import read_yaml; from config.paths_config import CONFIG_PATH; print(read_yaml(CONFIG_PATH))"

# Run pipeline with verbose logging
python pipeline/training_pipeline.py 2>&1 | tee pipeline_execution.log
```

## License

[Add your license information here]

## Contact

[Add your contact information here]

---

**Built with ❤️ for robust, scalable, and maintainable ML pipelines**