# 🏨 Hotel Reservation Prediction - MLOps Project

![Deployment Status](https://img.shields.io/badge/Deployment-Live-brightgreen)
![Platform](https://img.shields.io/badge/Platform-Google%20Cloud%20Run-blue)
![Model](https://img.shields.io/badge/Model-LightGBM-orange)
![Pipeline](https://img.shields.io/badge/Pipeline-Jenkins-red)
![Python](https://img.shields.io/badge/Python-3.9+-blue)

**🔗 Live Application**: https://ml-project-929445478726.us-central1.run.app

A complete MLOps pipeline for predicting hotel reservation cancellations using machine learning, containerization, and automated CI/CD deployment on Google Cloud Run.

## 🚀 Deployment Status


**Current Status**: Successfully deployed on Google Cloud Run
- **Last Updated**: August 9, 2025
- **Pipeline Status**: ✅ Passing
- **Model Version**: LightGBM with 5-fold CV
- **Container**: `gcr.io/solid-drive-467918-h9/ml-project:latest`
- **Build Time**: ~94 minutes (including model training)
- **Region**: us-central1

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Key Features](#-key-features)
- [Core Components](#-core-components)
- [Project Overview](#-project-overview)
- [Model Details](#-model-details)
- [Infrastructure](#️-infrastructure)
- [API Usage](#-api-usage)
- [Local Development](#-local-development)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Monitoring & Maintenance](#-monitoring--maintenance)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Troubleshooting](#-troubleshooting)

## 🎯 Quick Start

### Option 1: Use Live Application
Visit the deployed application: **https://ml-project-929445478726.us-central1.run.app**

1. Open the web interface
2. Fill in the hotel reservation details
3. Click "Predict" to get cancellation probability
4. View results with confidence scores

### Option 2: Local Development
```bash
git clone https://github.com/laavanjan/MLOPS-PROJECT-1.git
cd MLOPS-PROJECT-1
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
python app.py
```

### Option 3: Run Complete ML Pipeline
```bash
# Run entire automated pipeline
python training_pipeline.py

# Or run individual components
python src/data_ingestion.py
python src/data_preprocessing.py
python src/model_training.py
```

## ✨ Key Features

### 🔄 Complete Automated ML Pipeline
- **End-to-end pipeline execution** with single command
- **Sequential processing**: Data ingestion → Preprocessing → Model training → Deployment
- **Automated model persistence** and versioning
- **Error handling** across all pipeline stages

### 📊 Data Preprocessing
Comprehensive preprocessing pipeline including:
- **Label encoding** for categorical variables
- **Skewness handling** with log transformation
- **Data balancing** using SMOTE (Synthetic Minority Over-sampling Technique)
- **Feature selection** using Random Forest importance
- **Data validation** and quality checks

### 🤖 Model Training
- **LightGBM** with hyperparameter optimization
- **MLflow tracking** for experiment management
- **5-fold Cross Validation** for robust model evaluation
- **RandomizedSearchCV** for optimal parameter selection
- **Comprehensive evaluation** with multiple metrics

### 🏗️ Modular Architecture
- **Clean separation of concerns** with dedicated modules
- **Configuration management** via YAML files
- **Reusable components** for different ML tasks
- **Scalable design** for easy feature additions

### 📝 Comprehensive Logging
- **Centralized logging system** with daily rotation
- **Structured logging** with timestamps and levels
- **Error tracking** and debugging capabilities
- **Pipeline monitoring** throughout execution

### ⚙️ Configuration Management
- **YAML-based configuration** for easy parameter tuning
- **Centralized path management** via `config/paths_config.py`
- **Environment-specific settings** support
- **Dynamic parameter adjustment** without code changes

### 🛡️ Custom Exception Handling
- **Robust error handling** with detailed error messages
- **Custom exception classes** for different error types
- **Graceful failure recovery** mechanisms
- **Comprehensive error logging** for debugging

### 📈 MLflow Integration
- **Experiment tracking** and comparison
- **Model versioning** and registry
- **Parameter logging** and artifact storage
- **Performance metrics** tracking over time

### 🔧 Version Control
- **Proper .gitignore** for ML projects
- **Clean repository structure** without artifacts
- **Dependency management** with requirements.txt
- **Code versioning** with meaningful commits

## 🔧 Core Components

### 📥 Data Ingestion
The `DataIngestion` class handles:

```python
# Key capabilities
- Loading data from various sources (CSV, GCS, databases)
- Train-test split using scikit-learn
- Data validation and quality checks
- Storage in organized artifacts directory
- Support for multiple data formats
```

**Features:**
- **Multi-source support**: Local files, Google Cloud Storage, databases
- **Automated splitting**: Configurable train-test-validation splits
- **Data validation**: Schema validation and data quality checks
- **Artifact management**: Organized storage in `artifacts/` directory

### 🔄 Data Preprocessing
The `DataPreprocessor` class provides:

```python
# Core preprocessing capabilities
- Label Encoding: Converts categorical variables to numerical
- Skewness Handling: Log transformation for highly skewed features
- Data Balancing: SMOTE to handle class imbalance
- Feature Selection: Random Forest importance-based selection
- Data Validation: Ensures data quality throughout pipeline
```

**Advanced Features:**
- **Categorical Encoding**: Multiple encoding strategies (Label, One-Hot, Target)
- **Skewness Correction**: Automatic detection and transformation of skewed features
- **Class Balancing**: SMOTE, ADASYN, and other resampling techniques
- **Feature Engineering**: Automated feature creation and selection
- **Pipeline Serialization**: Save/load preprocessing pipelines

### 🎯 Model Training
The `ModelTraining` class provides:

```python
# Training capabilities
- LightGBM Implementation: Advanced gradient boosting
- Hyperparameter Optimization: RandomizedSearchCV and Optuna
- MLflow Integration: Complete experiment tracking
- Comprehensive Evaluation: Multiple performance metrics
- Automated Model Persistence: joblib and MLflow model saving
```

**Key Features:**
- **Algorithm Support**: LightGBM, XGBoost, Random Forest, SVM
- **Hyperparameter Tuning**: Grid Search, Random Search, Bayesian Optimization
- **Cross-Validation**: K-fold, Stratified K-fold, Time Series splits
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Model Comparison**: Automated model selection and comparison

### 🚀 Training Pipeline
The `training_pipeline.py` provides:

```python
# Pipeline orchestration
- Automated Pipeline Execution: Single script for entire ML workflow
- Sequential Processing: Organized execution flow
- Error Handling: Comprehensive error management
- Logging Integration: Complete pipeline monitoring
- Configuration Loading: YAML-based parameter management
```

**Pipeline Flow:**
1. **Configuration Loading**: Read YAML configs and validate parameters
2. **Data Ingestion**: Load and split data from configured sources
3. **Data Preprocessing**: Apply transformations and feature engineering
4. **Model Training**: Train and optimize ML models
5. **Model Evaluation**: Comprehensive performance assessment
6. **Model Deployment**: Save models and artifacts for production

### ⚙️ Configuration Management

#### `config/paths_config.py`
```python
# Centralized path management
ARTIFACTS_DIR = "artifacts"
DATA_DIR = "data"
MODELS_DIR = "models"
LOGS_DIR = "logs"
CONFIG_DIR = "config"
```

#### `config/config.yaml`
```yaml
# Model configuration
categorical_columns: ["meal", "market_segment", "deposit_type", "customer_type"]
numerical_columns: ["lead_time", "arrival_date_week_number", "adr"]
skewness_threshold: 0.5
n_features_to_select: 15
test_size: 0.2
random_state: 42

# Model parameters
lightgbm_params:
  n_estimators: [100, 200, 300]
  max_depth: [3, 5, 7]
  learning_rate: [0.01, 0.1, 0.2]
  min_child_samples: [10, 20, 30]
```

### 📝 Logging System
The `src/logger.py` module provides:

```python
# Logging features
- Daily log file rotation (log_YYYY-MM-DD.log)
- Structured logging with timestamps and levels
- Error tracking and debugging capabilities
- Custom formatters for different log types
- Integration with MLflow for experiment logging
```

**Log Structure:**
```
logs/
├── log_2025-08-09.log
├── error_2025-08-09.log
└── training_2025-08-09.log
```

### 🛠️ Utility Functions
`utils/common_functions.py` contains:

```python
def read_yaml(file_path: str) -> dict:
    """Safe YAML file reading with error handling"""
    
def load_data(file_path: str) -> pd.DataFrame:
    """CSV data loading with validation"""
    
def save_object(obj: object, file_path: str) -> None:
    """Generic object serialization"""
    
def load_object(file_path: str) -> object:
    """Generic object deserialization"""
    
def create_directories(dirs: list) -> None:
    """Create directory structure for artifacts"""
```

## 📊 Project Overview

This project implements an end-to-end MLOps pipeline for predicting hotel reservation cancellations. The system helps hotels optimize their booking strategies by predicting which reservations are likely to be cancelled.

### Business Value
- **Revenue Optimization**: Reduce revenue loss from cancellations
- **Inventory Management**: Better room allocation strategies
- **Customer Insights**: Understand cancellation patterns
- **Operational Efficiency**: Automated prediction workflow

## 🤖 Model Details

### Algorithm
- **Type**: LightGBM Classifier
- **Technique**: GOSS (Gradient-based One-Side Sampling)
- **Validation**: 5-fold Cross Validation
- **Hyperparameter Tuning**: RandomizedSearchCV (20 total fits)
- **Training Time**: ~57 minutes
- **Dataset Size**: 30,462 samples (balanced: 15,231 positive, 15,231 negative)

### Features
The model uses 20 key features including:
- **Booking Details**: Lead time, arrival dates, stay duration
- **Guest Information**: Adults, children, babies count
- **Service Preferences**: Meal plans, special requests
- **Historical Data**: Previous cancellations, bookings
- **Financial**: Average daily rate (ADR), deposit type
- **Operational**: Market segment, customer type, parking needs

### Performance Metrics
- **Cross-Validation**: 5-fold CV for robust evaluation
- **Balanced Dataset**: Equal representation of cancelled/non-cancelled reservations
- **Feature Engineering**: Automated preprocessing pipeline
- **Model Persistence**: Saved models for consistent predictions

## 🏗️ Infrastructure

### Cloud Deployment
- **Platform**: Google Cloud Run
- **Container Registry**: Google Container Registry (GCR)
- **Region**: us-central1
- **Auto-scaling**: Enabled (0-1000 instances)
- **Authentication**: Public (unauthenticated access)
- **Port**: 8080
- **Memory**: 2GB
- **CPU**: 1 vCPU

### CI/CD Pipeline
- **Tool**: Jenkins
- **Trigger**: GitHub webhook on push to main
- **Build Time**: ~94 minutes (including model training)
- **Stages**: 
  1. ✅ Source checkout from GitHub
  2. ✅ Virtual environment setup
  3. ✅ Dependency installation
  4. ✅ Model training (57 min)
  5. ✅ Docker build & push to GCR
  6. ✅ Cloud Run deployment

### Container Details
- **Image**: `gcr.io/solid-drive-467918-h9/ml-project:latest`
- **Base Image**: Python 3.9-slim
- **Size**: Optimized for Cloud Run
- **Exposed Port**: 8080
- **Health Checks**: Built-in Flask health endpoints

## 📡 API Usage

### Live Endpoint
**Base URL**: https://ml-project-929445478726.us-central1.run.app

### Web Interface
Access the interactive prediction form at the base URL.

### REST API Endpoint

#### Prediction Request
```bash
curl -X POST https://ml-project-929445478726.us-central1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "lead_time": 30,
    "arrival_date_week_number": 25,
    "arrival_date_day_of_month": 15,
    "stays_in_weekend_nights": 2,
    "stays_in_week_nights": 3,
    "adults": 2,
    "children": 0,
    "babies": 0,
    "meal": "BB",
    "market_segment": "Online TA",
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
    "booking_changes": 0,
    "deposit_type": "No Deposit",
    "days_in_waiting_list": 0,
    "customer_type": "Transient",
    "adr": 75.5,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 1
  }'
```

#### Response Format
```json
{
  "prediction": "Not Cancelled",
  "probability": 0.85,
  "confidence": "High",
  "model_version": "lightgbm_v1.0"
}
```

### Input Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `lead_time` | Integer | Days between booking and arrival | 30 |
| `arrival_date_week_number` | Integer | Week number of arrival (1-53) | 25 |
| `arrival_date_day_of_month` | Integer | Day of month (1-31) | 15 |
| `stays_in_weekend_nights` | Integer | Weekend nights count | 2 |
| `stays_in_week_nights` | Integer | Weekday nights count | 3 |
| `adults` | Integer | Number of adults | 2 |
| `children` | Integer | Number of children | 0 |
| `babies` | Integer | Number of babies | 0 |
| `meal` | String | Meal plan (BB, HB, FB, SC) | "BB" |
| `market_segment` | String | Market segment | "Online TA" |
| `is_repeated_guest` | Integer | 1 if repeat guest, 0 otherwise | 0 |
| `previous_cancellations` | Integer | Previous cancellation count | 0 |
| `previous_bookings_not_canceled` | Integer | Previous successful bookings | 0 |
| `booking_changes` | Integer | Number of booking changes | 0 |
| `deposit_type` | String | Deposit type | "No Deposit" |
| `days_in_waiting_list` | Integer | Days on waiting list | 0 |
| `customer_type` | String | Customer type | "Transient" |
| `adr` | Float | Average daily rate | 75.5 |
| `required_car_parking_spaces` | Integer | Parking spaces needed | 0 |
| `total_of_special_requests` | Integer | Special requests count | 1 |

## 💻 Local Development

### Prerequisites
- Python 3.9+
- Git
- Virtual environment tool

### Setup Instructions

1. **Clone Repository**
```bash
git clone https://github.com/laavanjan/MLOPS-PROJECT-1.git
cd MLOPS-PROJECT-1
```

2. **Create Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Run Complete Pipeline**
```bash
# Run entire ML pipeline
python training_pipeline.py

# Or run individual components
python src/data_ingestion.py
python src/data_preprocessing.py
python src/model_training.py
```

5. **Run Web Application**
```bash
python app.py
```

6. **Access Locally**
```
http://localhost:5000
```

### Development Commands

```bash
# Run tests
python -m pytest tests/

# Train model locally
python src/model_training.py

# Build Docker image
docker build -t hotel-prediction .

# Run Docker container
docker run -p 8080:8080 hotel-prediction

# View MLflow UI
mlflow ui
```

## 🔄 CI/CD Pipeline

### Jenkins Pipeline Stages

1. **Source Checkout** (2-3 minutes)
   - Clones repository from GitHub
   - Checks out latest commit on main branch

2. **Environment Setup** (5-8 minutes)
   - Creates Python virtual environment
   - Installs all dependencies from requirements.txt
   - Validates environment setup

3. **Model Training** (~57 minutes)
   - Loads and preprocesses dataset
   - Performs 5-fold cross-validation
   - Hyperparameter tuning with RandomizedSearchCV
   - Saves trained model artifacts

4. **Docker Build & Push** (~25 minutes)
   - Builds Docker image with trained model
   - Pushes to Google Container Registry
   - Tags with latest and build number

5. **Cloud Run Deployment** (3-5 minutes)
   - Deploys to Google Cloud Run
   - Configures auto-scaling and networking
   - Validates deployment health

### Pipeline Triggers
- **Automatic**: Push to main branch
- **Manual**: Jenkins dashboard trigger
- **Webhook**: GitHub webhook integration

### Pipeline Configuration
```yaml
# Jenkinsfile equivalent configuration
stages:
  - source_checkout
  - environment_setup  
  - model_training
  - docker_build_push
  - cloud_run_deploy

estimated_total_time: "94 minutes"
success_rate: "100%"
```

## 📊 Monitoring & Maintenance

### Health Check Endpoints
```bash
# Application health
curl https://ml-project-929445478726.us-central1.run.app/health

# Model status
curl https://ml-project-929445478726.us-central1.run.app/model/status
```

### Google Cloud Commands

#### Check Service Status
```bash
gcloud run services describe ml-project --region=us-central1
```

#### View Application Logs
```bash
# Recent logs
gcloud logs read --service=ml-project --region=us-central1

# Live log streaming
gcloud logs tail --service=ml-project --region=us-central1
```

#### Service Management
```bash
# Update service
gcloud run deploy ml-project --image=gcr.io/solid-drive-467918-h9/ml-project:latest --region=us-central1

# Scale service
gcloud run services update ml-project --max-instances=10 --region=us-central1

# View metrics
gcloud run services describe ml-project --region=us-central1 --format="value(status.url)"
```

### Performance Monitoring
- **Response Time**: Monitored via Cloud Run metrics
- **Error Rate**: Tracked in Cloud Logging
- **Resource Usage**: CPU and memory metrics available
- **Scaling Events**: Auto-scaling logs in Cloud Run console

## 📁 Project Structure

```
MLOPS-PROJECT-1/
├── 📊 data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Cleaned and preprocessed data
│   └── models/                 # Trained model artifacts
├── 📝 src/
│   ├── data_ingestion.py       # Data loading and splitting
│   ├── data_preprocessing.py   # Data cleaning and feature engineering
│   ├── model_training.py       # Model training and validation
│   ├── model_evaluation.py     # Performance metrics and evaluation
│   ├── logger.py              # Centralized logging system
│   └── utils.py               # Utility functions
├── ⚙️ config/
│   ├── config.yaml            # Main configuration file
│   └── paths_config.py        # Path management
├── 🛠️ utils/
│   └── common_functions.py    # Shared utility functions
├── 🌐 templates/
│   ├── index.html             # Main prediction interface
│   └── result.html            # Prediction results page
├── 🚀 static/
│   ├── css/                   # Stylesheets
│   ├── js/                    # JavaScript files
│   └── images/                # Static images
├── 🧪 tests/
│   ├── test_model.py          # Model testing
│   ├── test_api.py            # API endpoint testing
│   └── test_preprocessing.py   # Data processing tests
├── 📝 logs/                    # Application logs
├── 🔄 artifacts/               # ML artifacts and models
├── 🐳 Dockerfile              # Container configuration
├── 🔧 requirements.txt        # Python dependencies
├── 🌍 app.py                  # Flask application entry point
├── 🚀 training_pipeline.py    # Complete ML pipeline orchestrator
├── ⚙️ config.py               # Configuration settings
├── 🔄 Jenkinsfile             # Jenkins pipeline configuration
├── ☁️ cloudbuild.yaml         # Google Cloud Build config
└── 📖 README.md               # This file
```

## 🛠️ Technologies Used

### Machine Learning Stack
- **Framework**: Scikit-learn, LightGBM
- **Data Processing**: Pandas, NumPy
- **Feature Engineering**: SMOTE, Random Forest
- **Experiment Tracking**: MLflow
- **Model Validation**: Cross-validation, Grid Search

### Web Application
- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **API**: RESTful endpoints
- **Templates**: Jinja2

### DevOps & Infrastructure
- **Containerization**: Docker
- **CI/CD**: Jenkins
- **Cloud Platform**: Google Cloud Platform (GCP)
- **Container Registry**: Google Container Registry (GCR)
- **Hosting**: Google Cloud Run
- **Version Control**: Git, GitHub

### Development Tools
- **IDE**: Visual Studio Code
- **Environment**: Python Virtual Environment
- **Testing**: Pytest
- **Logging**: Python logging, Google Cloud Logging
- **Configuration**: YAML, Python config modules

## 🔧 Troubleshooting

### Common Issues

#### 1. Application Not Responding
**Problem**: Application URL returns 503 or timeout errors
**Solution**:
```bash
# Check service status
gcloud run services describe ml-project --region=us-central1

# View recent errors
gcloud logs read --service=ml-project --region=us-central1 --severity=ERROR
```

#### 2. Prediction Errors
**Problem**: API returns validation or processing errors
**Solution**:
- Verify input data format matches API schema
- Check all required fields are provided
- Ensure data types are correct (integers vs strings)

#### 3. Build Failures
**Problem**: Jenkins pipeline fails during build
**Solution**:
- Check Jenkins build logs for specific error
- Verify dependencies in requirements.txt
- Ensure Docker has sufficient resources

#### 4. Model Loading Issues
**Problem**: Model fails to load or predict
**Solution**:
```bash
# Check model files exist
ls -la data/models/

# Verify model format compatibility
python -c "import joblib; print(joblib.load('data/models/model.pkl'))"
```

#### 5. Pipeline Execution Errors
**Problem**: `training_pipeline.py` fails to run
**Solution**:
```bash
# Check configuration files
python -c "from utils.common_functions import read_yaml; print(read_yaml('config/config.yaml'))"

# Verify data files
ls -la data/raw/

# Check logs for specific errors
tail -f logs/log_$(date +%Y-%m-%d).log
```

### Performance Issues

#### Slow Response Times
- **Check**: Cloud Run instance count and scaling settings
- **Solution**: Increase min-instances for faster cold starts
```bash
gcloud run services update ml-project --min-instances=1 --region=us-central1
```

#### Memory Errors
- **Check**: Cloud Run memory allocation
- **Solution**: Increase memory limit
```bash
gcloud run services update ml-project --memory=4Gi --region=us-central1
```

### Getting Help

#### Log Analysis
```bash
# Application logs
gcloud logs read --service=ml-project --region=us-central1

# Training pipeline logs
cat logs/log_$(date +%Y-%m-%d).log

# MLflow tracking
mlflow ui --host 0.0.0.0 --port 5000

# Container logs
docker logs <container-id>
```

#### Support Resources
- **Google Cloud Documentation**: https://cloud.google.com/run/docs
- **Flask Documentation**: https://flask.palletsprojects.com/
- **LightGBM Documentation**: https://lightgbm.readthedocs.io/
- **Jenkins Documentation**: https://www.jenkins.io/doc/

#### Debug Mode
For local development, enable debug mode:
```python
# In app.py
app.run(debug=True, host='0.0.0.0', port=5000)

# In training_pipeline.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset source: Hotel booking dataset
- Google Cloud Platform for hosting infrastructure
- Jenkins community for CI/CD pipeline tools
- Open source ML libraries: Scikit-learn, LightGBM, MLflow

---

**🔗 Live Application**: https://ml-project-929445478726.us-central1.run.app

## 📞 Contact & Connect

**👨‍💻 Developer**: Laavanjan Luckkumikanthan

**📧 Email**: [laavanjanlaa@gmail.com](mailto:laavanjanlaa@gmail.com)  
**📱 LinkedIn**: [Connect with me](https://www.linkedin.com/in/laavanjan-luckkumikanthan)  
**🐙 GitHub**: [View my repositories](https://github.com/laavanjan)  

### 💬 Get in Touch
- 🤝 **Collaboration**: Open to MLOps and Data Science opportunities
- 💡 **Questions**: Feel free to ask about this project or ML implementations
- 🔗 **Networking**: Always happy to connect with fellow data enthusiasts
- 📝 **Feedback**: Your suggestions and improvements are welcome!

---

*Last updated: August 9, 2025*  
*Pipeline Status: ✅ Passing*  
*Deployment: 🟢 Live*