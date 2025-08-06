import os
import pandas
from src.logger import get_logger
from src.custom_exception import CustomException
import yaml
import pandas as pd


logger = get_logger(__name__)


## Data Ingestion Utility Functions with GCP and YAML support

def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'YAML file not found at {file_path}')
        
        with open(file_path,'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info(f'Successfully read YAML file')
            return config
        
    except Exception as e:
        logger.error(f'Error reading YAML file: {e}')
        raise CustomException(f'Error reading YAML file: {e}')


# Data Processing Function to Load Data from CSV

def load_data(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'CSV file not found at {file_path}')
        
        data = pandas.read_csv(file_path)
        logger.info(f'Successfully loaded data from {file_path}')
        return data
    
    except Exception as e:
        logger.error(f'Error loading data from CSV file: {e}')
        raise CustomException(f'Error loading data from CSV file: {e}')

