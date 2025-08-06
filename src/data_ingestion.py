import os
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml

logger = get_logger(__name__)
#initialize logger

class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config['bucket_name']
        self.file_name = self.config['bucket_file_name']
        self.train_test_ratio = self.config['train_ratio']

        os.makedirs(RAW_DIR, exist_ok=True)

        logger.info('Data Ingestion initialized with {self.bucket_name} and {self.file_name}')

    
    def download_csv_from_gcp(self):
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)

            blob.download_to_filename(RAW_FILE_PATH)

            logger.info(f'Downloaded {self.file_name} from bucket {self.bucket_name} to {RAW_FILE_PATH}')
        
        except Exception as e:
            logger.error('Error downloading file from GCP')
            raise CustomException(f'Failed to download file from GCP: {str(e)}')
        

    def split_data(self):
        try:
            logger.info('Starting data split process')
            df = pd.read_csv(RAW_FILE_PATH)
            train_data, test_data = train_test_split(df, train_size=self.train_test_ratio, random_state=42)

            train_data.to_csv(TRAIN_FILE_PATH)
            test_data.to_csv(TEST_FILE_PATH)

            logger.info(f'Training data saved to {TRAIN_FILE_PATH}')
            logger.info(f'Test data saved to {TEST_FILE_PATH}')

        except Exception as e:
            logger.error('Errror while splitting data')
            raise CustomException(f'Failed to split data into training and testing sets', e)
        

    def run(self):

        try:
            logger.info('Startig data ingestion process')

            self.download_csv_from_gcp()
            self.split_data()

            logger.info('Data ingestion process completed successfully')
        except Exception as e:
            logger.error(f'Custom exception occurred: {str(e)}')

        finally:
            logger.info('Data ingestion completed')


if __name__ == '__main__':
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()