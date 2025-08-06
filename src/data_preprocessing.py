import os
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataPreprocessor:
    def __init__(self,train_path, test_path,processs_dir,config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processs_dir = processs_dir

        self.config_path = read_yaml(config_path)

        if not os.path.exists(self.processs_dir):
            os.makedirs(self.processs_dir)

    def preprocess_data(self,df):
        try:
            logger.info("Starting our data preprocessing step")

            logger.info("Dropping the columns that are not required")
            df.drop(columns=['Unnamed: 0', 'Booking_ID'], inplace=True)
            df.drop_duplicates(inplace=True)

            cat_cols = self.config_path['data_processing']['categorical_columns']
            num_cols = self.config_path['data_processing']['numerical_columns']

            logger.info('Applying label encoding')
            labelEncoder = LabelEncoder()
            mappings = {}
            for col in cat_cols:
                df[col] = labelEncoder.fit_transform(df[col])
                mappings[col] = {label:code for label, code in zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_))}
            
            logger.info('Label encoding mappings are')
            for col, mapping in mappings.items():
                logger.info(f'{col}: {mapping}')

            logger.info('Doing Skewness Handling')
            skewness_threshold = self.config_path['data_processing']['skewness_threshold']
            skewness= df[num_cols].apply(lambda x:x.skew())

            for col in skewness[skewness > skewness_threshold].index:
                logger.info(f'Skewness detected in {col}: {skewness[col]}')
                df[col] = np.log1p(df[col])

            return df
        except Exception as e:
            logger.error(f'Error in data preprocess step: {e}')
            raise CustomException('Error in data preprocess step', e)

    def balance_data(self, df):
        try:
            logger.info("Handing imbalanced data")
            X= df.drop(columns=['book_status'])
            y=df['book_status']

            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_df['book_status'] = y_resampled

            logger.info("Imbalanced data handled successfully")
            return balanced_df
        except Exception as e:
            logger.error(f'Error in balancing data: {e}')
            raise CustomException('Error in balancing data', e)
        
    def select_features(self, df):
        try:
            logger.info("Starting feature selection")

            X= df.drop(columns=['book_status'])
            y=df['book_status']

            model= RandomForestClassifier(random_state=42)
            model.fit(X, y)

            feature_importances = model.feature_importances_

            feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importances})
            top_features = feature_importance_df.sort_values(by='importance', ascending=False)
            num_features_to_select = self.config_path['data_processing']['num_features']
            top_10_features= top_features['feature'].head(num_features_to_select).values

            top_10_df = df[top_10_features.tolist() + ['booking_status']]

            logger.info('feature selection completed successfully')
            return top_10_df
        except Exception as e:
            logger.error(f'Error in feature selection: {e}')
            raise CustomException('Error in feature selection', e)
        
    
    def save_data(self, df,file_path):
        try:
            logger.info(f'Saving processed data to {file_path}')
            df.to_csv(file_path, index=False)
            logger.info(f'Data saved successfully to {file_path}')
        except Exception as e:
            logger.error(f"Error during saving data step:{e}")
            raise CustomException('Error while saving data',e)
        
    def process(self):
        try:
            logger.info("Loading the Data RAW directory")

            train_df= load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df=self.preprocess_data(train_df)
            test_df= self.preprocess_data(test_df)

            train_df=self.balance_data(train_df)
            test_df=self.balance_data(test_df)

            train_df=self.select_features(train_df)
            test_df=test_df[train_df.columns]

            self.save_data(train_df,PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df,PROCESSED_TEST_DATA_PATH)

            logger.info("Data processing completed successfully")
        except Exception as e:
            logger.error(f"Error during preprocessing pipeline {e}")
            raise CustomException('Error during preprocessing pipeline',e)
        

if __name__ == '__main__':
    processor = DataPreprocessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
    processor.process()
