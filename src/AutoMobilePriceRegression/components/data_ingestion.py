
import os
import urllib.request as request
from src.AutoMobilePriceRegression import logger
from src.AutoMobilePriceRegression.utils.common import get_size
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
from AutoMobilePriceRegression.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            df=pd.read_csv(self.config.source_URL)

            df.to_csv(self.config.local_data_file,index=False,header=True)

        logger.info("Ingestion of the data is completed")
       
        
       
        