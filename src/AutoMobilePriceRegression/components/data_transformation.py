
import sys
import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from src.AutoMobilePriceRegression import logger
from src.AutoMobilePriceRegression.utils.common import get_size
from sklearn.model_selection import train_test_split
import os

from AutoMobilePriceRegression.config.configuration import DataTransfornmationConfig



class DataTransfornmation:
    def __init__(self,config:DataTransfornmationConfig):
        self.config= config


    def data_LabelEncoder(self):
        df=pd.read_csv(self.config.data_path)
        for col in df:
            if col in list(df.select_dtypes(include="object").columns):
                df[col]=LabelEncoder().fit_transform(df[col])
        return df    
                

    def train_test_splitting(self):

        df=self.data_LabelEncoder()

        train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

        train_set.to_csv(os.path.join(self.config.root_dir,"train.csv"),index=False)
        test_set.to_csv(os.path.join(self.config.root_dir,"test.csv"),index=False)

        logger.info("Data Splitting is completed") 
        logger.info(train_set.shape) 
        logger.info(test_set.shape) 