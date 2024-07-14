
import sys
import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from src.AutoMobilePriceRegression import logger
from src.AutoMobilePriceRegression.utils.common import get_size
from sklearn.model_selection import train_test_split
import os

from AutoMobilePriceRegression.config.configuration import PrepareBaseModelConfig



class PrepareBaseModel:
    def __init__(self,config:PrepareBaseModelConfig):
        self.config= config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file_path):
            df=pd.read_csv(self.config.source_URL)
            df.to_csv(self.config.local_data_file_path,index=False,header=True)
            CatEncod=LabelEncoder()
            categorical_columns = ['num_of_doors', 'body_style', 'drive_wheels',
                                   'engine_location','num_of_cylinders', 'fuel_system']

            for col in df:
                 if col in categorical_columns:
                      df[col]=CatEncod.fit_transform(df[col])
                              
                      
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            if not os.path.exists(self.config.train_data_path):
                        train_set.to_csv(self.config.train_data_path,index=False,header=True)            
                
            if not os.path.exists(self.config.test_data_path):
                        test_set.to_csv(self.config.test_data_path,index=False,header=True) 

            import pickle
            with open ("./artifacts/model_CatEncod.pkl","wb") as f:
                pickle.dump(CatEncod,f)             

            return(
                        self.config.train_data_path,
                        self.config.test_data_path
                    )

    def initiate_data_transformation(self): 
        train_df=pd.read_csv(r"./artifacts/prepare_base_model/train_data.csv")
        test_df=pd.read_csv(r"./artifacts/prepare_base_model/test_data.csv")

        logger.info("Read train and test data completed")

        sc=StandardScaler()     

        target_column_name="price"
        
        input_feature_train_df=train_df.drop(target_column_name,axis=1)
        target_feature_train_df=train_df[target_column_name]

        input_feature_test_df=test_df.drop(target_column_name,axis=1)
        target_feature_test_df=test_df[target_column_name]     

        input_feature_train_arr=sc.fit_transform(input_feature_train_df)
        input_feature_test_arr=sc.transform(input_feature_test_df)

        train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
                ]
        test_arr = np.c_[
                 input_feature_test_arr, np.array(target_feature_test_df)
                 ]
        
        if not os.path.exists(self.config.train_data_path):
                train_arr = pd.DataFrame(train_arr)#columns=[colnames])
                train_arr.to_csv(self.config.train_data_path,index=False)
        if not os.path.exists(self.config.test_data_path):
                test_arr = pd.DataFrame(test_arr)#,columns=[colnames])
                test_arr.to_csv(self.config.test_data_path,index=False)        

        return(
                self.config.train_data_path,
                self.config.test_data_path
         
        )      