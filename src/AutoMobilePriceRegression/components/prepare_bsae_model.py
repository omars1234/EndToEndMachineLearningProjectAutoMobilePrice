
import sys
import os
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder
from src.AutoMobilePriceRegression import logger
from src.AutoMobilePriceRegression.utils.common import get_size
#from src.AutoMobilePriceRegression.components.data_ingestion import DataIngestion

from sklearn.ensemble import GradientBoostingRegressor
from AutoMobilePriceRegression.config.configuration import PrepareBaseModelConfig



class PrepareBaseModel:
    def __init__(self,config:PrepareBaseModelConfig):
        self.config= config
    

    def get_data_transformer_object(self):
        numerical_columns = ['length', 'width', 'height', 'curb_weight', 'engine_size','peak_rpm', 'city_mpg', 'highway_mpg']
        categorical_columns = ['num_of_doors', 'body_style', 'drive_wheels','engine_location','num_of_cylinders', 'fuel_system']

        num_pipeline= Pipeline(steps=[
            ("scaler",StandardScaler())
            ])
        
        cat_pipeline=Pipeline(steps=[
            ("OneHotEncoder",OneHotEncoder(handle_unknown="ignore")),
            ("scaler",StandardScaler(with_mean=False))
            ])

        preprocessor=ColumnTransformer([
            ("num_pipeline",num_pipeline,numerical_columns),
            ("cat_pipelines",cat_pipeline,categorical_columns)
            ])
        
        import pickle
        with open ("./artifacts/model_preprocessor.pkl","wb") as f:
           pickle.dump(preprocessor,f)  
                           
        return preprocessor
        
    
    def initiate_data_transformation(self): 
        train_df=pd.read_csv(r"./artifacts/data_ingestion/train_data.csv")
        test_df=pd.read_csv(r"./artifacts/data_ingestion/test_data.csv")

        logger.info("Read train and test data completed")

        preprocessing_obj=self.get_data_transformer_object()        

        target_column_name="price"        

        input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
        target_feature_train_df=train_df[target_column_name]

        input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
        target_feature_test_df=test_df[target_column_name]     

        input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
        input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

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