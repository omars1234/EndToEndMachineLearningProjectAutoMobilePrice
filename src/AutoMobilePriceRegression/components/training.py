
import os
import pandas as pd
from numpy import loadtxt
import urllib.request as request
import pickle
import time

from sklearn.ensemble import GradientBoostingRegressor

from AutoMobilePriceRegression.config.configuration import TrainingConfig



class Training:
    def __init__(self,config:TrainingConfig):
        self.config= config

    def initiate_Training(self):   
        
        input_feature_train_arr=pd.read_csv('artifacts/prepare_base_model/train_data.csv', delimiter=',')
        input_feature_test_arr=pd.read_csv('artifacts/prepare_base_model/test_data.csv', delimiter=',')
        x_train,y_train,x_test,y_test=(
            input_feature_train_arr.drop("price",axis=1),
            input_feature_train_arr["price"],
            input_feature_test_arr.drop("price",axis=1),
            input_feature_test_arr["price"]
        )
        model=GradientBoostingRegressor(
                subsample=self.config.subsample,
                n_estimators=self.config.n_estimators,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate
        )
        model.fit(x_train,y_train)

        import pickle
        with open ("./artifacts/final_model.pkl","wb") as f:
           pickle.dump(model,f)
