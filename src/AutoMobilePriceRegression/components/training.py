
import os
import urllib.request as request
from sklearn.ensemble import GradientBoostingRegressor
import pickle,joblib
import pandas as pd
from AutoMobilePriceRegression.config.configuration import TrainingConfig



class Training:
    def __init__(self,config:TrainingConfig):
        self.config= config

    def initiate_Training(self):
        train_data=pd.read_csv(self.config.train_data_path)
        test_data=pd.read_csv(self.config.test_data_path)
        x_train,y_train,x_test,y_test=(
            train_data.drop(self.config.target_column,axis=1),train_data[self.config.target_column],
            test_data.drop(self.config.target_column,axis=1),test_data[self.config.target_column]
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

        joblib.dump(model,os.path.join(self.config.root_dir,self.config.model_name))