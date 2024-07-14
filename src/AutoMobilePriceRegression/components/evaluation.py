import pickle
import pandas as pd
import numpy as np
from numpy import loadtxt
from pathlib import Path

import urllib.request as request
from urllib.parse import urlparse
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score,mean_squared_error
from AutoMobilePriceRegression.utils.common import save_json
import time

from AutoMobilePriceRegression.config.configuration import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self,config:ModelEvaluationConfig):
        self.config= config

    def evaluation_metrics(self):
        with open("./artifacts/final_model.pkl","rb") as f:
           loaded_model=pickle.load(f)
        
        test_data=pd.read_csv('artifacts/prepare_base_model/test_data.csv', delimiter=',')
        x_test,y_test=(
            
            test_data.drop("price",axis=1),
            test_data["price"]
            )      
        rmse=np.sqrt(mean_squared_error(y_test,loaded_model.predict(x_test)))
        mse=mean_squared_error(y_test,loaded_model.predict(x_test))
        r2score=r2_score(y_test,loaded_model.predict(x_test))
        Scores={"rmse":rmse,"mse":mse,"r2score":r2score}
        save_json(path=Path("scores.json"),data=Scores)