import os
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
import joblib
from pathlib import Path
from AutoMobilePriceRegression.utils.common import save_json
from AutoMobilePriceRegression.config.configuration import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self,config:ModelEvaluationConfig):
        self.config= config

    def evaluation_metrics(self,actual,pred):
        RMSE=root_mean_squared_error(actual,pred) 
        return RMSE 

    def save_results(self):
        test_data=pd.read_csv(self.config.test_data_path)
        model=joblib.load(self.config.model_path)

        x_test,y_test=(
            test_data.drop(self.config.target_column,axis=1),
            test_data[self.config.target_column]
        )

        prediction=model.predict(x_test)

        (RMSE)=self.evaluation_metrics(y_test,prediction)

        scors={"RMSE":RMSE}

        save_json(path=Path(self.config.metrics_file_name),data=scors)         