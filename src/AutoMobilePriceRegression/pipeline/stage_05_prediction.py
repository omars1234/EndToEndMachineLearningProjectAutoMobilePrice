import numpy as np
import pickle
import pandas as pd
from pathlib import Path



from AutoMobilePriceRegression.utils.common import logger


STAGE_NAME = "model evaluation stage"

class PredictionPipeline:
    def __init__(self):
        with open("./artifacts/final_model.pkl","rb") as f:
           self.loaded_model=pickle.load(f)

    def predict(self,data):   

        prediction=self.loaded_model.predict(data)
        
        return prediction
    