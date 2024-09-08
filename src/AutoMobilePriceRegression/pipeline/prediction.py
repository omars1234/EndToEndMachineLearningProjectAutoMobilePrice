import joblib
import numpy as np
import pandas as pd
from pathlib import Path



class PredictionPipeline:
    def __init__(self):
        self.model=joblib.load(Path("artifacts/training/model.joblib"))

    def predict(self,input_data):
        prediction=self.model.predict(input_data)

        return np.round(prediction) 
    