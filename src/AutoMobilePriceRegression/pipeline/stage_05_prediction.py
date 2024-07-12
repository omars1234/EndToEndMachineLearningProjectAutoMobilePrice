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

        with open("./artifacts/model_preprocessor.pkl","rb") as f:
           self.loaded_preprocessor=pickle.load(f)   

    def predict(self,data):

        preprocessor=self.loaded_preprocessor
        InputData_scaled=preprocessor.transform(data)
        prediction=self.loaded_model.predict(InputData_scaled)   
        return prediction
        
'''
class InputData:
    def __init__(self,
                num_of_doors:str, body_style:str,drive_wheels:str,
                engine_location:str,num_of_cylinders:str, fuel_system:str,

                length:int,width:int,height:int,curb_weight:int,
                engine_size:int,peak_rpm:int,city_mpg:int,highway_mpg:int                     
                ):
        self.num_of_doors=num_of_doors, self.body_style=body_style,self.drive_wheels=drive_wheels,
        self.engine_location=engine_location,self.num_of_cylinders=num_of_cylinders,self.fuel_system=fuel_system,

        self.length=length, self.width=width, self.height=height, self.curb_weight=curb_weight,
        self.engine_size=engine_size,self.peak_rpm=peak_rpm,self.city_mpg=city_mpg,self.highway_mpg=highway_mpg

    def get_data_as_data_frame(self):
        try:
            input_data_dict={
                'num_of_doors':[self.num_of_doors],'body_style':[self.body_style],'drive_wheels':[self.drive_wheels],
                'engine_location':[self.engine_location],'num_of_cylinders':[self.num_of_cylinders], 'fuel_system':[self.fuel_system],

                'length':[self.length], 'width':[self.width], 'height':[self.height], 'curb_weight':[self.curb_weight],
                'engine_size':[self.engine_size],'peak_rpm':[self.peak_rpm], 'city_mpg':[self.city_mpg], 'highway_mpg':[self.highway_mpg]

            }   
            return pd.DataFrame(input_data_dict)
        
        except Exception as e:
            logger.exception(e)
            raise e

'''


