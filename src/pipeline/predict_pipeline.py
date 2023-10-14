from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd
import os
import sys

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = os.path.join('artifacts', 'model3.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor1.pkl')
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self,carat,cut,color,clarity,table,length,width,depth):
        self.carat = carat
        self.cut = cut
        self.color = color
        self.clarity = clarity
        self.table = table
        self.length = length
        self.width = width
        self.depth = depth

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "carat": [self.carat],
                "cut" : [self.cut],
                "color" : [self.color],
                "clarity" : [self.clarity],
                "table" : [self.table],
                "length" : [self.length],
                "width" : [self.width],
                "depth" : [self.depth],
            }  
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
                  