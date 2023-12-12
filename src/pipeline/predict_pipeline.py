import os
import sys 
import pandas as pd
from src.utils import load_object
from src.exception import CustomException

from src.utils import load_object


class PredictPipeline():
    def __init__(self):
        pass 

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            scaled_data = preprocessor.transform(features)
            predicton = model.predict(scaled_data)
            return predicton

        except Exception as e:
            raise CustomException(e,sys)
        


class CustomData:
    def __init__(self, age : int, bmi : float, children : int, sex : str, smoker : str):
        self.age = age
        self.bmi = bmi
        self.children = children
        self.sex = sex
        self.smoker = smoker

    def get_data_as_frame(self):
        try:
            custom_data_input_dict = {
                "age" : [self.age],
                "bmi" : [self.bmi],
                "children" : [self.children],
                "sex" : [self.sex],
                "smoker" : [self.smoker]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e,sys)    


# if __name__=="__main__":
#     cs = CustomData(age=45, bmi=53.2, children=3, sex='male', smoker='yes')
#     features = cs.get_data_as_frame()
#     pred = PredictPipeline()
#     print(pred.predict(features=features))