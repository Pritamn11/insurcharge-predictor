import os
import sys

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    preprocessor_obj_filepath = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_cols = ['age', 'bmi', 'children']
            categorical_cols = ['sex', 'smoker']
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]

            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder", OneHotEncoder())
                ]
            )

            logging.info(f"Numerical columns : {numerical_cols}")
            logging.info(f"Categorical columns : {categorical_cols}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_cols),
                    ("cat_pipeline", cat_pipeline, categorical_cols)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)   

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test data completed")
            
            numerical_cols = ['age', 'bmi', 'children']
            categorical_cols = ['sex', 'smoker']

            target_feature = 'charges'

            input_train_df = train_df.drop(columns=[target_feature], axis=1)
            target_train_df = train_df[target_feature]

            input_test_df = test_df.drop(columns=[target_feature], axis=1)
            target_test_df = test_df[target_feature]

            logging.info("Applying preprocessing object on train and test dataframe")

            preprocessor_obj = self.get_data_transformer_object()

            input_train_df_arr = preprocessor_obj.fit_transform(input_train_df)
            input_test_df_arr = preprocessor_obj.transform(input_test_df)

            train_arr = np.c_[
                input_train_df_arr, np.array(target_train_df)
            ]

            test_arr = np.c_[
                input_test_df_arr, np.array(target_test_df)
            ]

            logging.info("Saving preprocessing object")

            save_object(
               file_path= self.data_transformation_config.preprocessor_obj_filepath, 
               obj= preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_filepath
            )

        except Exception as e:
            raise CustomException(e,sys)
