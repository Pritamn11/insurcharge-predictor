import os
import sys 
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import *
from src.components.model_trainer import *
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split



@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    raw_data_path = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestionConfig = DataIngestionConfig()

    def InititateDataIngestion(self):
        logging.info("Entered the data ingestion method")
        try:
            logging.info("Started reading dataset")
            df = pd.read_csv('notebook\data\insurance.csv')
            logging.info('Read dataset from dataframe')

            os.makedirs(os.path.dirname(self.ingestionConfig.train_data_path), exist_ok=True)
            
            df.reset_index(drop=True, inplace=True)
            df.to_csv(self.ingestionConfig.raw_data_path, index=False, header=True)
            
            
            logging.info("Train Test split initiated")
            train_data, test_data = train_test_split(df, test_size=0.20, random_state=42)
            
            logging.info("Saving train and test data")
            train_data.to_csv(self.ingestionConfig.train_data_path, index=False, header=True)
            test_data.to_csv(self.ingestionConfig.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestionConfig.train_data_path,
                self.ingestionConfig.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)    


# if __name__=="__main__":
#     obj = DataIngestion()
#     # obj.InititateDataIngestion()
#     train_data, test_data = obj.InititateDataIngestion()

#     data_transformation = DataTransformation()
#     # data_transformation.initiate_data_transformation(train_data, test_data)
#     train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

#     model_trainer = ModelTrainer()
#     print(model_trainer.initiate_model_trainer(train_arr, test_arr))

