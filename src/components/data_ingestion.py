#read dataset from specific datasource
#read and split the data

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
#for creating class variables
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


#tosave train,test,raw data

#decorator
#use data class only if you are declaring variables, else use init method itself
@dataclass
class DataIngestionConfig:
    train_data_path = str=os.path.join('artifacts',"train.csv")
    test_data_path = str=os.path.join('artifacts',"test.csv")
    raw_data_path = str=os.path.join('artifacts',"data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    #read data from the stored databases
    def inititate_data_ingestion(self):
        logging.info("Entered the Data Ingestion method")

        try:
            #path to the local csv file, same to be used if extracting from some db
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Dataset read as Df')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train Test split inititated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=23)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Data Ingestion is complete")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj = DataIngestion()
    train_data,test_data = obj.inititate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))