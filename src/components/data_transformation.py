#data cleaning, data transformation, conversion

import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
#for missing values
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    #create models and save to pickle file for future purpose
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pki")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    #to create all pickle files, that convert categorical to numerical
    def get_data_transformer_object(self):

        '''
        This function is responsible for data transformation based on different types of data
        '''

        try:
            numerical_columns = ["writing_score","reading_score"]
            categorical_columns = ["gender",
                                    "race_ethnicity",
                                    "parental_level_of_education",
                                    "lunch",
                                    "test_preparation_course"]
            
            #create pipeline and handle missing values
            #pipeline helps in handling the missing values and does standard scaling
            num_pipeline = Pipeline(
                steps = [
                    #will be responsible for handling missing values along with startegy
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scalar",StandardScaler(with_mean=False))
                ]
            )

            #missing values in cat features and how to handle them, handling and converting them into numerical values

            cat_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="most_frequent")), #fills missing value with mode
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scalar",StandardScaler(with_mean=False))
                ]
            )


            logging.info("Numerical Columns: {numerical_columns}")
            logging.info("Categorical Columns: {categorical_columns}")
            
            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")

            #combine numerical pipelin with categorical pipeline
            preprocessor = ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)    
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys) 
        
    #train path and test path comes from dataingestion.py file
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining pre processing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score","reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)             
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on train and test dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info("saved Preprocessing object")

            #to save the pickle file
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )            
            
        except Exception as e:
            raise CustomException(e,sys)
 