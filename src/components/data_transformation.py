import sys 
import os

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
@dataclass
class DataTransformationConfig:
    #To create model as a pickle file
    preprocesser_obj_file_path=os.path.join ('artifact', "preprocessor.pkl")

class DataTransforamtion:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This Functiom is responsible for data transformation
        '''
        try:
            numerical_columns = ["writing_score","reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    #this handling the missing values in the data using median value
                    ("imputer", SimpleImputer(strategy="median")),
                    #Standardization ensures that all numerical columns are on the same scale, 
                    #which helps many machine learning models perform better.r
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    #this handling the missing values
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    #Machine learning models can’t directly work with text, so we convert categories into numbers.
                    ("one_hot_encoder", OneHotEncoder()),
                    #Standardization ensures that all numerical columns are on the same scale, 
                    #which helps many machine learning models perform better.r
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )
            
            logging.info(f"Numerical columns: {numerical_columns}" )
            logging.info(f"Categorical columns: {categorical_columns}")

            #combine the Categorical with Numerical pipeline using Columntransformer
            preprocessor= ColumnTransformer(
                    [
                        ("num_pipeline", num_pipeline,numerical_columns),
                        ("cat_pipeline", cat_pipeline,categorical_columns),
                    ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation (self,train_path,test_path):

        try:
            train_df = pd.read_csv (train_path)
            test_df = pd.read_csv (test_path)

            logging.info("Read Train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessor_obj=self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns  = ["writing_score","reading_score"]


            input_feature_train_df= train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df= train_df[target_column_name]

            input_feature_test_df= test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df= test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr  = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object. ")

            save_object(

                file_path=self.data_transformation_config.preprocesser_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocesser_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)