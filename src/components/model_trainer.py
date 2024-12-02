import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifact', "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test= (
                train_array[:,:-1], #take out the last column as its the *target_feature_train_df -> X_train
                train_array[:,-1],  #remove all the columns except last column the *target_feature_train_df -> y_train
                test_array[:,:-1],  #take out the last column as its the *target_feature_test_df -> X_test
                test_array[:,-1],   #remove all the columns except last column the *target_feature_test_df -> y_test
            )

            models = {
                "Random Forest Regressor": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(), 
                "XGBClassifier": XGBRFRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor(),
            }

            model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                                             models=models)
            
            ##To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ##To get best model name from dict
            best_model_name = list(model_report.keys())[            #this will create a list
                list(model_report.values()).index(best_model_score)  #choose the one with best index
            ]
            
            best_model = models [best_model_name]

            if best_model_score<=0.6:
                raise CustomException("No best model Found")
            
            logging.info(f"Best found model on both training and testing dataset")
            
            save_object (
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicated = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicated)

            return r2_square
        
        except Exception as e:
            raise CustomException (e,sys)