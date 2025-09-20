# Here we will train our model on training data .
# We will used various models and performance metrics
from dataclasses import dataclass
import os 
import sys 

from catboost import CatBoostRegressor

from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

# -----------------MODEL TRAINING-----------

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_cofig=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array): # output of transformation in form of array as a input
        try:
            logging.info("Spliting Training and test Input data .")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1], # all columns except leaving the taret feature
                train_array[:,-1],# last train test data 
                test_array[:,:-1], # X test data split
                test_array[:,-1] # y test data split

            )

            models={
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K -Neighbors Classifiers":KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor(),
            }
            
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)


            # To get best model score from dictionary 
            best_model_score= max(sorted(model_report.values())) # here we will get best model sore base on performance

            # To get best model name from dict 
            best_model_name=list(model_report.keys())[      
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best Model Found")
            logging.info(f"Best found models on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_cofig.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square=r2_score(y_test, predicted)
            return r2_square


        except Exception as e:
            raise CustomException(e,sys)
