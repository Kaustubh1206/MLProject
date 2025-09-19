# In data transformation we will tranform our data
# For example - Categorical features to numerical features
#             - One hot Encoding 

import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer # for encoding and Column Transformer
from sklearn.impute import SimpleImputer # Imuptation technique
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os 

from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")  # preprocessor will be stored in form of pixkle file 


class DataTransformation: 
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig() # 

    # in this function , it will be responsible to convert categorical feature to numerical
    def get_data_transformer_object(self):

        '''
            This Function is responsible for data transformation.
        '''
        try:
            numerical_columns=['reading_score', 'writing_score']
            categorical_columns=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']


            # Numerical Columns 
            num_pipeline= Pipeline(
                steps=[
                    # Here we can impute any imputation for missing values
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            # Categorical Columns
            cat_pipeline=Pipeline(
                # Here we can do any type of encoding 
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
             )   
            logging.info(f"Numerical columns : {categorical_columns}")
            logging.info(f"Categorical columns : {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def initate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            
            logging.info("Obtaining Preprocessing Object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns=["writing_score","reading_score"]

            input_feature_train_df=train_df.drop(columns=target_column_name,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=target_column_name,axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Appplying the preprocessing object on training dataframe and testing dataframe."
            
            )
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr =np.c_[                   # Here we have used c_ to stack the two array ion vertical form [1,2,3] [4,5,6] -> [1,4] ,[2,5]                
                input_feature_train_arr,np.array(target_feature_train_df)                 
            ]


            test_arr =np.c_[                   # Here we have used c_ to stack the two array ion vertical form [1,2,3] [4,5,6] -> [1,4] ,[2,5]                
                input_feature_test_arr,np.array(target_feature_test_df)                 
            ]
            
            logging.info(f"Saving preprocessing object.")


            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)