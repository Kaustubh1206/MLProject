# Data Ingestion is process where we read the data from external source ( database , or web scrapping a website)
# Reading a data is kind of module . Thats why we have put this in our component


# --------------------- Importing all Libraries -----------------------------------
#        os , sys  -> working with file paths and system errors 
#        dataclass -> simpleway to define config class
#        CustomException -> custom errors handling for easier debugging
#        logging -> logs messages for tracking execution
#
#        Model Trainer -> trains ML models
#        DataTransformation -> process the raw data 
#        train_test_split -> spliting the data into training and testing 
#        pandas -> reading and manipulating with dataframes


import os
import sys 
from src.exception import CustomException
from src.logger import logging 
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses  import dataclass 

from src.components.data_transformation import DataTransformation, DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer 


#------------------------------- Configuration Class -------------------------------
# Data ingestion config - providing input things which are required for data ingestion 

@dataclass # -> makes it easy to define classes with variables without writing constructor
# decorator - > directly define class variable
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv') # this will save all the train data in this path ( train.csv)
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','raw.csv')
    # input which we are giving and this will let them know where to save files


#-------------------------------- Data Ingestion Class ------------------------------
#               Initializes the configuration class so the path are avaiable for use.
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig() # calling all the files of DataIngestionConfig



#-------------------------------- Data Ingestion Methods ----------------------------
#       1. Reads raw data from CSV.
#       2. Creates a folder to save artifacts (train/test/raw).
#       3. Saves a copy of raw data.
#       4. Splits data into train (80%) and test (20%).
#       5. Saves the train/test data as CSV.
#       6. Returns paths to train and test CSV files for next steps.

    def initate_data_ingestion(self):   # initating data ingestioon
        logging.info("Entered the data ingestion method or component") # logging for data ingestion
        try:

            # Reading Dataset 
            df=pd.read_csv('/Users/kaustubh/Desktop/ML/Project/notebook/data/stud.csv') # getting data from local dataset
            logging.info('Read the Dataset as Dataframe') 
            
            # Make folder if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) # making directory , if the directory is alreaady exist then 0use it

            # save raw copy
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            # logging initated rain test split
            logging.info("Train test split initiated")
            
            # train test split
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            # after train test split we will store the split train data to 
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            # after train test split we will store the split test data to 
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of data is completeed") # logging got completed

            return(

                # We will return this data for next file that is Data Transformation 
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path
            )

            # If any error got raised, this will raise the customexception  
        except Exception as e: 
            raise CustomException(e,sys)

#------------------------------------- Main Execution ----------------------------
#   1. Data Ingestion: read + split + save data → returns file paths.

#   2. Data Transformation: process raw train/test CSV → returns arrays for ML.

#   3. Model Training: trains ML models and prints the result.
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initate_data_transformation(train_data,test_data)

    modelTrainer=ModelTrainer()
    print(modelTrainer.initiate_model_trainer(train_arr,test_arr))