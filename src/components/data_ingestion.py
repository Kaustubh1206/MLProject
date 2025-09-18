# Data Ingestion is process where we read the data from external source ( database , or web scrapping a website)
# Reading a data is kind of module . Thats why we have put this in our component
import os
import sys 
from src.exception import CustomException
from src.logger import logging 
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses  import dataclass 

# Data ingestion config
@dataclass # decorator - > directly define class variable
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv') # this will save all the train data in this path ( train.csv)
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','raw.csv')
    # input which we are giving and this will let them know where to save files

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig() # calling all the files of DataIngestionConfig

    def initate_data_ingestion(self):   # initating data ingestioon
        logging.info("Entered the data ingestion method or component") # logging for data ingestion
        try:
            # Reading Dataset 
            df=pd.read_csv('/Users/kaustubh/Desktop/ML/Project/notebook/data/stud.csv') # getting data from local dataset
            logging.info('Read the Dataset as Dataframe') 

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) # making directory , if the directory is alreaady exist then 0use it

            # Here we will take the raw data 
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

if __name__=="__main__":
    obj=DataIngestion()
    obj.initate_data_ingestion()
