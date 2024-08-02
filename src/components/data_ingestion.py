import sys
import os
import numpy as np
import pandas as pd
from pymongo import MongoClient
from zipfile import Path
from src.constant import *
from src.exception import CustomException
from src.logger import logging

from src.data_access.phising_data import PhisingData
from src.utils.main_utils import MainUtils
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(artifact_folder, "data_ingestion")


class DataIngestion:
    def __init__(self):

        self.data_ingestion_config = DataIngestionConfig()
        self.utils = MainUtils()

    def export_data_into_raw_data_dir(self) -> pd.DataFrame:
        """
        Method Name :   export_data_into_feature_store
        Description :   This method reads data from mongodb and saves it into artifacts. 
        
        Output      :   dataset is returned as a pd.DataFrame
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   0.1
       
        """
        try:
            logging.info(f"Exporting data from mongodb")
            raw_batch_files_path = self.data_ingestion_config.data_ingestion_dir
            os.makedirs(raw_batch_files_path, exist_ok=True)

            income_data = PhisingData(
                database_name=MONGO_DATABASE_NAME)

            logging.info(f"Saving exported data into feature store file path: {raw_batch_files_path}")
            for collection_name, dataset in income_data.export_collections_as_dataframe():
                logging.info(f"Shape of {collection_name}: {dataset.shape}")
                feature_store_file_path = os.path.join(raw_batch_files_path, collection_name + '.csv')
                print(f"feature_store_file_path-----{feature_store_file_path}")
                # dataset.rename(columns={"Unnamed: 0": "Wafer"}, inplace=True)
                dataset.to_csv(feature_store_file_path, index=False)


        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> Path:
        """
            Method Name :   initiate_data_ingestion
            Description :   This method initiates the data ingestion components of training pipeline 
            
            Output      :   train set and test set are returned as the artifacts of data ingestion components
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

        try:
            self.export_data_into_raw_data_dir()

            logging.info("Got the data from mongodb")

            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )

            return self.data_ingestion_config.data_ingestion_dir

        except Exception as e:
            raise CustomException(e, sys) from e
