import sys
from typing import List
import pandas as pd
import re
import os
import shutil
import json

from env.Lib.pathlib import Path
from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

LENGTH_OF_DATE_STAMP_IN_FILE = 8
LENGTH_OF_TIME_STAMP_IN_FILE = 6
NUMBER_OF_COLUMNS = 11


@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(artifact_folder, 'data_validation')
    valid_data_dir: str = os.path.join(data_validation_dir, 'validated')
    invalid_data_dir: str = os.path.join(data_validation_dir, 'invalid')
    schema_config_file_path: str = os.path.join('config', 'training_schema.json')


class DataValidation:
    def __init__(self,
                 raw_data_store_dir: Path):

        self.raw_data_store_dir = raw_data_store_dir
        self.data_validation_config = DataValidationConfig()

        self.utils = MainUtils()

    def valuesFromSchema(self):
        """
                        Method Name: valuesFromSchema
                        Description: This method extracts all the relevant information from the pre-defined "Schema" file.
                        Output: LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, Number of Columns
                        On Failure: Raise ValueError,KeyError,Exception

                         Written By: iNeuron Intelligence
                        Version: 1.0
                        Revisions: None

                                """
        try:
            with open(self.data_validation_config.schema_config_file_path, 'r') as f:
                dic = json.load(f)
                f.close()
            LengthOfDateStampInFile = dic['LengthOfDateStampInFile']
            LengthOfTimeStampInFile = dic['LengthOfTimeStampInFile']
            column_names = dic['ColName']
            number_of_columns = dic['NumberofColumns']

            return LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, number_of_columns

        except Exception as e:
            raise CustomException(e, sys)

    def validate_file_name(self,
                           file_path: str,
                           length_of_date_stamp: int,
                           length_of_time_stamp: int) -> bool:
        """
            Method Name :   validate_file_columns
            Description :   This method validates the file name for a particular raw file 
            
            Output      :   True or False value is returned based on the schema 
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """
        try:

            file_name = os.path.basename(file_path)
            regex = "['phising']+['\_'']+[\d_]+[\d]+\.csv"

            if re.match(regex, file_name):
                splitAtDot = re.split('.csv', file_name)
                splitAtDot = (re.split('_', splitAtDot[0]))
                filename_validation_status = len(splitAtDot[1]) == length_of_date_stamp and len(
                    splitAtDot[2]) == length_of_time_stamp
            else:
                filename_validation_status = False

            return filename_validation_status

        except Exception as e:
            raise CustomException(e, sys)

    def validate_no_of_columns(self, file_path: str,
                               schema_no_of_columns: int) -> bool:
        """
            Method Name :   validate_column_columns
            Description :   This method validates the number of columns for a particular raw file
            
            Output      :   True or False value is returned based on the schema 
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """

        try:
            dataframe = pd.read_csv(file_path)
            column_length_validation_status = len(dataframe.columns) == schema_no_of_columns

            return column_length_validation_status

        except Exception as e:
            raise CustomException(e, sys)

    def validate_missing_values_in_whole_column(self, file_path: str) -> bool:
        """
            Method Name :   validate_missing_values_in_whole_column
            Description :   This method validates if there is any column in the csv file 
                            which has all the values as null.
            
            Output      :   True or False value is returned based on the condition 
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """

        try:

            dataframe = pd.read_csv(file_path)
            no_of_columns_with_whole_null_values = 0
            for columns in dataframe:
                if (len(dataframe[columns]) - dataframe[columns].count()) == len(
                        dataframe[columns]):  # checking null values
                    no_of_columns_with_whole_null_values += 1

            if no_of_columns_with_whole_null_values == 0:
                missing_value_validation_status = True
            else:
                missing_value_validation_status = False

            return missing_value_validation_status

        except Exception as e:
            raise CustomException(e, sys)

    def get_raw_batch_files_paths(self) -> List:
        """
            Method Name :   get_raw_batch_files_paths
            Description :   This method returns all the raw file dir paths in a list.
                            
            
            Output      :   List of dir paths
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """

        try:
            raw_batch_files_names = os.listdir(self.raw_data_store_dir)
            raw_batch_files_paths = [os.path.join(self.raw_data_store_dir, raw_batch_file_name) for raw_batch_file_name
                                     in raw_batch_files_names]
            return raw_batch_files_paths

        except Exception as e:
            raise CustomException(e, sys)

    def move_raw_files_to_validation_dir(self, src_path: str, dest_path: str):

        """
            Method Name :   move_raw_files_to_validation_dir
            Description :   This method moves validated raw files to the validated directory.
                            
            
            Output      :   NA
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """

        try:
            os.makedirs(dest_path, exist_ok=True)
            if os.path.basename(src_path) not in os.listdir(dest_path):
                shutil.move(src_path, dest_path)
        except Exception as e:
            raise CustomException(e, sys)

    def validate_raw_files(self) -> bool:
        """
            Method Name :   validate_raw_files
            Description :   This method validates the raw files for training.
                            
            
            Output      :   True or False value is returned based on the validated file number 

            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """

        try:
            raw_batch_files_paths = self.get_raw_batch_files_paths()
            length_of_date_stamp, length_of_time_stamp, column_names, no_of_column = self.valuesFromSchema()

            validated_files = 0
            for raw_file_path in raw_batch_files_paths:
                file_name_validation_status = self.validate_file_name(
                    raw_file_path,
                    length_of_date_stamp=length_of_date_stamp,
                    length_of_time_stamp=length_of_time_stamp
                )
                column_length_validation_status = self.validate_no_of_columns(
                    raw_file_path,
                    schema_no_of_columns=no_of_column)

                missing_value_validation_status = self.validate_missing_values_in_whole_column(raw_file_path)

                if (file_name_validation_status
                        and column_length_validation_status
                        and missing_value_validation_status):

                    validated_files += 1

                    self.move_raw_files_to_validation_dir(raw_file_path, self.data_validation_config.valid_data_dir)
                else:
                    self.move_raw_files_to_validation_dir(raw_file_path, self.data_validation_config.invalid_data_dir)

            validation_status = validated_files > 0

            return validation_status

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self):
        """
        Method Name :   initiate_data_validation
        Description :   This method initiates the data validation component for the pipeline
        
        Output      :   Returns data validation artifact
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info("Entered initiate_data_validation method of Data_Validation class")

        try:
            logging.info("Initiated data validation for the dataset")
            validation_status = self.validate_raw_files()

            if validation_status:
                valid_data_dir = self.data_validation_config.valid_data_dir
                return valid_data_dir
            else:
                raise Exception("No data could be validated. Pipeline stopped.")




        except Exception as e:
            raise CustomException(e, sys) from e