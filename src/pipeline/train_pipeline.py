import pathlib
import sys, os

import numpy as np

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException


class TrainingPipeline:

    def start_data_ingestion(self):
        try:
            data_ingestion = DataIngestion()
            raw_data_dir = data_ingestion.initiate_data_ingestion()
            return raw_data_dir

        except Exception as e:
            raise CustomException(e, sys)

    def start_data_validation(self, raw_data_dir):
        try:
            data_validation = DataValidation(raw_data_store_dir=raw_data_dir)
            valid_data_dir = data_validation.initiate_data_validation()
            return valid_data_dir

        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transformation(self, valid_data_dir):
        try:
            data_transformation = DataTransformation(valid_data_dir=valid_data_dir)
            x_train, y_train, x_test, y_test, preprocessor_path = data_transformation.initiate_data_transformation()
            return x_train, y_train, x_test, y_test, preprocessor_path

        except Exception as e:
            raise CustomException(e, sys)

    def start_model_training(self,
                             x_train: np.array,
                             y_train: np.array,
                             x_test: np.array,
                             y_test: np.array,
                             preprocessor_path: pathlib.Path):
        try:
            model_trainer = ModelTrainer()
            model_score = model_trainer.initiate_model_trainer(
                x_train,
                y_train,
                x_test,
                y_test,
                preprocessor_path)

            return model_score

        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            raw_data_dir = self.start_data_ingestion()
            valid_data_dir = self.start_data_validation(raw_data_dir)
            x_train, y_train, x_test, y_test, preprocessor_path = self.start_data_transformation(valid_data_dir)
            r2_square = self.start_model_training(x_train, y_train, x_test, y_test, preprocessor_path)

            print("training completed. Trained model score : ", r2_square)

        except Exception as e:
            raise CustomException(e, sys)
