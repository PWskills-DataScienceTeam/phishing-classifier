import shutil
import os,sys
import pandas as pd
from src.logger import logging

from src.exception import CustomException
import sys
from flask import request
from src.constant import *
from src.utils.main_utils import MainUtils

from dataclasses import dataclass
        
        
@dataclass
class PredictionFileDetail:
    prediction_output_dirname: str = "predictions"
    prediction_file_name:str =  "predicted_file.csv"
    prediction_file_path:str = os.path.join(prediction_output_dirname,prediction_file_name)



class PredictionPipeline:
    def __init__(self, request: request):

        self.request = request
        self.utils = MainUtils()
        self.prediction_file_detail = PredictionFileDetail()



    def save_input_files(self)-> str:

        """
            Method Name :   save_input_files
            Description :   This method saves the input file to the prediction artifacts directory. 
            
            Output      :   input dataframe
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """

        try:
            pred_file_input_dir = "prediction_artifacts"
            os.makedirs(pred_file_input_dir, exist_ok=True)

            input_csv_file = self.request.files['file']
            pred_file_path = os.path.join(pred_file_input_dir, input_csv_file.filename)
            
            
            input_csv_file.save(pred_file_path)


            return pred_file_path
        except Exception as e:
            raise CustomException(e,sys)

    def predict(self, features):
            try:
                model_path = self.utils.download_model(
                    bucket_name=AWS_S3_BUCKET_NAME,
                    bucket_file_name="model.pkl",
                    dest_file_name="model.pkl",
                )

                model = self.utils.load_object(file_path=model_path)

                preds = model.predict(features)

                return preds

            except Exception as e:
                raise CustomException(e, sys)
        
    def get_predicted_dataframe(self, input_dataframe_path:pd.DataFrame):

        """
            Method Name :   get_predicted_dataframe
            Description :   this method returns the dataframw with a new column containing predictions

            
            Output      :   predicted dataframe
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """
   
        try:

            prediction_column_name : str = TARGET_COLUMN
            input_dataframe: pd.DataFrame = pd.read_csv(input_dataframe_path)
            
            predictions = self.predict(input_dataframe)
            input_dataframe[prediction_column_name] = [pred for pred in predictions]
            target_column_mapping = {0:'phising', 1:'safe'}

            input_dataframe[prediction_column_name] = input_dataframe[prediction_column_name].map(target_column_mapping)
            
            os.makedirs( self.prediction_file_detail.prediction_output_dirname, exist_ok= True)
            input_dataframe.to_csv(self.prediction_file_detail.prediction_file_path, index= False)
            logging.info("predictions completed. ")



        except Exception as e:
            raise CustomException(e, sys) from e
        

        
    def run_pipeline(self):
        try:
            input_csv_path = self.save_input_files()
            self.get_predicted_dataframe(input_csv_path)

            return self.prediction_file_detail


        except Exception as e:
            raise CustomException(e,sys)
            
        

 
        

        