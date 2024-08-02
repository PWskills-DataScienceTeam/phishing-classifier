from datetime import datetime
import os

AWS_S3_BUCKET_NAME = "sensorpw"
MONGO_DATABASE_NAME = "phising"

TARGET_COLUMN = "Result"

MODEL_FILE_NAME = "model"
MODEL_FILE_EXTENSION = ".pkl"

artifact_folder_name = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
artifact_folder = os.path.join("artifacts", artifact_folder_name)
