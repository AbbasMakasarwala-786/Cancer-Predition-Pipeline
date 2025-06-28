import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *

import mlflow
import mlflow.sklearn

logger=get_logger(__name__)

class ModelTraining:
    def __init__(self):
        self.processed_data_path = PROCESSED_DIR
        self.model_dir= MODEL_DIR
        os.makedirs(self.model_dir,exist_ok=True)

        logger.info("model training intialization")

    
    def load_data(self):
        try:
            self.X_train = joblib.load(X_TRAIN_ARRAY)
            self.X_test = joblib.load(X_TEST_ARRAY)
            self.y_train = joblib.load(Y_TRAIN)
            self.y_test = joblib.load(Y_TEST)

            logger.info("Data loaded for model")
        except Exception as e:
            logger.error(f"Error while loading the model")
            raise CustomException("Failed to load data for the model")
    
    def train_model(self):
        try: 
            self.model= GradientBoostingClassifier(n_estimators=10,learning_rate=0.01,max_depth=4,random_state=42)
            self.model.fit(self.X_train,self.y_train)

            joblib.dump(self.model,os.path.join(MODEL_DIR,"model.pkl"))
            logger.info("Model trained and saved !!")
        except Exception as e:
            logger.error(f"Error while Model trained and saved the model",e)
            raise CustomException("Failed to Model trained and saved for the model")
        
    
    def evaluate(self):
        try:
            y_pred  =self.model.predict(self.X_test)
            y_proba = self.model.predict_proba(self.X_test)[ : ,1] 

            accuracy = accuracy_score(self.y_test,y_pred) 
            precision = precision_score(self.y_test,y_pred,average="weighted")
            recall = recall_score(self.y_test,y_pred,average="weighted")
            f1 = f1_score(self.y_test,y_pred,average="weighted")

            mlflow.log_metric("accuracy",accuracy)
            mlflow.log_metric("Precision",precision)
            mlflow.log_metric("Recall Score",recall)
            mlflow.log_metric("F1 Score",f1)

            logger.info(f"Accracy:{accuracy}; Precision:{precision}; Recall:{recall}; F1_score: {f1}")
            roc_auc = roc_auc_score(self.y_test,y_proba)
            mlflow.log_metric("roc-auc",roc_auc)

            logger.info(f"Roc-Auc-Score : {roc_auc}")
            logger.info("Model eval completed")

        except Exception as e:
            logger.error(f"Error while MModel eval",e)
            raise CustomException("Failed to Model eval")
    
    def run(self):
        self.load_data()
        self.train_model()
        self.evaluate()


if __name__ == "__main__":
    with mlflow.start_run():
        model_training = ModelTraining()
        model_training.run()
        