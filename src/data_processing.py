import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest,chi2
from src.logger import get_logger
from src.custom_exception import CustomException
import sys
logger = get_logger(__name__)

class DataProccessing:
    def __init__(self,input_path,output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.label_encoders={}
        self.scaler=StandardScaler()
        self.df =None
        self.X = None
        self.y=None
        self.selected_features =[]

        os.makedirs(output_path,exist_ok=True)
        logger.info("Data processing intialized")

    def load_data(self):
        try:
            self.df =pd.read_csv(self.input_path)
            logger.info("Data loaded successfully")
        except Exception as e:
            logger.error(f"Error while loading the data",e)
            raise CustomException("Failed to load data")
    
    def preprocess_data(self):
        try:
            self.df.drop(columns=['Patient_ID'],inplace=True)
            self.X = self.df.drop(columns=['Survival_Prediction'])
            self.y = self.df[['Survival_Prediction']]

            categorical_cols = self.X.select_dtypes(include=['object']).columns
            self.label_encoders ={}

            for col in categorical_cols:
                le=LabelEncoder()
                self.X[col]=le.fit_transform(self.X[col])
                self.label_encoders[col] = [le.classes_,self.X[col].unique()]
                
            logger.info("Basic Processing done !!")

        except Exception as e:
            logger.error(f"Error while Processing the data",e)
            raise CustomException(f"Failed to Processing data",sys)
    
    def feature_selection(self):
        try:
            X_train,_,y_train,_ =train_test_split(self.X,self.y,random_state=42,test_size=0.2)
            x_cat = X_train.select_dtypes(include=['int64','float64'])
            chi2_selector = SelectKBest(score_func=chi2,k="all")
            chi2_selector.fit(x_cat,y_train)

            chi2_scores = pd.DataFrame({
                "Feature":x_cat.columns,
                "Chi2 Square":chi2_selector.scores_
            }).sort_values(by='Chi2 Square',ascending=False)

            self.selected_features = chi2_scores.head(5)['Feature'].to_list()
            logger.info(f"selected feature are")

            self.X=self.X[self.selected_features]
            logger.info("Feature selection completed !")
        except Exception as e:
            logger.error(f"Error while feature selection the data",e)
            raise CustomException("Failed to feature selection data")
    

    def split_and_scale_data(self):
        try:
            X_train,X_test,y_train,y_test =train_test_split(self.X,self.y,random_state=42,test_size=0.2,stratify=self.y)
            X_train=self.scaler.fit_transform(X_train)
            X_test =self.scaler.fit_transform(X_test)

            logger.info("splitting and scaling Done!")
            return X_train,X_test,y_train,y_test
        except Exception as e:
            logger.error(f"Error while splitting and scaling the data",e)
            raise CustomException("Failed to splitting and scaling data")
        

    def save_data_scaler(self,X_train,X_test,y_train,y_test):
        try:
            joblib.dump(X_train,os.path.join(self.output_path,"X_train.pkl"))
            joblib.dump(X_test,os.path.join(self.output_path,"X_test.pkl"))
            joblib.dump(y_train,os.path.join(self.output_path,"y_train.pkl"))
            joblib.dump(y_test,os.path.join(self.output_path,"y_test.pkl"))

            joblib.dump(self.scaler,os.path.join(self.output_path,"scaler.pkl"))
            logger.info("save_data_scaler completed !")
        except Exception as e:
            logger.error(f"Error while save_data_scaler the data",e)
            raise CustomException("Failed to save_data_scaler data")
        
    def run(self):
        self.load_data()
        self.preprocess_data()
        self.feature_selection()
        X_train,X_test,y_train,y_test = self.split_and_scale_data()
        self.save_data_scaler(X_train,X_test,y_train,y_test)

        logger.info("Data processing pipleline executed successfully!!")


if __name__ =="__main__":
    input_path = 'artifacts/raw/data.csv'
    output_path = 'artifacts/processed'

    processor= DataProccessing(input_path,output_path)
    processor.run()
  