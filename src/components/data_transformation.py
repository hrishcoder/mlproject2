import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object


@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation():
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_column=['carat','depth','table','x','y','z']#all numerical data
            categorical_column=['cut','color']#all categorical data


            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore')),
                    ('scaler',StandardScaler(with_mean=False)),
                ]
            )

            logging.info(f"categorical columns:{categorical_column}")
            logging.info(f"Numerical columns:{numerical_column}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_column),
                    ("cat_pipeline",cat_pipeline,categorical_column)
                ]
            )


            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def  initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessor object")

            preprocessor_obj=self.get_data_transformer_object()
            deleted_column_name='clarity'
            target_column_name='price'

            input_feature_train=train_df.drop(columns=[deleted_column_name,target_column_name],axis=1)
            output_feature_train=train_df.drop(columns=[target_column_name])

            input_feature_test=test_df.drop(columns=[deleted_column_name,target_column_name],axis=1)
            output_feature_test=test_df.drop(columns=[target_column_name])

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test)

            train_arr=np.c_[input_feature_train_arr,np.array(output_feature_train)]
            test_arr=np.c_[input_feature_test_arr,np.array(output_feature_test)]

            logging.info(f"saved preprocessing object")
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path, 
                obj=preprocessor_obj

            )

            return(train_arr,
                   test_arr,
                   self.data_transformation_config.preprocessor_obj_file_path)


            
        except Exception as e:
            raise CustomException(e,sys)