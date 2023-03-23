import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

from HotelBooking.exception import CustomException
from HotelBooking.logger import logging
from HotelBooking.utils import save_object

@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")



class DataTransformation():
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation.
        '''
        try:
            #Handling Missing  values
            df = pd.read_csv("notebook\data\hotel_bookings.csv")
            df  = df.drop(columns=['agent','company']) #droping columns 

            #filling the missing values with most occurance
            df['country'].fillna(df['country'].value_counts().index[0],inplace=True)

            #rest filling with 0
            df.fillna(0, inplace=True)
            logging.info("Done with Missing values")

            #Removing dirtiness from data
            filter = (df['children']==0) & (df['adults']==0) & (df['babies']==0)
            data = df[~filter]
            logging.info("Remove wrong data")

            #numerical and categorical data
            #numerical
            list_not = ['days_in_waiting_list','arrival_date_year']
            num_features = [col for col in data.columns if data[col].dtype != 'O' and col not in list_not]
            #Categorical
            cat_not= ['arrival_date_year','assigned_room_type','booking_changes','reservation_status','country','days_in_waiting_list']
            cat_features = [col for col in data.columns if data[col].dtype == 'O' and col not in cat_not]
            logging.info("Numerical and categorical features")

            #Creating dataframe of categorical features for feature encoding
            data_cat = data[cat_features]
            #reservation_status_date  datatype is object by default but we have to convert it into datetime to extract dates.
            data_cat['reservation_status_date'] = pd.to_datetime(data_cat['reservation_status_date'])
            #creating different columns for month year and day
            data_cat['year'] = data_cat['reservation_status_date'].dt.year
            data_cat['month'] = data_cat['reservation_status_date'].dt.month
            data_cat['day'] = data_cat['reservation_status_date'].dt.day
            #droping reservation_status_date
            data_cat.drop("reservation_status_date", axis=1, inplace=True)
            logging.info("done with categorical features")

            #Feature encoding
            data_cat['cancellation'] = data['is_canceled']
            cols = data_cat.columns[0:8]
            for col in cols:
                dict = data_cat.groupby([col])['cancellation'].mean().to_dict()
                data_cat[col] = data_cat[col].map(dict)
            
            logging.info("Feature encoding complete")
            
            

            dataframe = pd.concat([data_cat,data[num_features]], axis=1)
            dataframe.drop("cancellation", axis=1, inplace=True)

            #outlier removing
            def handel_outlier(col):
                dataframe[col] = np.log1p(dataframe[col])

            handel_outlier('lead_time')
            handel_outlier('adr')
            logging.info("Outlier convertion successful")

            ## Independent and Dependent features
            
            dataframe.dropna(inplace=True)

            x = dataframe.drop('is_canceled', axis=1)
            y = dataframe['is_canceled']

            preprocessor = SelectFromModel(Lasso(alpha=0.005, random_state=0))
            preprocessor = preprocessor.fit(x,y)



            return preprocessor

            
        except Exception as e:
            raise CustomException(e,sys)
            
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test data completed")

            logging.info("obtaining preprocessing object")

            preprocessing_obj  = self.get_data_transformer_object()

            target_column_name = "is_canceled"

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Appling preprocessing object on training dataframe and testing dataframe"
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df, 'y')
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
                ]
            
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info(f"Saved preprocessing object")

            save_object(

                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path

            )

        except Exception as e:
            raise CustomException(e,sys)
            