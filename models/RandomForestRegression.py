from email.policy import default
from typing import Any, Dict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from models.base import BaseModel



class RandomForestRegression(BaseModel):

   # def __init__(self, model_args: Dict[str, Any]):
        #super().__init__(model_args)
        #self.model = None

    def myName():
        return "RandomForestRegression"
    
    def splitData(self, features: pd.DataFrame, target: pd.DataFrame, size: float):
        X_train, X_test, y_train, y_test = train_test_split(features, target, train_size = size)
        return [X_train, X_test, y_train, y_test]
        
    def fit(self, train_x_df: pd.DataFrame,train_y_df: pd.DataFrame, **kwargs: Dict[str, Any]):
        model = RandomForestRegressor(**kwargs)
        model = model.fit(train_x_df, train_y_df)
        self.model = model
        return model
    
    def predict(self, model, test_df: pd.DataFrame) -> pd.DataFrame:
        return model.predict(test_df)
    
    def get_options():
        return {
           "max_depth" : {
               "label" : "max_depth",
               "value" : 20,
               "min_value" : 10,
               "max_value" : 100,
               "step": 10,
               "type" : "slider"
           },
           "ccp_alpha" :{
               "label" : "ccp_alpha",
               "value" : 0.02,
               "min_value" : 0.01,
               "max_value" : 0.1,
               "step": 0.01,
               "type" : "slider"
           },
           "n_estimators" :{
                "label" : "n_estimators",
                "value" : [50, 100, 150, 200, 250, 'No limit'],
                "type" : "selectbox"
           },
           "random_state" :{
                "label" : "random_state",
                "value" : [50, 100, 150, 200, 250, 'No limit'],
                "type" : "selectbox"
           },
           "criterion" :{
                "label" : "criterion",
                "value" : ["squared_error", "absolute_error", "poisson"],
                "type" : "selectbox"
           },
           "bootstrap" : {
                "label" : "bootstrap",
                "value" : [True, False],
                "type" : "selectbox"
           }, 
           "n_jobs" :{
                "label" : "n_jobs",
                "value" : [10, 20, 30, 40, 50, 'None'],
                "type" : "selectbox"
           }
        }
    
   
        
      