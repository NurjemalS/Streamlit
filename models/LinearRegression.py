from typing import Any, Dict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from models.base import BaseModel


class LinearRegression(BaseModel):

    def __init__(self, model_args: Dict[str, Any]):
        super().__init__(model_args)
        self.model = None

    def myName():
        return "linear"
    

    def splitData(self, features: pd.DataFrame, target: pd.DataFrame, size: float):
        X_train, X_test, y_train, y_test = train_test_split(features, target, size = size)
        return [X_train, X_test, y_train, y_test]
        


    def fit(self, train_x_df: pd.DataFrame,train_y_df: pd.DataFrame, **kwargs: Dict[str, Any]):
        model = LinearRegression(**kwargs)
        model = model.fit(train_x_df, train_y_df)
        self.model = model
        return model
    
    def predict(self, model, test_df: pd.DataFrame) -> pd.DataFrame:
        return model.predict(test_df)
    
   
        
      