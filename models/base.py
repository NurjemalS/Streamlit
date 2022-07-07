
from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd

class BaseModel(ABC):
    
    #def __init__(self, model_args: Dict):
        #self.model_args = model_args
    

    @abstractmethod
    def splitData(self, features:  pd.DataFrame, target: pd.DataFrame, size: float):
        """
            train test splitting, 
        """


    @abstractmethod
    def fit(self, train_x_df: pd.DataFrame, train_y_df: pd.DataFrame):
        """
        Abstract base method that trains the forecasting model.
        Should return the model instance

        """

    @abstractmethod
    def predict(self, model, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract base method that predicts on the trained forecasting model.
        Returns a pd.DataFrame instance

        :param model: trained prophet model instance
        :type model: Any
        :param test_df: test dataframe
        :type test_df: pd.DataFrame
        :return: predictions
        :rtype: pd.DataFrame
        """
    
    @staticmethod
    @abstractmethod
    def get_options() -> Dict[str, Any]:
        """
        Abstract base method that returns a dictionary of model parameters
        and their widget dictionaries that will be rendered

        :return: dictionary of model parameters and their widget
            dictionaries that will be rendered
        :rtype: Dict[str, Any]
        """


