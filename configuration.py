from abc import ABC, abstractmethod
from typing import Any, Dict
from models import LinearRegression, RandomForestRegression

MODELS = {
    "LinearRegression": {
        "model_class": LinearRegression,
        "explainers": [
            "LinearExplainer",
            "AlibiALEExplainer",
            "LIME",
        ]
    },
    "RandomForestRegression": {
        "model_class": RandomForestRegression,
        "default_parameters": {
            "fit_intercept" : "100",
            "normalize" : "None",
            "n_jobs" : "squared_error",
            "positive" : "True"
        },
        "explainers": [
            "ShapTreeExplainer",
            "AlibiALEExplainer"
        ]
    }
}



def get_models():
    return MODELS.keys()



class Explainer:
    
    def __init__(self):
        pass
    
    
class ShapTreeExplainer(Explainer):
    
    def explain(self, model, x_train):
        
        return explanation
        
        
        

