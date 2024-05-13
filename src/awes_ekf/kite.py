# This file contains the class definition for the KiteModel class
from model_definitions import kite_models

class Kite:
    def __init__(self, model_name):
        """"Create kite model class from model name and model dictionary"""
        if model_name in kite_models:
            model_params = kite_models[model_name]
            for key, value in model_params.items():
                # Set each key-value pair as an attribute of the instance
                setattr(self, key, value)
        else:
            raise ValueError("Invalid kite model, add specs into kite_models dictionary")


