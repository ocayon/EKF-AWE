# This file contains the class definition for the KiteModel class
class Kite:
    def __init__(self, **kwargs):
        self.mass = kwargs.get('mass')
        self.area = kwargs.get('area')
        self.span = kwargs.get('span')
        self.model_name = kwargs.get('model_name')
        


