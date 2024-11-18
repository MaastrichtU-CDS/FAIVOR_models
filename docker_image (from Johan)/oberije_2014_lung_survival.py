
from math import log, exp
from model_execution import logistic_regression
import json

class oberije_lung_survival(logistic_regression):
    def __init__(self):
        self._model_parameters = {
            "model_uri": "https://cancerdata.org/id/10.5072/candat.2015.02",
            "model_name": "Oberije survival prediction model for lung cancer patients",
            "intercept": -0.5,
            "covariate_weights": {
                "gender": 0.8325,
                "who": -0.4395,
                "fev1": 0.0056,
                "lymph": -0.28002,
                "gtv": -0.7746
            }
        }
    
    def _preprocess(self, data):
        """
        This function is used to convert the input data into the correct format for the model.

        Parameters:
        - input_object: a dictionary, or list with multiple dictionaries, containing the input data

        Returns:
        - preprocessed_data: a dictionary, or list with multiple dictionaries, containing the preprocessed data
        """

        # perform log transformation on the gtv value which is in the data list/dictionary
        if isinstance(data, list):
            for i in range(len(data)):
                data[i]['gtv'] = log(data[i]['gtv'])
        else:
            data['gtv'] = log(data['gtv'])
        
        return data

if __name__ == "__main__":
    model_obj = oberije_lung_survival()
    print(model_obj.get_input_parameters())
    print(model_obj.predict(
        {
            "gender": 0,
            "who": 1,
            "fev1": 2,
            "lymph": 3,
            "gtv": 800
        }
    ))