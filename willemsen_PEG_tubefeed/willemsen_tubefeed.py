from math import log, exp
from model_execution import logistic_regression
import json

class willemsem_tubefeed(logistic_regression):
    def __init__(self):
        with open('modeldata_PEG_tube_feeding.json') as f:
            self._model_parameters = json.load(f)
        # json. {
        #     "model_uri": "https://cancerdata.org/id/10.5072/candat.2015.02",
        #     "model_name": "Oberije survival prediction model for lung cancer patients",
        #     "intercept": -0.5,
        #     "covariate_weights": {
        #         "gender": 0.8325,
        #         "who": -0.4395,
        #         "fev1": 0.0056,
        #         "lymph": -0.28002,
        #         "gtv": -0.7746
        #     }
        # }

    def _preprocess(self, data):
        """
        This function is used to convert the input data into the correct format for the model.

        Parameters:
        - input_object: a dictionary, or list with multiple dictionaries, containing the input data

        Returns:
        - preprocessed_data: a dictionary, or list with multiple dictionaries, containing the preprocessed data
        """

        # perform log transformation on the gtv value which is in the data list/dictionary
        # if isinstance(data, list):
        #     for i in range(len(data)):
        #         data[i]['gtv'] = log(data[i]['gtv'])
        # else:
        #     data['gtv'] = log(data['gtv'])

        return data


if __name__ == "__main__":
    model_obj = willemsem_tubefeed()
    print(model_obj.get_input_parameters())
    print(model_obj.predict(
        {
            "BMI": 19.5,
            "WeightLoss": -8,
            "TF": 1,
            "PS": 0,
            "Tumorlocation": 1,
            "Tclassification": 1,
            "Nclassification": 1,
            "Systherapy": 0,
            "RTdose_subman": 36,
            "RTdosesalivary": 29
        }
    ))