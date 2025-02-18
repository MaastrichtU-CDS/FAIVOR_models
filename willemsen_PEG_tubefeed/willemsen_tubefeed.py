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
        with open('modeldata_PEG_tube_feeding.json') as f:

            try:

                ## check the values and perform the conversion of categorical variables

                # Tumor TNM finding validation
                # Extract allowed values from JSON
                allowed_values = set(f["Tclassification"]["values"].values())

                if isinstance(data, list):
                    for i in range(len(data)):
                        if "Tclassification" not in data[i]:
                            return {"error": f"Missing Tclassification in item {i}"}
                        if data[i]["Tclassification"] not in allowed_values:
                            return {
                                "error": f"Invalid Tclassification value in item {i}: {data[i]['Tclassification']}"}

                        # Apply transformation
                        data[i]['Tclassification'] = 1 if data[i]['Tclassification'] in [2, 3] else 0
                else:
                    if "Tclassification" not in data:
                        return {"error": "Missing Tclassification"}
                    if data["Tclassification"] not in allowed_values:
                        return {"error": f"Invalid Tclassification value: {data['Tclassification']}"}

                    # Apply transformation
                    data['Tclassification'] = 1 if data['Tclassification'] in [2, 3] else 0

                # Regional Lymph nodes TNM finding
                # Extract allowed values from JSON
                allowed_values = set(f["Nclassification"]["values"].values())

                if isinstance(data, list):
                    for i in range(len(data)):
                        if "Nclassification" not in data[i]:
                            return {"error": f"Missing Nclassification in item {i}"}
                        if data[i]["Nclassification"] not in allowed_values:
                            return {
                                "error": f"Invalid Nclassification value in item {i}: {data[i]['Nclassification']}"}

                        # Apply transformation
                        data[i]['Nclassification'] = 1 if data[i]['Nclassification'] in [2, 3] else 0
                else:
                    if "Nclassification" not in data:
                        return {"error": "Missing Nclassification"}
                    if data["Nclassification"] not in allowed_values:
                        return {"error": f"Invalid Nclassification value: {data['Nclassification']}"}

                    # Apply transformation
                    data['Nclassification'] = 1 if data['Nclassification'] in [2, 3] else 0

                # WHO performance status finding
                # Extract allowed values from JSON
                allowed_values = set(f["PS"]["values"].values())

                if isinstance(data, list):
                    for i in range(len(data)):
                        if "PS" not in data[i]:
                            return {"error": f"Missing PS in item {i}"}
                        if data[i]["PS"] not in allowed_values:
                            return {
                                "error": f"Invalid PS value in item {i}: {data[i]['PS']}"}

                        # Apply transformation
                        data[i]['PS'] = 1 if data[i]['PS'] >0 else 0
                else:
                    if "PS" not in data:
                        return {"error": "Missing PS"}
                    if data["PS"] not in allowed_values:
                        return {"error": f"Invalid PS value: {data['PS']}"}

                    # Apply transformation
                    data['PS'] = 1 if data['PS'] > 0  else 0

                # Systemic therapy
                allowed_values = set(f["Systherapy"]["values"].values())

                if isinstance(data, list):
                    for i in range(len(data)):
                        if "Systherapy" not in data[i]:
                            return {"error": f"Missing Systherapy in item {i}"}
                        if data[i]["Systherapy"] not in allowed_values:
                            return {
                                "error": f"Invalid Systherapy value in item {i}: {data[i]['Systherapy']}"}

                else:
                    if "Systherapy" not in data:
                        return {"error": "Missing Systherapy"}
                    if data["Systherapy"] not in allowed_values:
                        return {"error": f"Invalid Systherapy value: {data['Systherapy']}"}

                # Texture modified diet or tube feeding
                allowed_values = set(f["TF"]["values"].values())

                if isinstance(data, list):
                    for i in range(len(data)):
                        if "TF" not in data[i]:
                            return {"error": f"Missing TF in item {i}"}
                        if data[i]["TF"] not in allowed_values:
                            return {
                                "error": f"Invalid TF value in item {i}: {data[i]['TF']}"}

                else:
                    if "TF" not in data:
                        return {"error": "Missing TF"}
                    if data["TF"] not in allowed_values:
                        return {"error": f"Invalid TF value: {data['TF']}"}

                # Tumorlocation
                allowed_values = set(f["Tumorlocation"]["values"].values())

                if isinstance(data, list):
                    for i in range(len(data)):
                        if "Tumorlocation" not in data[i]:
                            return {"error": f"Missing Tumorlocation in item {i}"}
                        if data[i]["Tumorlocation"] not in allowed_values:
                            return {
                                "error": f"Invalid Tumorlocation value in item {i}: {data[i]['Tumorlocation']}"}

                else:
                    if "Tumorlocation" not in data:
                        return {"error": "Missing Tumorlocation"}
                    if data["Tumorlocation"] not in allowed_values:
                        return {"error": f"Invalid Tumorlocation value: {data['Tumorlocation']}"}

                ## check the values in numerical variables

                # BMI
                min=set(f["BMI"]["min"].values())
                max = set(f["BMI"]["max"].values())

                if isinstance(data, list):
                    for i in range(len(data)):
                        if "BMI" not in data[i]:
                            return {"error": f"Missing BMI in item {i}"}
                        if not isinstance(data[i]["BMI"], (int, float)):
                            return {"error": f"Invalid BMI type in item {i}, expected a number"}
                        if not (min<data[i]["BMI"] < max):
                            return {
                                "error": f"Invalid BMI value in item {i}: {data[i]['BMI']}"}

                else:
                    if "BMI" not in data:
                        return {"error": "Missing BMI"}
                    if not isinstance(data["BMI"], (int, float)):
                        return {"error": f"Invalid BMI type, expected a number"}
                    if not (min<data["BMI"]<max):
                        return {"error": f"Invalid BMI value: {data['BMI']}"}

                # WeightLoss
                min = set(f["WeightLoss"]["min"].values())
                max = set(f["WeightLoss"]["max"].values())

                if isinstance(data, list):
                    for i in range(len(data)):
                        if "WeightLoss" not in data[i]:
                            return {"error": f"Missing WeightLoss in item {i}"}
                        if not isinstance(data[i]["WeightLoss"], (int, float)):
                            return {"error": f"Invalid WeightLoss type in item {i}, expected a number"}
                        if not (min < data[i]["WeightLoss"] < max):
                            return {
                                "error": f"Invalid WeightLoss value in item {i}: {data[i]['WeightLoss']}"}

                else:
                    if "WeightLoss" not in data:
                        return {"error": "Missing WeightLoss"}
                    if not isinstance(data["WeightLoss"], (int, float)):
                        return {"error": f"Invalid WeightLoss type, expected a number"}
                    if not (min < data["WeightLoss"] < max):
                        return {"error": f"Invalid WeightLoss value: {data['WeightLoss']}"}

                # RTdose_subman
                min = set(f["RTdose_subman"]["min"].values())
                max = set(f["RTdose_subman"]["max"].values())

                if isinstance(data, list):
                    for i in range(len(data)):
                        if "RTdose_subman" not in data[i]:
                            return {"error": f"Missing RTdose_subman in item {i}"}
                        if not isinstance(data[i]["RTdose_subman"], (int, float)):
                            return {"error": f"Invalid RTdose_subman type in item {i}, expected a number"}
                        if not (min < data[i]["RTdose_subman"] < max):
                            return {
                                "error": f"Invalid RTdose_subman value in item {i}: {data[i]['RTdose_subman']}"}

                else:
                    if "RTdose_subman" not in data:
                        return {"error": "Missing RTdose_subman"}
                    if not isinstance(data["RTdose_subman"], (int, float)):
                        return {"error": f"Invalid RTdose_subman type, expected a number"}
                    if not (min < data["RTdose_subman"] < max):
                        return {"error": f"Invalid RTdose_subman value: {data['RTdose_subman']}"}

                # RTdosesalivary
                min = set(f["RTdosesalivary"]["min"].values())
                max = set(f["RTdosesalivary"]["max"].values())

                if isinstance(data, list):
                    for i in range(len(data)):
                        if "RTdosesalivary" not in data[i]:
                            return {"error": f"Missing RTdosesalivary in item {i}"}
                        if not isinstance(data[i]["RTdosesalivary"], (int, float)):
                            return {"error": f"Invalid RTdosesalivary type in item {i}, expected a number"}
                        if not (min < data[i]["RTdosesalivary"] < max):
                            return {
                                "error": f"Invalid RTdosesalivary value in item {i}: {data[i]['RTdosesalivary']}"}

                else:
                    if "RTdosesalivary" not in data:
                        return {"error": "Missing RTdosesalivary"}
                    if not isinstance(data["RTdosesalivary"], (int, float)):
                        return {"error": f"Invalid RTdosesalivary type, expected a number"}
                    if not (min < data["RTdosesalivary"] < max):
                        return {"error": f"Invalid RTdosesalivary value: {data['RTdosesalivary']}"}

                return data

            except Exception as e:

                return {"error": f"Unexpected error: {str(e)}"}

        # perform log transformation on the gtv value which is in the data list/dictionary
        # if isinstance(data, list):
        #     for i in range(len(data)):
        #         data[i]['gtv'] = log(data[i]['gtv'])
        # else:
        #     data['gtv'] = log(data['gtv'])


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