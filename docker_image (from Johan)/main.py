import os
from typing import List, Union
from model_execution import logistic_regression
from fastapi import FastAPI
from oberije_2014_lung_survival import oberije_lung_survival

app = FastAPI()

def get_model():
    """
    If the model is specified in /app/model.json, load the model from the file. Otherwise, return the oberije_lung_survival object.
    """
    if os.path.exists("/app/model.json"):
        return logistic_regression(model_path="/app/model.json")
    return oberije_lung_survival()

@app.get("/")
def read_root():
    """
    Get the available models and their endpoints.
    """
    model_metadata = get_model().get_model_metadata()
    return {
        "models": [
                {
                    "model_uri": model_metadata["model_uri"],
                    "model_name": model_metadata["model_name"],
                    "path": "/predict",	
                    "path_parameters": "/input_parameters",
                }
            ]
    }

@app.post("/predict")
def predict(data: Union[dict, List[dict]]):
    """
    Calculate the probability for the current model.

    Parameters:
    - data: a dictionary (or list of dictionaries) containing the input data

    Returns:
    - probability: the probability which the model calculates
    """
    model_obj = get_model()
    return model_obj.predict(data)

@app.get("/input_parameters")
def get_input_parameters():
    """
    Get the input parameters of the model.

    Returns:
    - input_parameters: a list of input parameters
    """
    model_obj = get_model()
    return model_obj.get_input_parameters()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)