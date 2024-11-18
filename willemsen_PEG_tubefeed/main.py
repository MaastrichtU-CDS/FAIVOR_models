import os
from typing import List, Union
from fastapi import FastAPI

app = FastAPI()

def get_model():
    """
    Get the model object based on the environment variables.

    Returns:
    - instance: the model object
    """
    module_name = os.environ.get("MODULE_NAME")
    class_name = os.environ.get("CLASS_NAME")

    # import the module
    module = __import__(module_name)
    class_ = getattr(module, class_name)
    instance = class_()
    return instance

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