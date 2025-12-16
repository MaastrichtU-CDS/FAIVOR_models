from model_execution import logistic_regression
from typing import Tuple, Any, Dict, List, Optional


def get_range(metadata: Dict[str, Any], feature: str) -> Tuple[float, float]:
    """
    Return (min_value, max_value) for a numerical feature from metadata.

    Raises:
    - ValueError if the feature is not found or min/max cannot be parsed to float.
    """
    # Find the entry where "Description" "@value" equals the feature name
    entry = next(
        (entry for entry in metadata.get("Input data", []) if entry.get("Description", {}).get("@value") == feature),
        None,
    )

    if not entry:
        raise ValueError(f"No range metadata found for feature '{feature}'")

    try:
        min_value = float(entry.get("Minimum - for numerical", {}).get("@value", "0"))
        max_value = float(entry.get("Maximum - for numerical", {}).get("@value", "0"))
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid min/max values for feature '{feature}': {e}")

    return min_value, max_value


def get_categorical_values(metadata: Dict[str, Any], description_value: str):
    """
    Return a list of category identifications for a categorical feature description.
    Returns None if no matching entry or no categories found.
    """
    category_entry = next(
        (entry for entry in metadata.get("Input data", []) if isinstance(entry, dict) and entry.get("Description", {}).get("@value") == description_value),
        None,
    )
    if not category_entry:
        return None

    categories = category_entry.get("Categories", [])
    categories_id = [
        category.get("Identification for category used in model", {}).get("@value")
        for category in categories
        if isinstance(category, dict)
    ]

    return categories_id or None


def validate_numerical_feature(data: Any, feature: str, min_value: float, max_value: float) -> bool:
  """
  Validates a numerical feature in a dictionary or list of dictionaries.

  Parameters:
  - data: dict or list of dicts containing the input data
  - feature: str, the name of the feature to validate
  - min_value: float, minimum allowed value (must not be None)
  - max_value: float, maximum allowed value (must not be None)

  Raises:
  - ValueError if the feature is missing or out of range
  - TypeError if the feature is not a number (rejects bools)
  """
  # Guard against missing range bounds
  if min_value is None or max_value is None:
    raise ValueError(f"Allowed range for feature '{feature}' is not available")

  def _check_value(val: Any, idx: Optional[int] = None):
    label = f"item {idx}" if idx is not None else "object"
    # Reject booleans explicitly: isinstance(True, int) == True, so check bool first
    if isinstance(val, bool) or not isinstance(val, (int, float)):
      raise TypeError(f"Invalid {feature} type in {label}, expected a number")
    if not (min_value <= val <= max_value):
      raise ValueError(f"Invalid {feature} value in {label}: {val} (Allowed range: {min_value}-{max_value})")

  if isinstance(data, list):
    for i, item in enumerate(data):
      if not isinstance(item, dict):
        raise TypeError(f"Invalid input at item {i}: expected a dict")
      if feature not in item:
        raise ValueError(f"Missing {feature} in item {i}")
      # safe get, avoid raising KeyError
      val = item.get(feature)
      _check_value(val, i)
  else:
    if not isinstance(data, dict):
      raise TypeError("Input data must be a dict or list of dicts")
    if feature not in data:
      raise ValueError(f"Missing {feature}")
    val = data.get(feature)
    _check_value(val, None)

  return True

class stiphout_pCR_Clinical(logistic_regression):
    def __init__(self):
        #with open('willemsen_tubefeed.json') as f:
        self._model_parameters = {
    #Part of metadata that is only inside the docker container
    "intercept": -0.6,
    "covariate_weights": {
        "cT": -0.074,
        "cN": -0.060,
        "tLength": -0.085
    },
            "model_type": "logistic_regression",
            "model_uri": "https://v2.fairmodels.org/instance/3f400afb-df5e-4798-ad50-0687dd439d9b",
            "model_name": "Stiphout pCR prediction - clinical parameters",
    }

    def _preprocess(self, data):
        """
        This function is used to convert the input data into the correct format for the model.

        Parameters:
        - input_object: a dictionary, or list with multiple dictionaries, containing the input data

        Returns:
        - preprocessed_data: a dictionary, or list with multiple dictionaries, containing the preprocessed data
        """
        # fetched metadata from faif
        f= {
          "@context": {
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "pav": "http://purl.org/pav/",
            "schema": "http://schema.org/",
            "oslc": "http://open-services.net/ns/core#",
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "rdfs:label": {
              "@type": "xsd:string"
            },
            "schema:isBasedOn": {
              "@type": "@id"
            },
            "schema:name": {
              "@type": "xsd:string"
            },
            "schema:description": {
              "@type": "xsd:string"
            },
            "pav:derivedFrom": {
              "@type": "@id"
            },
            "pav:createdOn": {
              "@type": "xsd:dateTime"
            },
            "pav:createdBy": {
              "@type": "@id"
            },
            "pav:lastUpdatedOn": {
              "@type": "xsd:dateTime"
            },
            "oslc:modifiedBy": {
              "@type": "@id"
            },
            "skos:notation": {
              "@type": "xsd:string"
            },
            "Input data": "https://schema.metadatacenter.org/properties/d1cfe8ac-fe0e-4679-ac4f-2d7c1ca03c7b",
            "General Model Information": "https://schema.metadatacenter.org/properties/61b8809b-12c4-44e8-8a51-cfcfd13fc87d",
            "Outcome": "https://schema.metadatacenter.org/properties/e45d35b4-90a1-4da5-96d8-d11c947a88a4",
            "Applicability criteria": "https://schema.metadatacenter.org/properties/80f4b20e-9da8-493c-a380-10859551014f",
            "Foundational model or algorithm used": "https://schema.metadatacenter.org/properties/f9370862-b55f-4e91-b4c5-4d216ee38d6b",
            "Primary intended use(s)": "https://schema.metadatacenter.org/properties/15e1ff32-aa1d-4318-a818-f7de336469d6",
            "Primary intended users": "https://schema.metadatacenter.org/properties/60e22218-fc13-4d44-b115-77310f8e7501",
            "Out-of-scope use cases": "https://schema.metadatacenter.org/properties/3ce0ea76-1209-4f15-bdf3-d728b38b55ef",
            "Data": "https://schema.metadatacenter.org/properties/eb491acc-f4a5-4e9e-8c3d-62a7e30099af",
            "Human life": "https://schema.metadatacenter.org/properties/d0ffc5a7-bd62-4187-b2f0-07a66f897837",
            "Mitigations": "https://schema.metadatacenter.org/properties/a00fcb24-bbc0-4689-aabd-c54792413a16",
            "Risks and harms": "https://schema.metadatacenter.org/properties/4feab147-4d43-434e-a340-7970e695eb2c",
            "Use cases": "https://schema.metadatacenter.org/properties/a6e4f2d0-50bc-403a-a969-568b11a5c509",
            "Additional concerns": "https://schema.metadatacenter.org/properties/4ee36e81-2af5-4128-9ec0-048349f3665c",
            "Evaluation results": "https://schema.metadatacenter.org/properties/2e3610dd-3659-4df5-bab0-7342f8418d03",
            "Previous model tests": "https://schema.metadatacenter.org/properties/03c75354-79b0-4abb-81b9-81fede03c4e9"
          },
          "Input data": [
            {
              "@context": {
                "Description": "https://schema.metadatacenter.org/properties/4c6f052b-1e7d-4565-88a9-494d8aafcb31",
                "Type of input": "https://schema.metadatacenter.org/properties/95342d30-2c39-4919-876d-ae3e95961b20",
                "Minimum - for numerical": "https://schema.metadatacenter.org/properties/9e41f733-5b8b-499a-9248-29dbccd2e270",
                "Maximum - for numerical": "https://schema.metadatacenter.org/properties/c471135e-7017-4ae6-a78b-669adf181265",
                "Categories": "https://schema.metadatacenter.org/properties/2c885c9f-73b1-4442-aa25-f7800a8f9911",
                "Input feature": "https://schema.metadatacenter.org/properties/f6df0f4f-af95-4d52-b003-f9d6d1b474db"
              },
              "Description": {
                "@value": "tLength"
              },
              "Type of input": {
                "@value": "numerical"
              },
              "Minimum - for numerical": {
                "@value": "0",
                "@type": "xsd:decimal"
              },
              "Maximum - for numerical": {
                "@value": "1000",
                "@type": "xsd:decimal"
              },
              "Categories": [
                {}
              ],
              "Input feature": {
                "@id": "http://www.cancerdata.org/roo/C100074",
                "rdfs:label": "Tumor Length"
              }
            },
            {
              "@context": {
                "Description": "https://schema.metadatacenter.org/properties/4c6f052b-1e7d-4565-88a9-494d8aafcb31",
                "Type of input": "https://schema.metadatacenter.org/properties/95342d30-2c39-4919-876d-ae3e95961b20",
                "Minimum - for numerical": "https://schema.metadatacenter.org/properties/9e41f733-5b8b-499a-9248-29dbccd2e270",
                "Maximum - for numerical": "https://schema.metadatacenter.org/properties/c471135e-7017-4ae6-a78b-669adf181265",
                "Categories": "https://schema.metadatacenter.org/properties/2c885c9f-73b1-4442-aa25-f7800a8f9911",
                "Input feature": "https://schema.metadatacenter.org/properties/f6df0f4f-af95-4d52-b003-f9d6d1b474db"
              },
              "Description": {
                "@value": "cT"
              },
              "Type of input": {
                "@value": "numerical"
              },
              "Minimum - for numerical": {
                "@value": "0",
                "@type": "xsd:decimal"
              },
              "Maximum - for numerical": {
                "@value": "4",
                "@type": "xsd:decimal"
              },
              "Categories": [
                {}
              ],
              "Input feature": {
                "@id": "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48885",
                "rdfs:label": "Generic Primary Tumor TNM Finding"
              }
            },
            {
              "@context": {
                "Description": "https://schema.metadatacenter.org/properties/4c6f052b-1e7d-4565-88a9-494d8aafcb31",
                "Type of input": "https://schema.metadatacenter.org/properties/95342d30-2c39-4919-876d-ae3e95961b20",
                "Minimum - for numerical": "https://schema.metadatacenter.org/properties/9e41f733-5b8b-499a-9248-29dbccd2e270",
                "Maximum - for numerical": "https://schema.metadatacenter.org/properties/c471135e-7017-4ae6-a78b-669adf181265",
                "Categories": "https://schema.metadatacenter.org/properties/2c885c9f-73b1-4442-aa25-f7800a8f9911",
                "Input feature": "https://schema.metadatacenter.org/properties/f6df0f4f-af95-4d52-b003-f9d6d1b474db"
              },
              "Description": {
                "@value": "cN"
              },
              "Type of input": {
                "@value": "numerical"
              },
              "Minimum - for numerical": {
                "@value": "0",
                "@type": "xsd:decimal"
              },
              "Maximum - for numerical": {
                "@value": "3",
                "@type": "xsd:decimal"
              },
              "Categories": [
                {}
              ],
              "Input feature": {
                "@id": "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48884",
                "rdfs:label": "Generic Regional Lymph Nodes TNM Finding"
              }
            }
          ],
          "General Model Information": {
            "@context": {
              "Title": "https://schema.metadatacenter.org/properties/6610ba27-0f64-4b57-913d-8f673b5eceb1",
              "Editor Note": "https://schema.metadatacenter.org/properties/d1ceb33d-b777-491a-a584-3810a1d9ab3f",
              "Created by": "https://schema.metadatacenter.org/properties/91aa48d3-cd57-4b56-81b7-a75ff33c57f0",
              "References to papers": "https://schema.metadatacenter.org/properties/5a0fa59e-d3e3-41d1-a734-57fb950ae7b8",
              "Contact email": "https://schema.metadatacenter.org/properties/045be6c2-a26f-4ca8-b0c2-e872036def9a",
              "Creation date": "https://schema.metadatacenter.org/properties/87af2a5e-2f89-4448-9143-369705fef329",
              "References to code": "https://schema.metadatacenter.org/properties/1eb8a612-f726-4b6a-aa8f-1ebff646774d",
              "Software License": "https://schema.metadatacenter.org/properties/cfc3fb18-f0a9-45f3-a9ac-bebae520ac12",
              "FAIRmodels image name": "https://schema.metadatacenter.org/properties/f75e9b9c-0891-4040-849d-fdc034a35e7d"
            },
            "Title": {
              "@value": "Prediction of pathologic complete response for rectal cancer patients"
            },
            "Editor Note": {
              "@value": "Based on Stiphout et al. (2011) - clinical variables only"
            },
            "Created by": {
              "@value": "Johan van Soest"
            },
            "References to papers": [
              {
                "@value": "https://doi.org/10.1016/j.radonc.2010.12.002"
              }
            ],
            "Contact email": {
              "@value": "j.vansoest@maastrichtuniversity.nl"
            },
            "Creation date": {
              "@value": "2024-08-24",
              "@type": "xsd:date"
            },
            "References to code": [
              {
                "@value": ""
              }
            ],
            "Software License": {
              "@id": "http://www.ebi.ac.uk/swo/license/SWO_1000001",
              "rdfs:label": "Creative Commons"
            },
            "FAIRmodels image name": {
              "@value": "jvsoest/stiphout_pcr_clinical:latest"
            }
          },
          "Outcome": {
            "@id": "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C123603",
            "rdfs:label": "Pathologic Complete Response"
          },
          "Applicability criteria": [
            {
              "@value": "Patients diagnosed with rectal cancer"
            },
            {
              "@value": "Patients (to be) treated with radiation therapy"
            }
          ],
          "Foundational model or algorithm used": {
            "@id": "https://w3id.org/aio/LogisticRegression",
            "rdfs:label": "Logistic Regression"
          },
          "Primary intended use(s)": [
            {
              "@value": "Prediction of pathologic complete response before start of treatment"
            }
          ],
          "Primary intended users": [
            {
              "@value": "Clinicians"
            }
          ],
          "Out-of-scope use cases": [
            {
              "@value": "Automated clinical decision making"
            }
          ],
          "Data": [
            {
              "@value": "Data used during model development was derived from clinical practice (patients). All data was retrospectively analyzed, and approved by the clinical trial for which patients did provide informed consent."
            }
          ],
          "Human life": [
            {
              "@value": "Model is intended to support clinicians in deciding whether adjuvant treatment (after radiation therapy) is necessary. If a pathologic complete response is predicted, clinicians might opt for watch-and-wait (more follow-up CT/MRI scans) instead of surgery. In case of pCR it could save the patient from potential unnecessary surgery"
            }
          ],
          "Mitigations": [
            {
              "@value": "If pCR is predicted, the risk of false-positives (prediction of pCR, but tumor tissue still available) can be mitigated by watch-and-wait approach with more frequent medical imaging."
            }
          ],
          "Risks and harms": [
            {
              "@value": "Risk could be the low prediction rate, which would indicate a low number of pCR cases (10-15-20% in usual cases), which would mean patients would be referred for surgery instead of more frequent follow-up."
            }
          ],
          "Use cases": {
            "@value": ""
          },
          "Additional concerns": {
            "@value": ""
          },
          "Evaluation results": [
            {
              "@context": {
                "Performance metric": "https://schema.metadatacenter.org/properties/e3fc4bb2-5c13-4ac5-9dbd-1ec0b6b1c406",
                "sha256 of docker image": "https://schema.metadatacenter.org/properties/bf01e549-190e-4429-8a97-4880e2144968",
                "user/hospital": "https://schema.metadatacenter.org/properties/1f03bb07-331b-4675-bbe6-6605eeaceabd",
                "User Note": "https://schema.metadatacenter.org/properties/59fd38ff-bb00-4ef7-9283-3f7dfbd296c6",
                "Dataset characteristics": "https://schema.metadatacenter.org/properties/dc4a07fc-2645-4c70-9330-3e249031f046"
              },
              "Performance metric": [
                {
                  "@context": {
                    "Metric Label": "https://schema.metadatacenter.org/properties/f247f273-151f-4c1d-bd69-8ed6a728ed33",
                    "Measured metric (mean value)": "https://schema.metadatacenter.org/properties/e6a0cec6-e37d-431c-8480-e3eba2f0d071",
                    "Measured metric (low 95% confidence interval)": "https://schema.metadatacenter.org/properties/d93c93b8-f1ce-4d2b-b78d-1b0f2d0c1ef2",
                    "Measured metric (up 95% confidence interval)": "https://schema.metadatacenter.org/properties/5aa71926-22e4-4628-b563-5c116fd8f67e",
                    "Acceptance level": "https://schema.metadatacenter.org/properties/668d9bbc-ae81-4af0-a11e-7a401c409f98",
                    "Additional information (if needed)": "https://schema.metadatacenter.org/properties/cbcf4b91-fcb5-4462-8eae-77cba960c0e0"
                  },
                  "Metric Label": {},
                  "Measured metric (mean value)": {
                    "@value": "",
                    "@type": "xsd:decimal"
                  },
                  "Measured metric (low 95% confidence interval)": {
                    "@value": "",
                    "@type": "xsd:decimal"
                  },
                  "Measured metric (up 95% confidence interval)": {
                    "@value": ""
                  },
                  "Acceptance level": {
                    "@value": "",
                    "@type": "xsd:decimal"
                  },
                  "Additional information (if needed)": {
                    "@value": ""
                  }
                }
              ],
              "sha256 of docker image": {
                "@value": "vghvg"
              },
              "user/hospital": {
                "@value": ""
              },
              "User Note": {
                "@value": ""
              },
              "Dataset characteristics": [
                {
                  "@context": {
                    "Input feature": "https://schema.metadatacenter.org/properties/f53523d7-aa41-46c2-b8ab-39772bce6037",
                    "Volume": "https://schema.metadatacenter.org/properties/3aaeb721-75db-4257-915f-3b49a2f9b5b9",
                    "The characteristics of dataset": "https://schema.metadatacenter.org/properties/105fdab7-6612-4209-bc80-3def1013a9a1",
                    "Number of missing values": "https://schema.metadatacenter.org/properties/e3a95563-8ac6-4cd3-b9df-23412bc0283c",
                    "Categories distribution": "https://schema.metadatacenter.org/properties/19741c6b-354a-47bd-a934-9b96a747b5b6",
                    "The number of subject for evaluation": "https://schema.metadatacenter.org/properties/ce4ca156-2fc2-4072-8c8e-9b8c29016c18",
                    "The mean value - for numerical feature": "https://schema.metadatacenter.org/properties/0b258d16-e907-4f5b-9ff3-1d77d20fdabb",
                    "The low 95% confidence interval - for numerical feature": "https://schema.metadatacenter.org/properties/0f2bc2b0-37e6-4392-aec1-41c86a61d009",
                    "The high 95% confidence interval - for numerical feature": "https://schema.metadatacenter.org/properties/ea2134fc-80dc-44df-9120-1b8be9f5c089"
                  },
                  "Input feature": {
                    "@id": "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C200479",
                    "rdfs:label": "Tumor Length"
                  },
                  "Volume": {
                    "@value": "400"
                  },
                  "The characteristics of dataset": {
                    "@value": ""
                  },
                  "Number of missing values": {
                    "@value": "",
                    "@type": "xsd:decimal"
                  },
                  "Categories distribution": [
                    {
                      "@context": {
                        "Category Label": "https://schema.metadatacenter.org/properties/e6cf70d0-7e05-4122-b318-b4d93ee63c86",
                        "Distribution for category": "https://schema.metadatacenter.org/properties/0ff233ed-291e-40d4-96b1-ef71bd5d5871"
                      },
                      "Category Label": {},
                      "Distribution for category": {
                        "@value": ""
                      }
                    }
                  ],
                  "The number of subject for evaluation": {
                    "@value": "",
                    "@type": "xsd:decimal"
                  },
                  "The mean value - for numerical feature": {
                    "@value": "",
                    "@type": "xsd:decimal"
                  },
                  "The low 95% confidence interval - for numerical feature": {
                    "@value": "",
                    "@type": "xsd:decimal"
                  },
                  "The high 95% confidence interval - for numerical feature": {
                    "@value": "",
                    "@type": "xsd:decimal"
                  }
                }
              ]
            }
          ],
          "Previous model tests": [
            {
              "@context": {
                "Performance metric": "https://schema.metadatacenter.org/properties/f95c7b9f-a5bc-410b-ad38-d719a3044fe4",
                "Link to dataset": "https://schema.metadatacenter.org/properties/d750fe0d-6813-4536-a918-bd9218f472bf",
                "Link to reference paper": "https://schema.metadatacenter.org/properties/d59a2224-fbe8-4380-b66b-a2cc05c6b3ae",
                "Notes": "https://schema.metadatacenter.org/properties/c6428800-ffc3-41b6-986a-3a95ae936c6e"
              },
              "Performance metric": [
                {
                  "@context": {
                    "Metric Label": "https://schema.metadatacenter.org/properties/f247f273-151f-4c1d-bd69-8ed6a728ed33",
                    "Measured metric (mean value)": "https://schema.metadatacenter.org/properties/e6a0cec6-e37d-431c-8480-e3eba2f0d071",
                    "Measured metric (low 95% confidence interval)": "https://schema.metadatacenter.org/properties/d93c93b8-f1ce-4d2b-b78d-1b0f2d0c1ef2",
                    "Measured metric (up 95% confidence interval)": "https://schema.metadatacenter.org/properties/5aa71926-22e4-4628-b563-5c116fd8f67e",
                    "Acceptance level": "https://schema.metadatacenter.org/properties/668d9bbc-ae81-4af0-a11e-7a401c409f98",
                    "Additional information (if needed)": "https://schema.metadatacenter.org/properties/cbcf4b91-fcb5-4462-8eae-77cba960c0e0"
                  },
                  "Metric Label": {
                    "@id": "http://purl.obolibrary.org/obo/STATO_0000608",
                    "rdfs:label": "area under the receiver operator characteristic curve"
                  },
                  "Measured metric (mean value)": {
                    "@value": "0.61",
                    "@type": "xsd:decimal"
                  },
                  "Measured metric (low 95% confidence interval)": {
                    "@value": "0.607",
                    "@type": "xsd:decimal"
                  },
                  "Measured metric (up 95% confidence interval)": {
                    "@value": "0.612"
                  },
                  "Acceptance level": {
                    "@value": "",
                    "@type": "xsd:decimal"
                  },
                  "Additional information (if needed)": {
                    "@value": ""
                  }
                }
              ],
              "Link to dataset": {
                "@id": "https://cancerdata.org/id/10.5072/candat.2015.02"
              },
              "Link to reference paper": {
                "@id": "https://doi.org/10.1016/j.radonc.2010.12.002"
              },
              "Notes": {
                "@value": ""
              }
            }
          ],
          "21c6f103-2897-46e4-9800-d6756aec8fea": {},
          "eb387f72-1d95-42fd-a368-dc157fd6d8bc": {},
          "8a5c5fda-0607-4cbf-8941-a8be9d05a83b": {},
          "202471c3-13e6-42af-a104-3df02b9e0507": {},
          "9128f569-aca5-487d-93d6-3a99ad2073a1": {},
          "1bb43806-9e82-4e65-86f9-cebb4e7f3b38": {},
          "b4db2a96-c081-453b-b09c-caf7516d147d": {},
          "schema:isBasedOn": "https://repo.metadatacenter.org/templates/b73f7c2c-b8fa-4b8c-b9e8-258a24bb1df7",
          "schema:name": "Model Card metadata",
          "schema:description": "The update template for FAIVOR project"
        }
        # check numerical data
        # Tumor length
        feature = "tLength"
        min_f, max_f = get_range(f, feature)
        validate_numerical_feature(data, feature, min_f, max_f)

        # Generic Primary Tumor TNM Finding (T stage)
        feature = "cT"
        min_f, max_f = get_range(f, feature)
        validate_numerical_feature(data, feature, min_f, max_f)

        # Generic Regional Lymph Nodes TNM Finding ( N stage)
        feature = "cN"
        min_f, max_f = get_range(f, feature)
        validate_numerical_feature(data, feature, min_f, max_f)

        return data

if __name__ == "__main__":
    model_obj = stiphout_pCR_Clinical()
    model_obj.get_input_parameters()
    print(model_obj.predict(
        {
            "cT": 4,
            "cN": 1,
            "tLength": 15
        }
    ))

