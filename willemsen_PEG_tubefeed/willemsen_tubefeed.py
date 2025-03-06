from math import log, exp
from model_execution import logistic_regression
import json
from fastapi import HTTPException


def get_range(metadata, feature):
    # Find the entry where "Description" has value "BMI"
    entry = next((entry for entry in metadata["Input data"] if entry.get("Description", {}).get("@value") == feature), None)

    if not entry:
        return None, None  # Return None if BMI entry is not found

    # Extract min and max values
    min_value = float(entry.get("Minimum - for numerical", {}).get("@value", "0"))
    max_value = float(entry.get("Maximum - for numerical", {}).get("@value", "0"))

    return min_value, max_value
def get_categorical_values(metadata, description_value):
    # Find the entry where "Description" matches the given value
    category_entry = next((entry for entry in metadata.get("Input data", [])
                           if isinstance(entry, dict) and entry.get("Description", {}).get("@value") == description_value), None)
    categories = category_entry.get("Categories", [])  # Get the Categories list
    categories_id= [category.get("Identification for category used in model", {}).get("@value")
        for category in categories
        if isinstance(category, dict)]

    if not categories_id:
        return None  # Return None if no matching entry is found

    return categories_id

class willemsen_tubefeed(logistic_regression):
    def __init__(self):
        #with open('willemsen_tubefeed.json') as f:
        self._model_parameters = {
    #Part of metadata that is only inside the docker container
    "intercept": -0.506,
    "covariate_weights": {
        "BMI": -0.042,
        "WeightLoss": -0.03,
        "TF": 0.452,
        "PS": 0.608,
        "Tumorlocation": -0.51,
        "Tclassification": 0.311,
        "Nclassification": 0.561,
        "Systherapy": -0.655,
        "RTdose_subman": 0.015,
        "RTdosesalivary": 0.042
    }}

    #Part of metadata that could be fetched from faifmodels.org -> f
#     "model_name": "PEG tube feeding",
#     "model_uri": "https://www.predictcancer.ai/Main.php?page=TFDInfo",
#     "full_model_name": "Prediction model for tube feeding dependency during chemoradiotherapy for at least four weeks in head and neck cancer patients",
#     "List of essential  metrics": [
#         "PPV",
#         "NPV",
#         "Calibration"
#     ],
#     "Performance metrics": {
#         "AUC": "74.8% (95% CI  70.1-79.6%)",
#         "PPV": "90%",
#         "NPV": "64%"
#     },
#     "outcome": {
#         "description": "Prediction to identify patients who need prophylactic gastrostomy insertion for at least four weeks",
#         "type": "numerical",
#         "code": "61420007",
#         "terminology": "SNOMEDCT"
#     },
#     "features": {
#         "BMI": {
#             "description": "Body mass index",
#             "type": "numerical",
#             "code": "Z68",
#             "terminology": "ICD10CM",
#             "min": 5,
#             "max": 40,
#             "unit": "kg/m2"
#
#         },
#         "WeightLoss": {
#             "description": "Weight change finding",
#             "type": "numerical",
#             "code": "365921005",
#             "terminology": "SNOMEDCT",
#             "min": -30,
#             "max": 30,
#             "unit": "%"
#         },
#         "TF": {
#             "description": "Texture modified diet or tube feeding",
#             "type": "categorical",
#             "code": "435801000124108",
#             "terminology": "SNOMEDCT",
#             "values": {
#                 "yes": 1,
#                 "no": 0
#             }
#         },
#         "PS": {
#             "description": "WHO performance status finding",
#             "type": "categorical",
#             "code": "373802001",
#             "terminology": "SNOMEDCT",
#             "values": {
#                 "373803006": 0,
#                 "373804000": 1,
#                 "373805004": 2,
#                 "373806003": 3,
#                 "373807007": 4
#             }
#         },
#         "Tumorlocation": {
#             "description": "Tumor location (Head finding)",
#             "type": "categorical",
#             "code": "406122000",
#             "terminology": "SNOMEDCT",
#             "values": {
#                 "235075007": 1,
#                 "232388008": 2,
#                 "74409009": 3,
#                 "126809003": 4,
#                 "126686005": 5,
#                 "126692004": 6,
#                 "406122000": 7,
#                 "Other": 8
#             }
#         },
#         "Tclassification": {
#             "description": "Generic Primary Tumor TNM Finding",
#             "type": "categorical",
#             "code": "C48885",
#             "terminology": "ROO",
#             "values": {
#                 "C48719": 0,
#                 "C48720": 1,
#                 "C48724": 2,
#                 "C48728": 3,
#                 "C48732": 4,
#                 "C48737": "x"
#             }
#         },
#         "Nclassification": {
#             "description": "Generic Regional Lymph Nodes TNM Finding",
#             "type": "categorical",
#             "code": "C48884",
#             "terminology": "ROO",
#             "values": {
#                 "C48705": 0,
#                 "C48706": 1,
#                 "C48786": 2,
#                 "C48714": 3,
#                 "C48718": "x"
#             }
#         },
#         "Systherapy": {
#             "description": "Systemic Therapy",
#             "type": "categorical",
#             "code": "C15698",
#             "terminology": "NCI_Thesaurus",
#             "values": {
#                 "C173291": 1,
#                 "C92991": 0
#             }
#         },
#         "RTdose_subman": {
#             "description": "Mean Radiation Dose on contralateral submandibular gland [Gy]",
#             "type": "numerical",
#             "code": "C190594",
#             "terminology": "NCI_Thesaurus",
#             "min": 0,
#             "max": 60,
#             "unit": "Gy"
#         },
#         "RTdosesalivary": {
#             "description": "Mean Radiation Dose contralateral parotid salivary gland [Gy]",
#             "type": "numerical",
#             "code": "C190594",
#             "terminology": "NCI_Thesaurus",
#             "min": 0,
#             "max": 60,
#             "unit": "Gy"
#         }
#     },
#     "Description": "Model for predicting the prophylactic gastronomy insertion (tube feeding) for at least four weeks in head and neck cancer patients undergoing chemoradiotherapy",
#     "Applicability criteria": "Patients diagnosed with locally advanced head and neck squamous cell carcinoma (LAHNSCC). Patients undergoing chemoradiotherapy (CRT) or bioradiotherapy (BRT) with curative intent. Patients who are starting primary or adjuvant radiotherapy combined with either cisplatin, carboplatin, or cetuximab.",
#     "Primary intended use": "To support personalized decision-making on prophylactic gastrostomy insertion in head and neck cancer patients undergoing chemoradiotherapy.",
#     "Primary intended users": "Clinicians and healthcare providers treating head and neck cancer patients."
# }

    def _preprocess(self, data):
        """
        This function is used to convert the input data into the correct format for the model.

        Parameters:
        - input_object: a dictionary, or list with multiple dictionaries, containing the input data

        Returns:
        - preprocessed_data: a dictionary, or list with multiple dictionaries, containing the preprocessed data
        """
        # f = {
        #     "model_name": "PEG tube feeding",
        #     "model_uri": "https://www.predictcancer.ai/Main.php?page=TFDInfo",
        #     "full_model_name": "Prediction model for tube feeding dependency during chemoradiotherapy for at least four weeks in head and neck cancer patients",
        #     "intercept": -0.506,
        #     "covariate_weights": {
        #         "BMI": -0.042,
        #         "WeightLoss": -0.03,
        #         "TF": 0.452,
        #         "PS": 0.608,
        #         "Tumorlocation": -0.51,
        #         "Tclassification": 0.311,
        #         "Nclassification": 0.561,
        #         "Systherapy": -0.655,
        #         "RTdose_subman": 0.015,
        #         "RTdosesalivary": 0.042
        #     },
        #     "List of essential  metrics": [
        #         "PPV",
        #         "NPV",
        #         "Calibration"
        #     ],
        #     "Performance metrics": {
        #         "AUC": "74.8% (95% CI  70.1-79.6%)",
        #         "PPV": "90%",
        #         "NPV": "64%"
        #     },
        #     "outcome": {
        #         "description": "Prediction to identify patients who need prophylactic gastrostomy insertion for at least four weeks",
        #         "type": "numerical",
        #         "code": "61420007",
        #         "terminology": "SNOMEDCT"
        #     },
        #     "features": {
        #         "BMI": {
        #             "description": "Body mass index",
        #             "type": "numerical",
        #             "code": "Z68",
        #             "terminology": "ICD10CM",
        #             "min": 5,
        #             "max": 40,
        #             "unit": "kg/m2"
        # 
        #         },
        #         "WeightLoss": {
        #             "description": "Weight change finding",
        #             "type": "numerical",
        #             "code": "365921005",
        #             "terminology": "SNOMEDCT",
        #             "min": -30,
        #             "max": 30,
        #             "unit": "%"
        #         },
        #         "TF": {
        #             "description": "Texture modified diet or tube feeding",
        #             "type": "categorical",
        #             "code": "435801000124108",
        #             "terminology": "SNOMEDCT",
        #             "values": {
        #                 "yes": 1,
        #                 "no": 0
        #             }
        #         },
        #         "PS": {
        #             "description": "WHO performance status finding",
        #             "type": "categorical",
        #             "code": "373802001",
        #             "terminology": "SNOMEDCT",
        #             "values": {
        #                 "373803006": 0,
        #                 "373804000": 1,
        #                 "373805004": 2,
        #                 "373806003": 3,
        #                 "373807007": 4
        #             }
        #         },
        #         "Tumorlocation": {
        #             "description": "Tumor location (Head finding)",
        #             "type": "categorical",
        #             "code": "406122000",
        #             "terminology": "SNOMEDCT",
        #             "values": {
        #                 "235075007": 1,
        #                 "232388008": 2,
        #                 "74409009": 3,
        #                 "126809003": 4,
        #                 "126686005": 5,
        #                 "126692004": 6,
        #                 "406122000": 7
        #             }
        #         },
        #         "Tclassification": {
        #             "description": "Generic Primary Tumor TNM Finding",
        #             "type": "categorical",
        #             "code": "C48885",
        #             "terminology": "ROO",
        #             "values": {
        #                 "C48719": 0,
        #                 "C48720": 1,
        #                 "C48724": 2,
        #                 "C48728": 3,
        #                 "C48732": 4,
        #                 "C48737": "x"
        #             }
        #         },
        #         "Nclassification": {
        #             "description": "Generic Regional Lymph Nodes TNM Finding",
        #             "type": "categorical",
        #             "code": "C48884",
        #             "terminology": "ROO",
        #             "values": {
        #                 "C48705": 0,
        #                 "C48706": 1,
        #                 "C48786": 2,
        #                 "C48714": 3,
        #                 "C48718": "x"
        #             }
        #         },
        #         "Systherapy": {
        #             "description": "Systemic Therapy",
        #             "type": "categorical",
        #             "code": "C15698",
        #             "terminology": "NCI_Thesaurus",
        #             "values": {
        #                 "C173291": 1,
        #                 "C92991": 0
        #             }
        #         },
        #         "RTdose_subman": {
        #             "description": "Mean Radiation Dose on contralateral submandibular gland [Gy]",
        #             "type": "numerical",
        #             "code": "C190594",
        #             "terminology": "NCI_Thesaurus",
        #             "min": 0,
        #             "max": 60,
        #             "unit": "Gy"
        #         },
        #         "RTdosesalivary": {
        #             "description": "Mean Radiation Dose contralateral parotid salivary gland [Gy]",
        #             "type": "numerical",
        #             "code": "C190594",
        #             "terminology": "NCI_Thesaurus",
        #             "min": 0,
        #             "max": 60,
        #             "unit": "Gy"
        #         }
        #     },
        #     "Description": "Model for predicting the prophylactic gastronomy insertion (tube feeding) for at least four weeks in head and neck cancer patients undergoing chemoradiotherapy",
        #     "Applicability criteria": "Patients diagnosed with locally advanced head and neck squamous cell carcinoma (LAHNSCC). Patients undergoing chemoradiotherapy (CRT) or bioradiotherapy (BRT) with curative intent. Patients who are starting primary or adjuvant radiotherapy combined with either cisplatin, carboplatin, or cetuximab.",
        #     "Primary intended use": "To support personalized decision-making on prophylactic gastrostomy insertion in head and neck cancer patients undergoing chemoradiotherapy.",
        #     "Primary intended users": "Clinicians and healthcare providers treating head and neck cancer patients."
        # 
        # }

        #try:
            # return data
        #     ## check the values and perform the conversion of categorical variables
        #
            # Tumor TNM finding validation
            # Extract allowed values from JSON
        f={
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
                            "@value": "BMI"
                        },
                        "Type of input": {
                            "@value": "numerical"
                        },
                        "Minimum - for numerical": {
                            "@value": "5",
                            "@type": "xsd:decimal"
                        },
                        "Maximum - for numerical": {
                            "@value": "40",
                            "@type": "xsd:decimal"
                        },
                        "Categories": [
                            {}
                        ],
                        "Input feature": {
                            "@id": "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C16358",
                            "rdfs:label": "Body Mass Index"
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
                            "@value": "WeightLoss"
                        },
                        "Type of input": {
                            "@value": "numerical"
                        },
                        "Minimum - for numerical": {
                            "@value": "-30",
                            "@type": "xsd:decimal"
                        },
                        "Maximum - for numerical": {
                            "@value": "30",
                            "@type": "xsd:decimal"
                        },
                        "Categories": [
                            {}
                        ],
                        "Input feature": {
                            "@id": "http://purl.bioontology.org/ontology/SNOMEDCT/365921005",
                            "rdfs:label": "Weight change finding"
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
                            "@value": "TF"
                        },
                        "Type of input": {
                            "@value": "categorical"
                        },
                        "Minimum - for numerical": {
                            "@value": "",
                            "@type": "xsd:decimal"
                        },
                        "Maximum - for numerical": {
                            "@value": "",
                            "@type": "xsd:decimal"
                        },
                        "Categories": [
                            {
                                "@context": {
                                    "Category Label": "https://schema.metadatacenter.org/properties/e6cf70d0-7e05-4122-b318-b4d93ee63c86",
                                    "Identification for category used in model": "https://schema.metadatacenter.org/properties/0ff233ed-291e-40d4-96b1-ef71bd5d5871"
                                },
                                "Category Label": {
                                    "@id": "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C49488",
                                    "rdfs:label": "Yes"
                                },
                                "Identification for category used in model": {
                                    "@value": "1"
                                }
                            },
                            {
                                "@context": {
                                    "Category Label": "https://schema.metadatacenter.org/properties/e6cf70d0-7e05-4122-b318-b4d93ee63c86",
                                    "Identification for category used in model": "https://schema.metadatacenter.org/properties/0ff233ed-291e-40d4-96b1-ef71bd5d5871"
                                },
                                "Category Label": {
                                    "@id": "http://purl.bioontology.org/ontology/SNOMEDCT/373067005",
                                    "rdfs:label": "No"
                                },
                                "Identification for category used in model": {
                                    "@value": "0"
                                }
                            }
                        ],
                        "Input feature": {
                            "@id": "http://purl.bioontology.org/ontology/SNOMEDCT/435801000124108",
                            "rdfs:label": "Texture modified diet"
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
                            "@value": "PS"
                        },
                        "Type of input": {
                            "@value": "categorical"
                        },
                        "Minimum - for numerical": {
                            "@value": "",
                            "@type": "xsd:decimal"
                        },
                        "Maximum - for numerical": {
                            "@value": "",
                            "@type": "xsd:decimal"
                        },
                        "Categories": [
                            {
                                "@context": {
                                    "Category Label": "https://schema.metadatacenter.org/properties/e6cf70d0-7e05-4122-b318-b4d93ee63c86",
                                    "Identification for category used in model": "https://schema.metadatacenter.org/properties/0ff233ed-291e-40d4-96b1-ef71bd5d5871"
                                },
                                "Category Label": {
                                    "@id": "http://purl.bioontology.org/ontology/SNOMEDCT/373803006",
                                    "rdfs:label": "WHO performance status grade 0"
                                },
                                "Identification for category used in model": {
                                    "@value": "0"
                                }
                            },
                            {
                                "@context": {
                                    "Category Label": "https://schema.metadatacenter.org/properties/e6cf70d0-7e05-4122-b318-b4d93ee63c86",
                                    "Identification for category used in model": "https://schema.metadatacenter.org/properties/0ff233ed-291e-40d4-96b1-ef71bd5d5871"
                                },
                                "Category Label": {
                                    "@id": "http://purl.bioontology.org/ontology/SNOMEDCT/373804000",
                                    "rdfs:label": "WHO performance status grade 1"
                                },
                                "Identification for category used in model": {
                                    "@value": "1"
                                }
                            },
                            {
                                "@context": {
                                    "Category Label": "https://schema.metadatacenter.org/properties/e6cf70d0-7e05-4122-b318-b4d93ee63c86",
                                    "Identification for category used in model": "https://schema.metadatacenter.org/properties/0ff233ed-291e-40d4-96b1-ef71bd5d5871"
                                },
                                "Category Label": {},
                                "Identification for category used in model": {
                                    "@value": "2"
                                }
                            },
                            {
                                "@context": {
                                    "Category Label": "https://schema.metadatacenter.org/properties/e6cf70d0-7e05-4122-b318-b4d93ee63c86",
                                    "Identification for category used in model": "https://schema.metadatacenter.org/properties/0ff233ed-291e-40d4-96b1-ef71bd5d5871"
                                },
                                "Category Label": {},
                                "Identification for category used in model": {
                                    "@value": "3"
                                }
                            },
                            {
                                "@context": {
                                    "Category Label": "https://schema.metadatacenter.org/properties/e6cf70d0-7e05-4122-b318-b4d93ee63c86",
                                    "Identification for category used in model": "https://schema.metadatacenter.org/properties/0ff233ed-291e-40d4-96b1-ef71bd5d5871"
                                },
                                "Category Label": {},
                                "Identification for category used in model": {
                                    "@value": "4"
                                }
                            }
                        ],
                        "Input feature": {
                            "@id": "http://purl.bioontology.org/ontology/SNOMEDCT/373802001",
                            "rdfs:label": "WHO performance status finding"
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
                            "@value": "Tumorlocation"
                        },
                        "Type of input": {
                            "@value": "categorical"
                        },
                        "Minimum - for numerical": {
                            "@value": "",
                            "@type": "xsd:decimal"
                        },
                        "Maximum - for numerical": {
                            "@value": "",
                            "@type": "xsd:decimal"
                        },
                        "Categories": [
                            {
                                "@context": {
                                    "Category Label": "https://schema.metadatacenter.org/properties/e6cf70d0-7e05-4122-b318-b4d93ee63c86",
                                    "Identification for category used in model": "https://schema.metadatacenter.org/properties/0ff233ed-291e-40d4-96b1-ef71bd5d5871"
                                },
                                "Category Label": {
                                    "@id": "http://purl.bioontology.org/ontology/SNOMEDCT/235075007",
                                    "rdfs:label": "Tumor of oral cavity"
                                },
                                "Identification for category used in model": {
                                    "@value": "1"
                                }
                            },
                            {
                                "@context": {
                                    "Category Label": "https://schema.metadatacenter.org/properties/e6cf70d0-7e05-4122-b318-b4d93ee63c86",
                                    "Identification for category used in model": "https://schema.metadatacenter.org/properties/0ff233ed-291e-40d4-96b1-ef71bd5d5871"
                                },
                                "Category Label": {
                                    "@id": "http://purl.bioontology.org/ontology/SNOMEDCT/232388008",
                                    "rdfs:label": "Tumor of nasal cavity and nasopharynx"
                                },
                                "Identification for category used in model": {
                                    "@value": "2"
                                }
                            },
                            {
                                "@context": {
                                    "Category Label": "https://schema.metadatacenter.org/properties/e6cf70d0-7e05-4122-b318-b4d93ee63c86",
                                    "Identification for category used in model": "https://schema.metadatacenter.org/properties/0ff233ed-291e-40d4-96b1-ef71bd5d5871"
                                },
                                "Category Label": {
                                    "@id": "http://purl.bioontology.org/ontology/SNOMEDCT/74409009",
                                    "rdfs:label": "Endodermal sinus tumor"
                                },
                                "Identification for category used in model": {
                                    "@value": "3"
                                }
                            },
                            {
                                "@context": {
                                    "Category Label": "https://schema.metadatacenter.org/properties/e6cf70d0-7e05-4122-b318-b4d93ee63c86",
                                    "Identification for category used in model": "https://schema.metadatacenter.org/properties/0ff233ed-291e-40d4-96b1-ef71bd5d5871"
                                },
                                "Category Label": {
                                    "@id": "http://purl.bioontology.org/ontology/SNOMEDCT/126809003",
                                    "rdfs:label": "Neoplasm of oropharynx"
                                },
                                "Identification for category used in model": {
                                    "@value": "4"
                                }
                            },
                            {
                                "@context": {
                                    "Category Label": "https://schema.metadatacenter.org/properties/e6cf70d0-7e05-4122-b318-b4d93ee63c86",
                                    "Identification for category used in model": "https://schema.metadatacenter.org/properties/0ff233ed-291e-40d4-96b1-ef71bd5d5871"
                                },
                                "Category Label": {
                                    "@id": "http://purl.bioontology.org/ontology/SNOMEDCT/126686005",
                                    "rdfs:label": "Neoplasm of hypopharynx"
                                },
                                "Identification for category used in model": {
                                    "@value": "5"
                                }
                            },
                            {
                                "@context": {
                                    "Category Label": "https://schema.metadatacenter.org/properties/e6cf70d0-7e05-4122-b318-b4d93ee63c86",
                                    "Identification for category used in model": "https://schema.metadatacenter.org/properties/0ff233ed-291e-40d4-96b1-ef71bd5d5871"
                                },
                                "Category Label": {
                                    "@id": "http://purl.bioontology.org/ontology/SNOMEDCT/126692004",
                                    "rdfs:label": "Neoplasm of larynx"
                                },
                                "Identification for category used in model": {
                                    "@value": "6"
                                }
                            },
                            {
                                "@context": {
                                    "Category Label": "https://schema.metadatacenter.org/properties/e6cf70d0-7e05-4122-b318-b4d93ee63c86",
                                    "Identification for category used in model": "https://schema.metadatacenter.org/properties/0ff233ed-291e-40d4-96b1-ef71bd5d5871"
                                },
                                "Category Label": {
                                    "@id": "http://purl.bioontology.org/ontology/SNOMEDCT/406122000",
                                    "rdfs:label": "Head finding"
                                },
                                "Identification for category used in model": {
                                    "@value": "7"
                                }
                            }
                        ],
                        "Input feature": {
                            "@id": "http://purl.bioontology.org/ontology/SNOMEDCT/406122000",
                            "rdfs:label": "Head finding"
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
                            "@value": "Tclassification"
                        },
                        "Type of input": {
                            "@value": "categorical"
                        },
                        "Minimum - for numerical": {
                            "@value": "",
                            "@type": "xsd:decimal"
                        },
                        "Maximum - for numerical": {
                            "@value": "",
                            "@type": "xsd:decimal"
                        },
                        "Categories": [
                            {
                                "@context": {
                                    "Category Label": "https://schema.metadatacenter.org/properties/e6cf70d0-7e05-4122-b318-b4d93ee63c86",
                                    "Identification for category used in model": "https://schema.metadatacenter.org/properties/0ff233ed-291e-40d4-96b1-ef71bd5d5871"
                                },
                                "Category Label": {
                                    "@id": "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48719",
                                    "rdfs:label": "T0 Stage Finding"
                                },
                                "Identification for category used in model": {
                                    "@value": "0"
                                }
                            },
                            {
                                "@context": {
                                    "Category Label": "https://schema.metadatacenter.org/properties/e6cf70d0-7e05-4122-b318-b4d93ee63c86",
                                    "Identification for category used in model": "https://schema.metadatacenter.org/properties/0ff233ed-291e-40d4-96b1-ef71bd5d5871"
                                },
                                "Category Label": {
                                    "@id": "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48720",
                                    "rdfs:label": "T1 Stage Finding"
                                },
                                "Identification for category used in model": {
                                    "@value": "1"
                                }
                            },
                            {
                                "@context": {
                                    "Category Label": "https://schema.metadatacenter.org/properties/e6cf70d0-7e05-4122-b318-b4d93ee63c86",
                                    "Identification for category used in model": "https://schema.metadatacenter.org/properties/0ff233ed-291e-40d4-96b1-ef71bd5d5871"
                                },
                                "Category Label": {
                                    "@id": "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48724",
                                    "rdfs:label": "T2 Stage Finding"
                                },
                                "Identification for category used in model": {
                                    "@value": "2"
                                }
                            },
                            {
                                "@context": {
                                    "Category Label": "https://schema.metadatacenter.org/properties/e6cf70d0-7e05-4122-b318-b4d93ee63c86",
                                    "Identification for category used in model": "https://schema.metadatacenter.org/properties/0ff233ed-291e-40d4-96b1-ef71bd5d5871"
                                },
                                "Category Label": {
                                    "@id": "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48728",
                                    "rdfs:label": "T3 Stage Finding"
                                },
                                "Identification for category used in model": {
                                    "@value": "3"
                                }
                            },
                            {
                                "@context": {
                                    "Category Label": "https://schema.metadatacenter.org/properties/e6cf70d0-7e05-4122-b318-b4d93ee63c86",
                                    "Identification for category used in model": "https://schema.metadatacenter.org/properties/0ff233ed-291e-40d4-96b1-ef71bd5d5871"
                                },
                                "Category Label": {
                                    "@id": "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48732",
                                    "rdfs:label": "T4 Stage Finding"
                                },
                                "Identification for category used in model": {
                                    "@value": "4"
                                }
                            },
                            {
                                "@context": {
                                    "Category Label": "https://schema.metadatacenter.org/properties/e6cf70d0-7e05-4122-b318-b4d93ee63c86",
                                    "Identification for category used in model": "https://schema.metadatacenter.org/properties/0ff233ed-291e-40d4-96b1-ef71bd5d5871"
                                },
                                "Category Label": {
                                    "@id": "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48737",
                                    "rdfs:label": "TX Stage Finding"
                                },
                                "Identification for category used in model": {
                                    "@value": "x"
                                }
                            }
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
                            "@value": "Nclassification"
                        },
                        "Type of input": {
                            "@value": "categorical"
                        },
                        "Minimum - for numerical": {
                            "@value": "",
                            "@type": "xsd:decimal"
                        },
                        "Maximum - for numerical": {
                            "@value": "",
                            "@type": "xsd:decimal"
                        },
                        "Categories": [
                            {
                                "@context": {
                                    "Category Label": "https://schema.metadatacenter.org/properties/e6cf70d0-7e05-4122-b318-b4d93ee63c86",
                                    "Identification for category used in model": "https://schema.metadatacenter.org/properties/0ff233ed-291e-40d4-96b1-ef71bd5d5871"
                                },
                                "Category Label": {
                                    "@id": "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48705",
                                    "rdfs:label": "N0 Stage Finding"
                                },
                                "Identification for category used in model": {
                                    "@value": "0"
                                }
                            },
                            {
                                "@context": {
                                    "Category Label": "https://schema.metadatacenter.org/properties/e6cf70d0-7e05-4122-b318-b4d93ee63c86",
                                    "Identification for category used in model": "https://schema.metadatacenter.org/properties/0ff233ed-291e-40d4-96b1-ef71bd5d5871"
                                },
                                "Category Label": {
                                    "@id": "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48706",
                                    "rdfs:label": "N1 Stage Finding"
                                },
                                "Identification for category used in model": {
                                    "@value": "1"
                                }
                            },
                            {
                                "@context": {
                                    "Category Label": "https://schema.metadatacenter.org/properties/e6cf70d0-7e05-4122-b318-b4d93ee63c86",
                                    "Identification for category used in model": "https://schema.metadatacenter.org/properties/0ff233ed-291e-40d4-96b1-ef71bd5d5871"
                                },
                                "Category Label": {
                                    "@id": "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48786",
                                    "rdfs:label": "N2 Stage Finding"
                                },
                                "Identification for category used in model": {
                                    "@value": "2"
                                }
                            },
                            {
                                "@context": {
                                    "Category Label": "https://schema.metadatacenter.org/properties/e6cf70d0-7e05-4122-b318-b4d93ee63c86",
                                    "Identification for category used in model": "https://schema.metadatacenter.org/properties/0ff233ed-291e-40d4-96b1-ef71bd5d5871"
                                },
                                "Category Label": {
                                    "@id": "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48714",
                                    "rdfs:label": "N3 Stage Finding"
                                },
                                "Identification for category used in model": {
                                    "@value": "3"
                                }
                            },
                            {
                                "@context": {
                                    "Category Label": "https://schema.metadatacenter.org/properties/e6cf70d0-7e05-4122-b318-b4d93ee63c86",
                                    "Identification for category used in model": "https://schema.metadatacenter.org/properties/0ff233ed-291e-40d4-96b1-ef71bd5d5871"
                                },
                                "Category Label": {
                                    "@id": "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48718",
                                    "rdfs:label": "NX Stage Finding"
                                },
                                "Identification for category used in model": {
                                    "@value": "x"
                                }
                            }
                        ],
                        "Input feature": {
                            "@id": "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C48884",
                            "rdfs:label": "Generic Regional Lymph Nodes TNM Finding"
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
                            "@value": "Systherapy"
                        },
                        "Type of input": {
                            "@value": "categorical"
                        },
                        "Minimum - for numerical": {
                            "@value": "",
                            "@type": "xsd:decimal"
                        },
                        "Maximum - for numerical": {
                            "@value": "",
                            "@type": "xsd:decimal"
                        },
                        "Categories": [
                            {
                                "@context": {
                                    "Category Label": "https://schema.metadatacenter.org/properties/e6cf70d0-7e05-4122-b318-b4d93ee63c86",
                                    "Identification for category used in model": "https://schema.metadatacenter.org/properties/0ff233ed-291e-40d4-96b1-ef71bd5d5871"
                                },
                                "Category Label": {
                                    "@id": "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C173291",
                                    "rdfs:label": "Systemic Immunotherapy"
                                },
                                "Identification for category used in model": {
                                    "@value": "1"
                                }
                            },
                            {
                                "@context": {
                                    "Category Label": "https://schema.metadatacenter.org/properties/e6cf70d0-7e05-4122-b318-b4d93ee63c86",
                                    "Identification for category used in model": "https://schema.metadatacenter.org/properties/0ff233ed-291e-40d4-96b1-ef71bd5d5871"
                                },
                                "Category Label": {
                                    "@id": "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C92991",
                                    "rdfs:label": "Systemic Radiation Therapy"
                                },
                                "Identification for category used in model": {
                                    "@value": "0"
                                }
                            }
                        ],
                        "Input feature": {
                            "@id": "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C15698",
                            "rdfs:label": "Systemic Therapy"
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
                            "@value": "RTdose_subman"
                        },
                        "Type of input": {
                            "@value": "numerical"
                        },
                        "Minimum - for numerical": {
                            "@value": "0",
                            "@type": "xsd:decimal"
                        },
                        "Maximum - for numerical": {
                            "@value": "60",
                            "@type": "xsd:decimal"
                        },
                        "Categories": [
                            {}
                        ],
                        "Input feature": {
                            "@id": "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C190594",
                            "rdfs:label": "Radiation Dose"
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
                            "@value": "RTdosesalivary"
                        },
                        "Type of input": {
                            "@value": "numerical"
                        },
                        "Minimum - for numerical": {
                            "@value": "0",
                            "@type": "xsd:decimal"
                        },
                        "Maximum - for numerical": {
                            "@value": "60",
                            "@type": "xsd:decimal"
                        },
                        "Categories": [
                            {}
                        ],
                        "Input feature": {
                            "@id": "http://purl.bioontology.org/ontology/LNC/MTHU049454",
                            "rdfs:label": "Radiation dose"
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
                        "@value": "Prediction model for tube feeding dependency during chemoradiotherapy for at least four weeks in head and neck cancer patients"
                    },
                    "Editor Note": {
                        "@value": "Short title: PEG tube feeding"
                    },
                    "Created by": {
                        "@value": "Willemsen A.C.H. et al."
                    },
                    "References to papers": [
                        {
                            "@value": "https://doi.org/10.1016/j.clnu.2019.11.033"
                        }
                    ],
                    "Contact email": {
                        "@value": "j.vansoest@maastrichtuniversity.nl"
                    },
                    "Creation date": {
                        "@value": "2020-08-01",
                        "@type": "xsd:date"
                    },
                    "References to code": [
                        {
                            "@value": ""
                        }
                    ],
                    "Software License": {},
                    "FAIRmodels image name": {
                        "@value": "jvsoest/willemsen_tubefeed"
                    }
                },
                "Outcome": {
                    "@id": "http://purl.bioontology.org/ontology/SNOMEDCT/61420007",
                    "rdfs:label": "Tube feeding of patient"
                },
                "Applicability criteria": [
                    {
                        "@value": "Patients diagnosed with locally advanced head and neck squamous cell carcinoma (LAHNSCC). Patients undergoing chemoradiotherapy (CRT) or bioradiotherapy (BRT) with curative intent. Patients who are starting primary or adjuvant radiotherapy combined with either cisplatin, carboplatin, or cetuximab."
                    }
                ],
                "Foundational model or algorithm used": {
                    "@id": "https://w3id.org/aio/LogisticRegression",
                    "rdfs:label": "Logistic Regression"
                },
                "Primary intended use(s)": [
                    {
                        "@value": "To support personalized decision-making on prophylactic gastrostomy insertion in head and neck cancer patients undergoing chemoradiotherapy."
                    }
                ],
                "Primary intended users": [
                    {
                        "@value": "Clinicians and healthcare providers treating head and neck cancer patients."
                    }
                ],
                "Out-of-scope use cases": [
                    {
                        "@value": "Not intended for predicting long-term tube feeding dependency beyond the acute treatment phase."
                    }
                ],
                "Data": [
                    {
                        "@value": ""
                    }
                ],
                "Human life": [
                    {
                        "@value": ""
                    }
                ],
                "Mitigations": [
                    {
                        "@value": ""
                    }
                ],
                "Risks and harms": [
                    {
                        "@value": ""
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
                            "@value": "njnjnj"
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
                                    "@id": "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C190594",
                                    "rdfs:label": "Radiation Dose"
                                },
                                "Volume": {
                                    "@value": "600"
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
                                    "@value": "0.748",
                                    "@type": "xsd:decimal"
                                },
                                "Measured metric (low 95% confidence interval)": {
                                    "@value": "0.701",
                                    "@type": "xsd:decimal"
                                },
                                "Measured metric (up 95% confidence interval)": {
                                    "@value": "0.796"
                                },
                                "Acceptance level": {
                                    "@value": "",
                                    "@type": "xsd:decimal"
                                },
                                "Additional information (if needed)": {
                                    "@value": ""
                                }
                            },
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
                                    "@id": "http://purl.obolibrary.org/obo/STATO_0000619",
                                    "rdfs:label": "negative predictive value"
                                },
                                "Measured metric (mean value)": {
                                    "@value": "0.64",
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
                        "Link to dataset": {},
                        "Link to reference paper": {
                            "@id": "https://doi.org/10.1016/j.clnu.2019.11.033"
                        },
                        "Notes": {
                            "@value": "Data were collected in patients with LAHNSCC starting CRT/BRT in Maastricht University Medical Center (MUMCþ) and the University Medical Center Utrecht (UMCU) between January 1st 2013 and December 31st 2016."
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

        # check categorical data
        allowed_values = get_categorical_values(f,"Tclassification")#set(f["Input data"]["Tclassification"]["values"].values())

        if isinstance(data, list):
            for i in range(len(data)):
                if "Tclassification" not in data[i]:
                    raise ValueError(f"Missing Tclassification in item {i}")
                if data[i]["Tclassification"] not in allowed_values:
                    raise ValueError(f"Invalid Tclassification value in item {i}: {data[i]['Tclassification']}")

                # Apply transformation
                data[i]['Tclassification'] = 1 if int(data[i]['Tclassification']) in [2, 3] else 0
        else:
            if "Tclassification" not in data:
                raise ValueError("Missing Tclassification")
            if data["Tclassification"] not in allowed_values:
                raise ValueError(f"Invalid Tclassification value: {data['Tclassification']}")

            # Apply transformation
            data['Tclassification'] = 1 if int(data['Tclassification']) in [2, 3] else 0
        #
        # Regional Lymph nodes TNM finding
        # Extract allowed values from JSON
        allowed_values = get_categorical_values(f,"Nclassification")
        #set(f["features"]["Nclassification"]["values"].values())

        if isinstance(data, list):
            for i in range(len(data)):
                if "Nclassification" not in data[i]:
                    raise ValueError(f"Missing Nclassification in item {i}")
                if data[i]["Nclassification"] not in allowed_values:
                    raise ValueError(f"Invalid Nclassification value in item {i}: {data[i]['Nclassification']}")

                # Apply transformation
                data[i]['Nclassification'] = 1 if int(data[i]['Nclassification']) in [2, 3] else 0
        else:
            if "Nclassification" not in data:
                raise ValueError("Missing Nclassification")
            if data["Nclassification"] not in allowed_values:
                raise ValueError(f"Invalid Nclassification value: {data['Nclassification']}")

            # Apply transformation
            data['Nclassification'] = 1 if int(data['Nclassification']) in [2, 3] else 0

        # WHO performance status finding
        # Extract allowed values from JSON
        allowed_values = get_categorical_values(f,"PS")#set(f["features"]["PS"]["values"].values())

        if isinstance(data, list):
            for i in range(len(data)):
                if "PS" not in data[i]:
                    raise ValueError(f"Missing PS in item {i}")
                if data[i]["PS"] not in allowed_values:
                    raise ValueError(f"Invalid PS value in item {i}: {data[i]['PS']}")

                # Apply transformation
                data[i]['PS'] = 1 if int(data[i]['PS']) >0 else 0
        else:
            if "PS" not in data:
                raise ValueError("Missing PS")
            if data["PS"] not in allowed_values:
                raise ValueError(f"Invalid PS value: {data['PS']}")

            # Apply transformation
            data['PS'] = 1 if int(data['PS']) > 0  else 0

        # Systemic therapy
        allowed_values = get_categorical_values(f,"Systherapy")#set(f["features"]["Systherapy"]["values"].values())

        if isinstance(data, list):
            for i in range(len(data)):
                if "Systherapy" not in data[i]:
                    raise ValueError(f"Missing Systherapy in item {i}")
                if data[i]["Systherapy"] not in allowed_values:
                    raise ValueError(f"Invalid Systherapy value in item {i}: {data[i]['Systherapy']}")

        else:
            if "Systherapy" not in data:
                raise ValueError("Missing Systherapy")
            if data["Systherapy"] not in allowed_values:
                raise ValueError(f"Invalid Systherapy value: {data['Systherapy']}")

        # Texture modified diet or tube feeding
        allowed_values = get_categorical_values(f,"TF")#set(f["features"]["TF"]["values"].values())

        if isinstance(data, list):
            for i in range(len(data)):
                if "TF" not in data[i]:
                    raise ValueError(f"Missing TF in item {i}")
                if data[i]["TF"] not in allowed_values:
                    raise ValueError(f"Invalid TF value in item {i}: {data[i]['TF']}")

        else:
            if "TF" not in data:
                raise ValueError("Missing TF")
            if data["TF"] not in allowed_values:
                raise ValueError(f"Invalid TF value: {data['TF']}")

        # Tumorlocation
        allowed_values = get_categorical_values(f,"Tumorlocation") #set(f["features"]["Tumorlocation"]["values"].values())

        if isinstance(data, list):
            for i in range(len(data)):
                if "Tumorlocation" not in data[i]:
                    raise ValueError(f"Missing Tumorlocation in item {i}")
                if data[i]["Tumorlocation"] not in allowed_values:
                    raise ValueError(f"Invalid Tumorlocation value in item {i}: {data[i]['Tumorlocation']}")

        else:
            if "Tumorlocation" not in data:
                raise ValueError("Missing Tumorlocation")
            if data["Tumorlocation"] not in allowed_values:
                raise ValueError(f"Invalid Tumorlocation value: {data['Tumorlocation']}")

        #     ## check the values in numerical variables
        #
            # BMI
        min,max = get_range(f,"BMI")
        # min=f["features"]["BMI"]["min"]
        # max = f["features"]["BMI"]["max"]

        if isinstance(data, list):
            for i in range(len(data)):
                if "BMI" not in data[i]:
                    raise ValueError(f"Missing BMI in item {i}")
                if not isinstance(data[i]["BMI"], (int, float)):
                    raise TypeError(f"Invalid BMI type in item {i}, expected a number")
                if not (min<data[i]["BMI"] < max):
                    raise ValueError(f"Invalid BMI value in item {i}: {data[i]['BMI']}")

        else:
            if "BMI" not in data:
                raise ValueError("Missing BMI")
            if not isinstance(data["BMI"], (int, float)):
                raise TypeError(f"Invalid BMI type, expected a number")
            if not (min<data["BMI"]<max):
                #return {"error": f"Invalid BMI value: {data['BMI']}"}
                raise ValueError(f"Invalid BMI value in item: {data['BMI']} (Allowed range: {min}-{max})")
    #
        # WeightLoss
        min, max= get_range(f,"WeightLoss")
        # min = f["features"]["WeightLoss"]["min"]
        # max = f["features"]["WeightLoss"]["max"]

        if isinstance(data, list):
            for i in range(len(data)):
                if "WeightLoss" not in data[i]:
                    raise ValueError(f"Missing WeightLoss in item {i}")
                if not isinstance(data[i]["WeightLoss"], (int, float)):
                    raise TypeError(f"Invalid WeightLoss type in item {i}, expected a number")
                if not (min < data[i]["WeightLoss"] < max):
                    raise ValueError(f"Invalid WeightLoss value in item {i}: {data[i]['WeightLoss']}")

        else:
            if "WeightLoss" not in data:
                raise ValueError("Missing WeightLoss")
            if not isinstance(data["WeightLoss"], (int, float)):
                raise TypeError(f"Invalid WeightLoss type, expected a number")
            if not (min < data["WeightLoss"] < max):
                raise ValueError(f"Invalid WeightLoss value: {data['WeightLoss']}")

        # # RTdose_subman
        min, max=get_range(f,"RTdose_subman")
        # min = f["features"]["RTdose_subman"]["min"]
        # max = f["features"]["RTdose_subman"]["max"]

        if isinstance(data, list):
            for i in range(len(data)):
                if "RTdose_subman" not in data[i]:
                    raise ValueError(f"Missing RTdose_subman in item {i}")
                if not isinstance(data[i]["RTdose_subman"], (int, float)):
                    raise TypeError(f"Invalid RTdose_subman type in item {i}, expected a number")
                if not (min < data[i]["RTdose_subman"] < max):
                    raise ValueError(f"Invalid RTdose_subman value in item {i}: {data[i]['RTdose_subman']}")

        else:
            if "RTdose_subman" not in data:
                raise ValueError("Missing RTdose_subman")
            if not isinstance(data["RTdose_subman"], (int, float)):
                raise TypeError(f"Invalid RTdose_subman type, expected a number")
            if not (min < data["RTdose_subman"] < max):
                raise ValueError(f"Invalid RTdose_subman value: {data['RTdose_subman']}")

        # # RTdosesalivary
        min, max=get_range(f,"RTdosesalivary")
        # min = f["features"]["RTdosesalivary"]["min"]
        # max = f["features"]["RTdosesalivary"]["max"]

        if isinstance(data, list):
            for i in range(len(data)):
                if "RTdosesalivary" not in data[i]:
                    raise ValueError(f"Missing RTdosesalivary in item {i}")
                if not isinstance(data[i]["RTdosesalivary"], (int, float)):
                    raise TypeError(f"Invalid RTdosesalivary type in item {i}, expected a number")
                if not (min < data[i]["RTdosesalivary"] < max):
                    raise ValueError(f"Invalid RTdosesalivary value in item {i}: {data[i]['RTdosesalivary']}")

        else:
            if "RTdosesalivary" not in data:
                raise ValueError("Missing RTdosesalivary")
            if not isinstance(data["RTdosesalivary"], (int, float)):
                raise TypeError(f"Invalid RTdosesalivary type, expected a number")
            if not (min < data["RTdosesalivary"] < max):
                raise ValueError(f"Invalid RTdosesalivary value: {data['RTdosesalivary']}")

        return data
        #except Exception as e:

         #   return {"error": f"Unexpected error: {str(e)}"}

        # perform log transformation on the gtv value which is in the data list/dictionary
        # if isinstance(data, list):
        #     for i in range(len(data)):
        #         data[i]['gtv'] = log(data[i]['gtv'])
        # else:
        #     data['gtv'] = log(data['gtv'])


if __name__ == "__main__":
    model_obj = willemsen_tubefeed()
    model_obj.get_input_parameters()
    print(model_obj.predict(
        {
            "BMI": 19.5,
            "WeightLoss": -8,
            "TF": '1',
            "PS": '0',
            "Tumorlocation": '1',
            "Tclassification": '2',
            "Nclassification": '2',
            "Systherapy": '0',
            "RTdose_subman": 36,
            "RTdosesalivary": 29
        }
    ))