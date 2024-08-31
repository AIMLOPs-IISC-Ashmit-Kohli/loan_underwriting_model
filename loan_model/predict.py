import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from loan_model import __version__ as _version
from loan_model.config.core import config
from loan_model.pipeline import loan_pipeline
from loan_model.processing.data_manager import load_pipeline
from loan_model.processing.data_manager import pre_pipeline_preparation
from loan_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
loan_pipe= load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    
    validated_data=validated_data.reindex(columns=config.model_config.features)
    #print(validated_data)
    results = {"predictions": None, "version": _version, "errors": errors}
    
    if not errors:

        predictions = loan_pipe.predict(validated_data)
        results = {"predictions": predictions,"version": _version, "errors": errors}
        print(results)
    
    else:
        print(results[errors])

    return results

if __name__ == "__main__":

    data_in={'Loan_ID':['LP00189'],'Gender':['Male'],'Married':['Yes'],'Dependents':['3+'],'Education':['Not Graduate'],'Self_Employed':['Yes'],
                'ApplicantIncome':[9000],'CoapplicantIncome':[5000.0],'LoanAmount':[350.0],'Loan_Amount_Term':[360.0],'Credit_History':[1.0],'Property_Area':['Semiurban']}
    
    make_prediction(input_data=data_in)
