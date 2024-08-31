from typing import Any, List, Optional

from pydantic import BaseModel
from loan_model.processing.validation import DataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[int]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [

                    {   'Loan_ID': 'LP00189',
                        'Gender': 'Male',
                        'Married': "Yes",
                        'Dependents': '3+',
                        'Education':'Not Graduate',
                        'Self_Employed': 'Yes',
                        'ApplicantIncome':9000,
                        'CoapplicantIncome': 5000.0,
                        'LoanAmount': 350.0,
                        'Loan_Amount_Term': 360.0,
                        'Credit_History': 1.0,
                        'Property_Area': 'Semiurban'
                        
                    }
                ]
                
            }
        }