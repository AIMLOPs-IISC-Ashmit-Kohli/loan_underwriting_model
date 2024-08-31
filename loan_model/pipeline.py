import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from loan_model.config.core import config
from loan_model.processing.features import *

# YOUR CODE HERE

loan_pipeline = Pipeline([

    ('Gender_mode_imputation', modeimputer(feature='Gender')),
    ('Married_mode_imputation', modeimputer(feature='Married')),
    ('Dependents_mode_imputation', modeimputer(feature='Dependents')),
    ('Credit_history_mode_imputation', modeimputer(feature='Credit_History')),
    ('Self_Employed_mode_imputation', modeimputer(feature='Self_Employed')),
    ('LoanAmount_mean_imputation', meanimputer(feature='LoanAmount')),
    ('Loan_Amount_Term_mean_imputation', meanimputer(feature='Loan_Amount_Term')),
    ('ApplicantIncome_OutlierHandler', OutlierHandler(feature='ApplicantIncome')),
    ('CoapplicantIncome_OutlierHandler', OutlierHandler(feature='CoapplicantIncome')),
    ('LoanAmount_OutlierHandler', OutlierHandler(feature='LoanAmount')),
    ('Gender_Mapping', Mapper('Gender', {'Male': 0, 'Female': 1})),
    ('Married_Mapping', Mapper('Married', {'Yes': 1, 'No': 0})),
    ('Self_Employed_Mapping', Mapper('Self_Employed', {'Yes': 1, 'No': 0})),
    ('Education_Mapping', Mapper('Education', {'Graduate': 1, 'Not Graduate': 0})),
    ('Dependents_OHE', CustomOneHotEncoder(col = 'Dependents')),
    ('Property_Area_OHE', CustomOneHotEncoder(col = 'Property_Area')),
    ('unused_column_dropper',columnDropperTransformer(col = 'Loan_ID')),
    #Scaling
    ('scaler', StandardScaler()),
    #Model_Fitting
    ('model_rf', RandomForestClassifier(n_estimators=500, max_depth=10,min_samples_split = 2,min_samples_leaf=2,max_features='sqrt',bootstrap=True, random_state=42))
])
