
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np

from loan_model.config.core import config
from loan_model.processing.features import *
from loan_model.processing.data_manager import pre_pipeline_preparation



def test_dependents_transformer(sample_input_data):
    # Given
    test_df = sample_input_data[0].copy()
    transformer = modeimputer(feature=config.model_config.dependents_var)
    
    assert pd.isnull(test_df.loc[228,config.model_config.dependents_var])

    # When
    subject = transformer.fit(test_df).transform(test_df)

    # Then
    assert subject.loc[228,config.model_config.dependents_var] == '0'
    
def test_loanamount_variable_transformer(sample_input_data):
    # Given
    test_df = sample_input_data[0].copy()
    transformer = meanimputer(feature=config.model_config.loanamount_var)
    
    assert pd.isnull(test_df.loc[322,config.model_config.loanamount_var])
    
    # When
    subject = transformer.fit(test_df).transform(test_df)

    # Then
    assert subject.loc[322,config.model_config.loanamount_var] == 137.03418803418805
    

def test_outlier_transformer(sample_input_data):
    # Given
    test_df = sample_input_data[0].copy()
    transformer = OutlierHandler(feature=config.model_config.applicantincome_var)
    
    Q1 = np.percentile(test_df.loc[:, config.model_config.applicantincome_var], 25)
    Q3 = np.percentile(test_df.loc[:, config.model_config.applicantincome_var], 75)
    deviation_allowed = 1.5*(Q3 - Q1)
    lower_bound = Q1 - deviation_allowed
    upper_bound = Q3 + deviation_allowed

    #Given
    assert len(test_df[test_df[config.model_config.applicantincome_var] > upper_bound]) > 0
    
    # When
    subject = transformer.fit(test_df).transform(test_df)

    # Then
    assert len(subject[subject[config.model_config.applicantincome_var] > upper_bound]) == 0
    

def test_columndropper_transformer(sample_input_data):
    # Given
    test_df = sample_input_data[0].copy()
    transformer = columnDropperTransformer(col=config.model_config.unused_fields)
    
    assert len(test_df.columns) == 12
    
    
    # When
    subject = transformer.fit(test_df).transform(test_df)

    # Then
    assert len(subject.columns) == 11
    
    
    
def test_mapper_education(sample_input_data):
    # Given
    test_df = sample_input_data[0].copy()
    transformer = Mapper(
        config.model_config.education_var,  config.model_config.education_mappings
    )
    
    assert set(test_df['Education'].unique()) == {'Graduate', 'Not Graduate'}

    # When
    subject = transformer.fit(test_df).transform(test_df)

    # Then
    assert set(subject['Education'].unique()) == {0,1} 
    
 
def test_dependents_ohe(sample_input_data):
    # Given
    test_df = sample_input_data[0].copy()

    transformer1 = modeimputer(feature=config.model_config.dependents_var)
    transformer2 = CustomOneHotEncoder(col=config.model_config.dependents_var)
    
    subject_1 = transformer1.fit(test_df).transform(test_df)
    assert list(subject_1[config.model_config.dependents_var].unique()).sort() == ['0', '2', '1', '3+'].sort()
    assert len(subject_1.columns) == 12
    # When
    
    subject_2 = transformer2.fit(subject_1).transform(subject_1)

    # Then
    assert len(subject_2.columns) == 15
    assert subject_2.loc[228,'0'] + subject_2.loc[228,'1'] + subject_2.loc[228,'1'] + subject_2.loc[228,'3+'] ==1 
  
    
