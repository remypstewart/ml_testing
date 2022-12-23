import joblib
import pandas as pd 
import pytest

model = joblib.load('./artifacts/model.pkl')
transformer = joblib.load('./artifacts/transformer.pkl')

@pytest.fixture
def user_success():
    record = pd.DataFrame.from_dict({'age': [75], 'contact': ['cellular'], 'duration': [900], 'education': ['secondary'], 'job': ['retired'], 'marital': ['divorced']})
    record = transformer.transform(record)
    return record

@pytest.fixture
def user_failure():
    record = pd.DataFrame.from_dict({'age': [25], 'contact': ['cellular'], 'duration': [30], 'education': ['secondary'], 'job': ['student'], 'marital': ['single']})
    record = transformer.transform(record)
    return record

# mean age is 41 with std dev 11 years. 
def test_scale_success(user_success):
    assert float(user_success[0, 0]) > 0

def test_scale_failure(user_failure): 
    assert float(user_failure[0, 0]) < 0 

def test_successful_conversion(user_success): 
    s_predict = model.predict(user_success)
    assert s_predict == 'success'

def test_failed_conversion(user_failure): 
    f_predict = model.predict(user_failure)
    assert f_predict == 'failure'