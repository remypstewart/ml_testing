import pandas as pd 
import pytest

@pytest.fixture
def load_data(): 
    data = pd.read_parquet('./artifacts/marketing_data.pqt') 
    return data 

def test_minors(load_data): 
    for elems in set(load_data['age']): 
        assert elems > 17

def test_pos_time(load_data): 
    for elems in set(load_data['duration']): 
        assert elems >= 0 

def test_marital_cat(load_data):
    for elems in set(load_data['marital']): 
        assert elems in ('married', 'single', 'divorced')

def test_educ_cat(load_data): 
    for elems in set(load_data['education']): 
        assert elems in ('primary', 'secondary', 'tertiary', 'unknown')