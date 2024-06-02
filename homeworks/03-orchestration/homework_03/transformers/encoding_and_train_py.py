if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd


@transformer
def transform(df, *args, **kwargs):
    categorical = ['PULocationID', 'DOLocationID']
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    target = 'duration'
    y_train = df[target].values
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print(lr.intercept_)
    


    return lr,dv


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'