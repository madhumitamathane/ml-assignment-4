import sys
import pandas as pd
import textwrap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.base import clone

# This is where we'll put constants
DEFAULT_PATH = "./Concrete_Data.xls"

def get_data_xy(data):
    X = data.iloc[:, 0:7]
    y = data.iloc[:,8]
    return X, y

def clean_data(data):
    data = data.dropna()
    return data

def read_data(path):
    dataset = pd.read_excel(path, header = 0)
    return dataset

def test_model_split(model, X, y):
    """Run a test on the given split. It works off of a clone of the given model."""
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)
    model = clone(model)
    model.fit(train_X, train_y)
    return test_model(model, test_X, test_y)
    
def test_model (model, test_X, test_y):
    """Get performance metrics based on the model's prediction results."""
    prediction = model.predict(test_X)
    
    mse = mean_squared_error(test_y, prediction)
    var_score = r2_score(test_y, prediction)
    y_bar_squared = (sum(test_y)/float(len(test_y)))**2
    mse_per = mse / y_bar_squared
    
    return (mse, mse_per, var_score)

def print_test_results (results):
    mse, mse_per, var_score = results
    print("MSE:")
    print(textwrap.indent(str(mse), " " * 4))
    
    print("")
    print("MSE%:")
    print(textwrap.indent(str(mse_per), " " * 4))
    
    print("")
    print("Variance Score:")
    print(textwrap.indent(str(var_score), " " * 4))

def prepare_models(models, X, y):
    """Prepare the given models and print training results"""
    print("Testing against training data with 75%-25% split...")
    print("")

    for model_name, model in models:
        print("'{0}' classifier".format(model_name))
        print("--------------------------------------")

        test_results = test_model_split(model, X, y)
        print_test_results(test_results)
        print("")
    return models

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH

    print("Reading data from '{0}'...".format(path))
    print("")

    dataset = read_data(path)

    X, y = get_data_xy(dataset)

    models = prepare_models(
        [
            ("Linear Regression (Ridge)", Ridge(alpha = 0.5)),
        ],
        X,
        y
    )

if __name__ == "__main__":
    main()