import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Train the model using our training data (75% training, 25% testing)

# Make predictions based on our testing dataset

# Compute mean square error of our application and R squared score
# of our model.


def add_arguments(parser):
    """
        Add arguments to the arg parser
    """
    parser.add_argument("-d, --data", dest="dataframe", action="store_true")
    parser.add_argument("-c, --correlation", dest="correlation", action="store_true")
    parser.print_help


def create_dataframe():
    # Load the boston dataset
    boston = load_boston()

    # Convert the boston house pricing data into a pandas dataframe
    bos_df = pd.DataFrame(boston.data)
    bos_df.columns = boston.feature_names
    bos_df["PRICE"] = boston.target
    return bos_df


def get_model_data(dataframe):
    """
        Get the training and testing data
    """
    print("Creating model data")
    X_data = []
    y_data = []

    for row in dataframe.values:
        X_data.append(row[:-1])
        y_data.append(row[-1])

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    return train_test_split(X_data, y_data)


def create_model(X_data, y_data):
    """
        Linear regression data
    """
    print("Creating linear regression model")
    linear_reg = LinearRegression()
    linear_reg.fit(X_data, y_data)
    return linear_reg


def get_correlation(dataframe):
    """
        Correlate our data
    """

    X_train, X_test, y_train, y_test = get_model_data(dataframe)

    linear_reg = create_model(X_train, y_train)

    corr = dataframe.corr()
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
    plt.show()

def test_models(1)

def main():
    """
        Main function for handling the CLI interaction
    """
    # Instantiate the arg parser with a description
    parser = argparse.ArgumentParser(description="Working with the boston dataset")

    # build out our arg parser
    add_arguments(parser)
    # parse the arguments entered by the userhj
    args = parser.parse_args()
    dataframe = create_dataframe()

    if args.correlation:
        get_correlation(dataframe)


if __name__ == "__main__":
    main()
