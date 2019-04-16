import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
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
    parser.add_argument("-t, --test", dest="test_models", action="store_true")


def create_dataframe():
    """
        Creete the boston housing dataframe
    """
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
        Create the Linear regression model
    """
    linear_reg = LinearRegression()
    linear_reg.fit(X_data, y_data)
    return linear_reg


def get_correlation(dataframe):
    """
        Correlate our data
    """
    corr = dataframe.corr()
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
    plt.show()


def test_models(dataframe):

    print("---STARTING MODEL TESTS---")
    print("Testing a model with no scaling applied")
    r2_scores = []
    mean_squared_errors = []
    for i in range(50):
        # Create training data
        X_train, X_test, y_train, y_test = get_model_data(dataframe)
        linear_reg = create_model(X_train, y_train)

        # Predict y values and then score our model
        y_pred = linear_reg.predict(X_test)
        r2_scores.append(r2_score(y_pred, y_test))
        mean_squared_errors.append(mean_squared_error(y_pred, y_test))

    print("All tests have finished running.")
    print(f"r^2 score mean: {sum(r2_scores) / len(r2_scores)}")
    print(
        f"mean squared error mean: {sum(mean_squared_errors) / len(mean_squared_errors)}"
    )


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

    if args.test_models:
        test_models(dataframe)


if __name__ == "__main__":
    main()
