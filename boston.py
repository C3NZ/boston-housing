import argparse

import pandas as pd
from sklearn.datasets import load_boston

# Create a correlation heatmap for determining how highly correletated our
# predictor values are (the higher, the worse)

# Train the model using our training data (75% training, 25% testing)

# Make predictions based on our testing dataset

# Compute mean square error of our application and R squared score
# of our model.


def add_arguments(parser):
    """
        Add arguments to the arg parser
    """
    parser.add_argument("-d, --data", dest="dataframe", action="store_true")
    parser.print_help


def create_dataframe():
    # Load the boston dataset
    boston = load_boston()

    # Convert the boston house pricing data into a pandas dataframe
    bos_df = pd.DataFrame(boston.data)
    bos_df.columns = boston.feature_names
    bos_df["PRICE"] = boston.target


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
    print(args)
    dataframe = create_dataframe()


if __name__ == "__main__":
    main()
