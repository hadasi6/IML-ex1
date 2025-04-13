

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_train(X: pd.DataFrame, y: pd.Series):
    """
    preprocess training data.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data
    y: pd.Series

    Returns
    -------
    A clean, preprocessed version of the data
    """
    global train_columns

    # Create copies to avoid modifying original data
    X = X.copy()
    y = y.copy()

    # Remove irrelevant features
    X.drop(columns=["id", "date", "lat", "long"], inplace=True, errors='ignore')

    # Remove duplicates
    X.drop_duplicates(inplace=True)

    # Remove rows with missing values
    X.dropna(inplace=True)
    y = y.loc[X.index]

    # Remove unrealistic or extreme values
    mask = (
        (X["bedrooms"] > 0) &
        (X["bathrooms"] > 0) &
        (X["sqft_living"] >= 200) &
        (X["sqft_lot"] > 0) &
        (X["sqft_above"] > 0) &
        (X["sqft_basement"] >= 0) &
        (X["yr_built"] > 0) &
        (X["yr_renovated"] >= 0)
    )
    X = X[mask]
    y = y.loc[X.index] 

    # Drop outliers (basic filtering)
    X = X[X["bedrooms"] < 20]
    X = X[X["sqft_lot"] < 1250000]
    y = y.loc[X.index]

    # Feature engineering
    current_year = 2015
    X["house_age"] = current_year - X["yr_built"]
    X["was_renovated"] = (X["yr_renovated"] > 0).astype(int)

    # Drop original year columns
    X.drop(columns=["yr_built", "yr_renovated"], inplace=True)

    # Save feature structure for test set
    train_columns = X.columns.tolist()

    # Reset index
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    return X, y

    # # remove rows with missing values
    # X_df = X.dropna().drop_duplicates()
    # # remove rows with missing values in y
    # y_df = y.loc[X.index]

    # # remove rows with invalid values:

    # # remove rows with yr_renovated < yr_built
    # if "yr_renovated" in X_df.columns and "yr_built" in X_df.columns:
    #     invalid_rows = X_df[X_df["yr_renovated"] < X_df["yr_built"]].index
    #     X_df = X_df.drop(index=invalid_rows)
    #     y_df = y_df.loc[X_df.index]
    
    # # remove rows with sqft_living <= 0
    # if "sqft_living" in X_df.columns:
    #     invalid_rows = X_df[X_df["sqft_living"] <= 0].index
    #     X_df = X_df.drop(index=invalid_rows)
    #     y_df = y_df.loc[X_df.index]
    
    # # remove irrelevant columns: "id", "lat", "long", "date"
    # irrelevant_columns = ["id", "lat", "long", "date"]
    # X_df = X_df.drop(columns=irrelevant_columns, axis=1)
    # y_df = y_df.loc[X_df.index]

    

    


def preprocess_test(X: pd.DataFrame):
    """
    preprocess test data. You are not allowed to remove rows from X, but only edit its columns.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data

    Returns
    -------
    A preprocessed version of the test data that matches the coefficients format.
    """
    global train_columns

    X = X.copy()

    # Apply same transformations as in train
    X.drop(columns=["id", "date", "lat", "long"], inplace=True, errors='ignore')
    current_year = 2015
    X["house_age"] = current_year - X["yr_built"]
    X["was_renovated"] = (X["yr_renovated"] > 0).astype(int)
    X.drop(columns=["yr_built", "yr_renovated"], inplace=True)

    # Ensure test set has same columns as train
    for col in train_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[train_columns]

    X.reset_index(drop=True, inplace=True)
    return X


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
        # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    for feature in X.columns:
        # Compute Pearson Correlation manually
        cov = np.cov(X[feature], y)[0, 1]
        std_x = np.std(X[feature])
        std_y = np.std(y)
        pearson_corr = cov / (std_x * std_y)

        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(X[feature], y, alpha=0.5)
        plt.title(f"Feature: {feature}\nPearson Correlation: {pearson_corr:.2f}")
        plt.xlabel(feature)
        plt.ylabel("Response (y)")
        plt.grid()

        # Save plot
        plot_path = os.path.join(output_path, f"{feature}_correlation.png")
        plt.savefig(plot_path)
        plt.close()


if __name__ == '__main__':
    df = pd.read_csv("house_prices.csv")
    X, y = df.drop("price", axis=1), df.price

    # Question 2 - split train test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)


    # Question 3 - preprocessing of housing prices train dataset
    X_train, y_train = preprocess_train(X_train, y_train)

    # Question 4 - Feature evaluation of train dataset with respect to response
    feature_evaluation(X_train, y_train, output_path="feature_evaluation")

    # Question 5 - preprocess the test data
    X_test = preprocess_test(X_test)

    # Question 6 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

