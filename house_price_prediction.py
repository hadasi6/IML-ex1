import plotly.graph_objects as go
import plotly.express as px
from typing import NoReturn
import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# import os

RANDOM_SEED = 0

def _add_columns(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns to the dataset.

    Parameters
    ----------
    X : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    pd.DataFrame
        The DataFrame with additional columns.
    """
     # Extract year of sale
    
    X['date'] = pd.to_datetime(X['date']).dt.year 
    # Calculate decade of construction 
    X['decade'] = (X['yr_built'] // 10) * 10  
    curr_yr = 2015
    X['renovated_last_decade'] = X['yr_renovated'].apply(
        lambda x: 1 if (x != 0) and ((curr_yr - x) <= 10) else 0
    )
    X['was_renovated'] = X['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
    X['bedrooms_to_living_ratio'] = X['bedrooms'] / X['sqft_living']
    return X

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
    # Add new columns
    X = _add_columns(X)

    # Remove irrelevant columns
    remove_columns = ['id', 'date', 'lat', 'long', 'yr_built', 'yr_renovated']
    X = X.drop(remove_columns, axis=1, errors='ignore')

    # Clean data: remove duplicates and missing values
    X.drop_duplicates(inplace=True) # Remove duplicates
    X.dropna(inplace=True) # Remove rows with missing values
    y = y.loc[X.index]
    y.dropna(inplace=True)  # Remove rows with missing values in y 
    X = X.loc[y.index]

    # X.dropna(inplace=True) # delete empty values & duplicates
    # X.drop_duplicates(inplace=True)
    # X = X.loc[y.dropna().index]
    # y = y.dropna()
    # y = y[X.index] # delete the same items from y


    # Remove unrealistic or extreme values
    mask = (
        (X['sqft_living'] <= X['sqft_lot']) &
        (X['bedrooms'] >= 0) &
        (X['bathrooms'] >= 0) &
        (X['floors'] >= 0) &
        (X['sqft_lot'] > 0) &
        (X['condition'].isin(range(1, 6))) &
        (X['decade'] > 0) &
        (X['view'].isin(range(5))) &
        (X['grade'].isin(range(1, 15))) &
        (X['sqft_above'] > 0) &
        (X['sqft_basement'] >= 0) &
        (X['waterfront'].isin([0, 1]))
    )
    X = X[mask]
    y = y.loc[X.index] 
    X.replace('nan', np.nan)
    y = y[X.index]

    return X, y
    
    
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
    # Add new columns
    X = _add_columns(X)

    # Remove irrelevant columns
    remove_columns = ['id', 'date', 'lat', 'long', 'yr_built', 'yr_renovated']
    X = X.drop(remove_columns, axis=1, errors='ignore')

    # Fill missing values with the mean of each column
    X.fillna(X.mean(), inplace=True)
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
    # correlations = {}
    for feature in X.columns:
        # Compute Pearson Correlation
        cov = np.cov(X[feature], y)[0, 1]
        std_x = np.std(X[feature])
        std_y = np.std(y)
        pearson_corr = cov / (std_x * std_y)
        # correlations[feature] = pearson_corr
        # Create scatter plot
        fig = px.scatter(
            x=X[feature],
            y=y,
            title=f"{feature} vs Price<br>Pearson Correlation: {pearson_corr:}",
            labels={"x": feature, "y": "Price"}
        )
        # Save plot
        fig.write_image(f"{output_path}/{feature}_vs_price.png")
        # fig.write_image(os.path.join(output_path, f"{feature}_vs_price.png"))
    # Sort features by absolute correlation
    # sorted_features = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    # Get the best and worst features
    # best_feature = sorted_features[0]
    # worst_feature = sorted_features[-1]
    # print(f"Best Feature: {best_feature[0]}, Correlation: {best_feature[1]:.3f}")
    # print(f"Worst Feature: {worst_feature[0]}, Correlation: {worst_feature[1]:.3f}")
    # print("All features sorted by absolute correlation:", sorted_features)

def _split_train_test(X: pd.DataFrame, y: pd.Series, train_size: float = 0.75, random_state: int = 0):
    """
    Split data into training and testing sets.
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    train_size : float
        Proportion of data to include in the training set.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
        Training and testing sets.
    """
    np.random.seed(random_state)
    train_indices = X.sample(frac=train_size, random_state=random_state).index
    test_indices = X.index.difference(train_indices)

    X_train, X_test = X.loc[train_indices], X.loc[test_indices]
    y_train, y_test = y.loc[train_indices], y.loc[test_indices]

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    df = pd.read_csv("house_prices.csv")
    X, y = df.drop("price", axis=1), df.price

    # Question 2 - split train test
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=RANDOM_SEED)
    X_train, X_test, y_train, y_test = _split_train_test(X, y, train_size=0.75, random_state=RANDOM_SEED)
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
    means, stds, percentages = [], [], list(range(10, 101))
    for p in percentages:
        losses = []
        for _ in range(10):
            sample_X = X_train.sample(frac=p / 100)
            sample_y = y_train.loc[sample_X.index]
            model = LinearRegression()
            model.fit(sample_X, sample_y)
            y_pred = model.predict(X_test)
            loss = np.mean((y_test - y_pred) ** 2)
            losses.append(loss)
        means.append(np.mean(losses))
        stds.append(np.std(losses))

    fig = go.Figure([
        go.Scatter(x=percentages, y=means, mode="lines+markers", name="Mean Loss"),
        go.Scatter(x=percentages, y=np.array(means) - 2 * np.array(stds),
                   name="Mean - 2*STD", line=dict(dash="dash")),
        go.Scatter(x=percentages, y=np.array(means) + 2 * np.array(stds),
                   name="Mean + 2*STD", line=dict(dash="dash"))
    ])
    fig.update_layout(title="Model Loss vs. Training Set Size",
                      xaxis_title="Training Set Size (%)",
                      yaxis_title="Mean Squared Error",
                      height=500)
    fig.show()
    fig.write_image("model_loss_vs_training_size.png")