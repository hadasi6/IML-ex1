import plotly.graph_objects as go
import plotly.express as px
from typing import NoReturn
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from config import HOUSE_PRICES_DATA_PATH, FEATURE_EVAL_OUTPUT_PATH, MODEL_LOSS_PLOT_PATH

SEED = 0
TRAIN_SIZE = 0.75
CURRENT_YEAR = 2015
NUM_REPEATS = 10
CONDITION_RANGE = range(1, 6)
VIEW_RANGE = range(5)
GRADE_RANGE = range(1, 15)


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
    X['renovated_last_decade'] = X['yr_renovated'].apply(
        lambda x: 1 if (x != 0) and ((CURRENT_YEAR - x) <= 10) else 0
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

    # Remove unrealistic or extreme values
    mask = (
        (X['sqft_living'] <= X['sqft_lot']) &
        (X['bedrooms'] >= 0) &
        (X['bathrooms'] >= 0) &
        (X['floors'] >= 0) &
        (X['sqft_lot'] > 0) &
        (X['condition'].isin(CONDITION_RANGE)) &
        (X['decade'] > 0) &
        (X['view'].isin(VIEW_RANGE)) &
        (X['grade'].isin(GRADE_RANGE)) &
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
    for feature in X.columns:
        # Compute Pearson Correlation
        cov = np.cov(X[feature], y)[0, 1]
        std_x = np.std(X[feature])
        std_y = np.std(y)
        pearson_corr = cov / (std_x * std_y)
        # Create scatter plot
        fig = px.scatter(
            x=X[feature],
            y=y,
            title=f"{feature} vs Price<br>Pearson Correlation: {pearson_corr:}",
            labels={"x": feature, "y": "Price"}
        )
        # Save plot
        fig.write_image(f"{output_path}/{feature}_vs_price.png")

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
    train_indices = X.sample(frac=train_size, random_state=random_state).index
    test_indices = X.index.difference(train_indices)

    X_train, X_test = X.loc[train_indices], X.loc[test_indices]
    y_train, y_test = y.loc[train_indices], y.loc[test_indices]

    return X_train, X_test, y_train, y_test

def _plot_loss_vs_training_size(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, output_path: str =MODEL_LOSS_PLOT_PATH) -> None:
    """
    Fit a linear model over increasing percentages of the training data and plot the loss.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.
    X_test : pd.DataFrame
        Testing feature matrix.
    y_test : pd.Series
        Testing target vector.
    output_path : str
        Path to save the plot image.
    """
    means, stds, percentages = [], [], list(range(10, 101))
    for p in percentages:
        losses = []
        for _ in range(NUM_REPEATS):
            sample_X = X_train.sample(frac=p / 100)
            sample_y = y_train.loc[sample_X.index]
            model = LinearRegression()
            model.fit(sample_X, sample_y)
            y_pred = model.predict(X_test)
            loss = np.mean((y_test - y_pred) ** 2)
            losses.append(loss)
        means.append(np.mean(losses))
        stds.append(np.std(losses))

    # Create the plot
    fig = go.Figure()

    # Add the main line (Mean Loss)
    fig.add_trace(go.Scatter(
        x=percentages,
        y=means,
        mode="lines+markers",
        name="Mean Loss",
        line=dict(color="blue", width=1.5),
        marker=dict(size=4.5)
    ))

    # Add the shaded area (Mean ± 2*STD)
    fig.add_trace(go.Scatter(
        x=percentages + percentages[::-1],  # Combine x values for upper and lower bounds
        y=(np.array(means) + 2 * np.array(stds)).tolist() + (np.array(means) - 2 * np.array(stds)).tolist()[::-1],
        fill="toself",
        fillcolor="rgba(0, 0, 255, 0.2)",  # Light blue with transparency
        line=dict(color="rgba(255,255,255,0)"),  # No border line
        hoverinfo="skip",
        name="Confidence Interval"
    ))

    # Update layout
    fig.update_layout(
        title=dict(
            text="Model Loss vs. Training Set Size",
            font=dict(size=16),  # Smaller font size
            x=0.5,  # Center the title
            xanchor="center"
        ),
        xaxis_title="Training Set Size (%)",
        yaxis_title="Mean Squared Error",
        height=500,  # Keep the height fixed
        width=700,  # Keep the width fixed
        template="plotly_white",
        showlegend=False  # Remove the legend
    )

    # Save the plot without the legend
    fig.write_image(output_path)

    # Add the legend back and reposition it
    fig.update_layout(
        showlegend=True,
        legend=dict(
            title="Legend",
            font=dict(size=10),  # Smaller font size for the legend
            x=0.95,  # Position the legend on the right edge of the plot
            y=0.95,  # Position the legend at the top of the plot
            bgcolor="rgba(255,255,255,0.5)",  # Semi-transparent background
            bordercolor="black",
            borderwidth=0.8
        )
    )
    # Save the plot
    fig.write_image(output_path)


if __name__ == '__main__':
    df = pd.read_csv(HOUSE_PRICES_DATA_PATH)
    X, y = df.drop("price", axis=1), df.price

    # split train test
    X_train, X_test, y_train, y_test = _split_train_test(X, y, train_size=TRAIN_SIZE, random_state=SEED)
    
    # preprocessing of housing prices train dataset
    X_train, y_train = preprocess_train(X_train, y_train)

    # Feature evaluation of train dataset with respect to response
    feature_evaluation(X_train, y_train, output_path=FEATURE_EVAL_OUTPUT_PATH)

    # preprocess the test data
    X_test = preprocess_test(X_test)

    # Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    _plot_loss_vs_training_size(X_train, y_train, X_test, y_test, output_path=MODEL_LOSS_PLOT_PATH)