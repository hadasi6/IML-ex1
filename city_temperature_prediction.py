import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from polynomial_fitting import PolynomialFitting

def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date']).dropna().drop_duplicates()
    
    # # Remove invalid data (e.g., negative temperatures) 
    df = df[df["Temp"] > -50] #todo - validate

    # Add 'DayOfYear' column
    df["DayOfYear"] = df["Date"].dt.dayofyear

    return df

def _plot_avg_daily_temp(israel_data: pd.DataFrame, output_path: str = "Avg_Daily_Temp.png") -> None:
    """
    Plot the average daily temperature in Israel as a function of the day of the year.
    The plot is color-coded by year with a discrete color scale.

    Parameters
    ----------
    israel_data : pd.DataFrame
        Filtered dataset containing only samples from Israel.

    output_path : str
        Path to save the scatter plot image.
    """
    # Create discrete colorbar
    years = israel_data["Year"].unique()
    cmap = ListedColormap(plt.colormaps["viridis"](np.linspace(0, 1, len(years))))
    norm = BoundaryNorm(np.arange(min(years), max(years) + 1), cmap.N)

    # Scatter plot
    plt.figure(figsize=(15, 6))
    scatter = plt.scatter(
        israel_data["DayOfYear"], israel_data["Temp"], 
        c=israel_data["Year"], s=10, cmap=cmap, norm=norm, alpha=0.6
    )
    plt.title("The Average Daily Temperature in Israel", pad=20)
    plt.xlabel("Day of The Year")
    plt.ylabel("Temperature")
    plt.colorbar(scatter, label="Year")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_monthly_std(israel_data: pd.DataFrame, output_path: str = "Monthly_Temp_Std.png") -> None:
    """
    Plot the standard deviation of daily temperatures in Israel grouped by month.

    Parameters
    ----------
    israel_data : pd.DataFrame
        Filtered dataset containing only samples from Israel.

    output_path : str
        Path to save the bar plot image.
    """
    # Group by month and calculate standard deviation
    monthly_std = israel_data.groupby("Month")["Temp"].std()

    # Bar plot
    plt.figure(figsize=(10, 6))
    monthly_std.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Monthly Temperature Standard Deviation in Israel")
    plt.xlabel("Month")
    plt.ylabel("Standard Deviation (°C)")
    plt.xticks(
        range(12), 
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 
        rotation=45
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def _plot_country_monthly_temp(df: pd.DataFrame, output_path: str = "Country_Monthly_Temp.png") -> None:
    """
    Group data by country and month, calculate average and standard deviation of temperature,
    and save a line plot with error bars.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset containing temperature data for all countries.

    output_path : str
        Path to save the line plot image.
    """
    # Group by country and month, calculate mean and std
    grouped = df.groupby(["Country", "Month"]).agg(
        avg_temp=("Temp", "mean"),
        std_temp=("Temp", "std")
    ).reset_index()

     # Initialize the plot
    plt.figure(figsize=(12, 6))
    for country, data in grouped.groupby("Country"):
        plt.errorbar(
            data["Month"],
            data["avg_temp"],
            yerr=data["std_temp"],
            label=country,
            capsize=4,
            marker="o",
            linestyle="--",
        )

    # Add plot details
    plt.title("Average Monthly Temperature by Country", fontsize=14, pad=15)
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Mean Temperature", fontsize=12)
    plt.legend(title="Country", fontsize=10, loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path)
    plt.close()

def _split_train_test(data: pd.DataFrame, train_size: float = 0.75, random_state: int = 0):
    """
    Split dataset into training and testing sets using random sampling.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to split.

    train_size : float, optional
        Proportion of the dataset to include in the training set, by default 0.75.

    random_state : int, optional
        Random seed for reproducibility, by default 0.

    Returns
    -------
    train : pd.DataFrame
        Training set.

    test : pd.DataFrame
        Testing set.
    """
    train = data.sample(frac=train_size, random_state=random_state)
    test = data.drop(train.index)
    return train, test

def _fit_and_evaluate_polynomial_models(train: pd.DataFrame, test: pd.DataFrame, output_path: str = "test_error_by_degree.png"):
    """
    Fit polynomial models of degrees 1 to 10 and evaluate their performance on the test set.

    Parameters
    ----------
    train : pd.DataFrame
        Training dataset containing "DayOfYear" and "Temp" columns.

    test : pd.DataFrame
        Testing dataset containing "DayOfYear" and "Temp" columns.

    output_path : str
        Path to save the bar plot of test errors.
    """
    # Extract features and labels
    X_train, y_train = train["DayOfYear"].values, train["Temp"].values
    X_test, y_test = test["DayOfYear"].values, test["Temp"].values

    # Fit models and evaluate
    test_errors = []
    for k in range(1, 11):
        model = PolynomialFitting(k)
        model.fit(X_train, y_train)
        loss = round(model.loss(X_test, y_test), 2)
        test_errors.append((k, loss))

    # Print test errors
    for k, loss in test_errors:
        print(f"Degree {k}: Test Error = {loss}")

    # Save bar plot of test errors
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        [k for k, _ in test_errors],
        [loss for _, loss in test_errors],
        color="skyblue",
        edgecolor="black"
    )
    for bar, (_, loss) in zip(bars, test_errors):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{loss:.2f}", ha="center", va="bottom")
    plt.title("Test Error by Polynomial Degree")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Test Error (MSE)")
    plt.xticks(range(1, 11))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return test_errors

def _evaluate_model_on_countries(df: pd.DataFrame, model: PolynomialFitting, output_path: str = "lossByCountry.png") -> pd.DataFrame:
    """
    Evaluate a fitted model on different countries and save a bar plot of the errors.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset containing temperature data for all countries.

    model : PolynomialFitting
        A fitted polynomial model.

    output_path : str
        Path to save the bar plot of errors.

    Returns
    -------
    loss_df : pd.DataFrame
        DataFrame containing the loss for each country.
    """
    # Evaluate on other countries
    loss_by_country = []
    other_countries = df[df["Country"] != "Israel"].Country.unique()
    for country in other_countries:
        country_data = df[df["Country"] == country]
        X_country, y_country = country_data["DayOfYear"].values, country_data["Temp"].values
        loss = round(model.loss(X_country, y_country), 2)
        loss_by_country.append((country, loss))

    # Create DataFrame for losses
    loss_df = pd.DataFrame(loss_by_country, columns=["Country", "Loss"])

    # Save bar plot of errors
    plt.figure(figsize=(12, 8))
    bars = plt.bar(loss_df["Country"], loss_df["Loss"], color="skyblue", edgecolor="black")
    for bar, loss in zip(bars, loss_df["Loss"]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{loss:.2f}", ha="center", va="bottom", fontsize=10)
    plt.title("Model Error on Other Countries\n(Fitted on Israel's Data)", fontsize=16, pad=20, weight="bold")
    plt.xlabel("Country", fontsize=12)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return loss_df

if __name__ == '__main__':
    # Question 2 - Load and preprocessing of city temperature dataset
    df = load_data("city_temperature.csv")

    # Question 3 - Exploring data for specific country

    # Filter data for Israel
    israel_data = df[df["Country"] == "Israel"] 

    # Plot average daily temperature
    _plot_avg_daily_temp(israel_data, output_path="Avg_Daily_Temp.png")

    # Plot monthly temperature standard deviation
    _plot_monthly_std(israel_data, output_path="Monthly_Temp_Std.png")

    # Question 4 - Exploring differences between countries
    _plot_country_monthly_temp(df, output_path="Country_Monthly_Temp.png")

    # Question 5 - Fitting model for different values of `k`
    # split dataset
    train, test = _split_train_test(israel_data, train_size=0.75, random_state=0)
    test_errors = _fit_and_evaluate_polynomial_models(train, test, output_path="test_error_by_degree.png")

    # Question 6 - Evaluating fitted model on different countries
    # Fit the best model on Israel's data
    best_k = min(test_errors, key=lambda x: (x[1], x[0]))[0]
    best_model = PolynomialFitting(best_k)
    best_model.fit(israel_data["DayOfYear"].values, israel_data["Temp"].values)

    # Evaluate on other countries and save the plot
    loss_df = _evaluate_model_on_countries(df, best_model, output_path="lossByCountry.png")