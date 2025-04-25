from pathlib import Path

# Folder for plots
PLOTS_FOLDER = Path("plots")
PLOTS_FOLDER.mkdir(parents=True, exist_ok=True)

# Folder for data
DATA_FOLDER = Path("data")
DATA_FOLDER.mkdir(parents=True, exist_ok=True)

# Paths to datasets
HOUSE_PRICES_DATA_PATH = DATA_FOLDER / "house_prices.csv"
CITY_TEMPERATURE_DATA_PATH = DATA_FOLDER / "city_temperature.csv"

# Subfolders for specific plots
TEMPERATURE_PLOTS_FOLDER = PLOTS_FOLDER / "temperature"
TEMPERATURE_PLOTS_FOLDER.mkdir(parents=True, exist_ok=True)

HOUSE_PRICE_PLOTS_FOLDER = PLOTS_FOLDER / "house_prices"
HOUSE_PRICE_PLOTS_FOLDER.mkdir(parents=True, exist_ok=True)

# Paths for specific plots
FEATURE_EVAL_OUTPUT_PATH = HOUSE_PRICE_PLOTS_FOLDER / "feature_evaluation"
FEATURE_EVAL_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

MODEL_LOSS_PLOT_PATH = HOUSE_PRICE_PLOTS_FOLDER / "model_loss_vs_training_size.png" 
AVG_DAILY_TEMP_PLOT_PATH = TEMPERATURE_PLOTS_FOLDER / "Avg_Daily_Temp.png"
MONTHLY_TEMP_STD_PLOT_PATH = TEMPERATURE_PLOTS_FOLDER / "Monthly_Temp_Std.png"
COUNTRY_MONTHLY_TEMP_PLOT_PATH = TEMPERATURE_PLOTS_FOLDER / "Country_Monthly_Temp.png"
TEST_ERROR_PLOT_PATH = TEMPERATURE_PLOTS_FOLDER / "test_error_by_degree.png"
LOSS_BY_COUNTRY_PLOT_PATH = TEMPERATURE_PLOTS_FOLDER / "lossByCountry.png"