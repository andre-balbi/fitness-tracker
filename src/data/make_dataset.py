import pandas as pd
from glob import glob
import re

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------

single_file_acc = pd.read_csv(
    "../../data/raw/A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)

single_file_gyr = pd.read_csv(
    "../../data/raw/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)

# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------

files = glob("../../data/raw/*.csv")
len(files)

# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------

# Extracting 3 values of file name:
# participant,
# label (exercise) and
# categoty (light or heavy)

f = files[0]
f.split("-")[0]
f.split("-")[1]
f.split("-")[2].split("_")[0]

data_path = "../../data/raw\\"

# Extrating the participant
participant = f.split("-")[0].replace(data_path, "")
# Extracting the label
label = f.split("-")[1]

# Removing number in any string
def remove_numbers(string):
    return re.sub(r"\d+", "", string)


category = remove_numbers(f.split("-")[2].split("_")[0])  # Extracting the category

# Adding extra columns
df = pd.read_csv(f)
df["participant"] = f.split("-")[0].replace(data_path, "")
df["label"] = f.split("-")[1]
df["category"] = remove_numbers(f.split("-")[2].split("_")[0])

# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

# Create df to store acc anf gyr
acc_df = pd.DataFrame()
gyr_df = pd.DataFrame()

# Creating a set to increment alfter each file is read
acc_set = 1
gyr_set = 1

# Lopping through all files
for f in files:
    # Reading the file
    df = pd.read_csv(f)
    # Extracting the participant
    participant = f.split("-")[0].replace(data_path, "")
    # Extracting the label
    label = f.split("-")[1]
    # Removing number in any string
    category = remove_numbers(f.split("-")[2].split("_")[0])  # Extracting the category
    df["participant"] = participant
    df["label"] = label
    df["category"] = category
    # If exist the word 'accelerometer' in the file name
    if "Accelerometer" in f:
        # Adding to acc_df
        df["set"] = acc_set
        acc_df = pd.concat([acc_df, df])
        acc_set += 1
    if "Gyroscope" in f:
        # Adding to gyr_df
        df["set"] = gyr_set
        gyr_df = pd.concat([gyr_df, df])
        gyr_set += 1

# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------
acc_df.info()

# Converting UNIX time to datetime
pd.to_datetime(df["epoch (ms)"], unit="ms")

# Conveting object to datetime
pd.to_datetime(df["time (01:00)"]).dt.weekday

# Set time to index
acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

# Removing columns
del acc_df["epoch (ms)"]
del acc_df["time (01:00)"]
del acc_df["elapsed (s)"]

del gyr_df["epoch (ms)"]
del gyr_df["time (01:00)"]
del gyr_df["elapsed (s)"]


acc_df.query('participant == "E"').query('label == "row" ').query(
    'category == "medium" '
)


# --------------------------------------------------------------
# Turn into a single function (read_data_from_files)
# --------------------------------------------------------------

files = glob("../../data/raw/*.csv")
data_path = "../../data/raw\\"

# Removing number in any string
def remove_numbers(string):
    return re.sub(r"\d+", "", string)


def read_data_from_files(files: list, data_path: str) -> tuple:
    """
    Read data from a list of files and extract & clean accelerometer and gyroscope data.

    Args:
        files (list): List of file paths to read data from.
        data_path (str): Path to the data directory.

    Returns:
        tuple: A tuple containing two DataFrames - `acc_df` (accelerometer data) and `gyr_df` (gyroscope data).
    """
    # Create empty DataFrames to store accelerometer and gyroscope data
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    # Set initial values for set counters
    acc_set = 1
    gyr_set = 1

    # Loop through all files
    for f in files:
        # Reading the file
        df = pd.read_csv(f)

        # Extracting participant, label, and category from the file name
        participant = f.split("-")[0].replace(data_path, "")
        label = f.split("-")[1]
        category = remove_numbers(f.split("-")[2].split("_")[0])

        # Add participant, label, and category columns to the DataFrame
        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        # Check if the file contains accelerometer data
        if "Accelerometer" in f:
            # Add set number to the DataFrame
            df["set"] = acc_set
            # Concatenate the DataFrame to the acc_df
            acc_df = pd.concat([acc_df, df])
            # Increment the acc_set counter
            acc_set += 1

        # Check if the file contains gyroscope data
        if "Gyroscope" in f:
            # Add set number to the DataFrame
            df["set"] = gyr_set
            # Concatenate the DataFrame to the gyr_df
            gyr_df = pd.concat([gyr_df, df])
            # Increment the gyr_set counter
            gyr_set += 1

    # Set time as the DataFrame index
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    # Remove unnecessary columns from the DataFrames
    columns_to_remove = ["epoch (ms)", "time (01:00)", "elapsed (s)"]
    acc_df.drop(columns_to_remove, axis=1, inplace=True)
    gyr_df.drop(columns_to_remove, axis=1, inplace=True)

    return acc_df, gyr_df


acc_df, gyr_df = read_data_from_files(files, data_path)


# --------------------------------------------------------------
# Merging datasets in a single dataframe
# --------------------------------------------------------------

# Selecting just first 3 columns from acc_df
data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)

# Rename columns
data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]

# Dropping the nan values
# There are two sensors measuring at diferent frequencies
# The change that the sensors measurement exact at same time is small
data_merged
data_merged.dropna(inplace=False)

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

aggregation = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last",
}


# Resample the DataFrame
data_merged[:1000].resample(rule="200ms").agg(aggregation, errors="ignore")

# tGenerates a list called df_by_days, where each element represents a DataFrame
# for a specific day. The groupby() operation allows you to group the original DataFrame
# data_merged by day, and then the list comprehension extracts and collects the
# DataFrames into the df_by_days list
df_by_days = [df_by_day for day, df_by_day in data_merged.groupby(pd.Grouper(freq="D"))]

# Takes the list of DataFrames df_by_days, performs resampling and aggregation on each
# DataFrame, drops missing values, and then concatenates the resampled and aggregated
# DataFrames into a single DataFrame called data_resampled.
data_resampled = pd.concat(
    [
        df.resample(rule="200ms").agg(aggregation, errors="ignore").dropna()
        for df in df_by_days
    ]
)
# this code takes the original DataFrame data_merged, groups it by day,
# and then performs resampling, aggregation, and concatenation on each daily DataFrame
# to create a single DataFrame data_resampled that contains the resampled and aggregated
# data for each day.

# --------------------------------------------------------------
# Turn into a single function (resample_and_aggregate_data)
# --------------------------------------------------------------


def resample_and_aggregate_data(
    acc: pd.DataFrame,
    gyr: pd.DataFrame,
    columns_name: list,
    rule: str,
    aggregation: dict,
) -> pd.DataFrame:
    """
    Resample and aggregate accelerometer and gyroscope data based on a specified rule and aggregation dictionary.

    Args:
        acc (pd.DataFrame): The accelerometer data.
        gyr (pd.DataFrame): The gyroscope data.
        columns_name (list): The column names for the merged DataFrame.
        rule (str): The resampling rule.
        aggregation (dict): The aggregation dictionary specifying the columns and aggregation methods.

    Returns:
        pd.DataFrame: The resampled and aggregated DataFrame.
    """
    # Selecting just the first 3 columns from acc_df and combining with gyr_df
    data_merged = pd.concat([acc.iloc[:, :3], gyr], axis=1)

    # Rename columns
    data_merged.columns = columns_name

    # Group by day
    df_by_days = [
        df_by_day for day, df_by_day in data_merged.groupby(pd.Grouper(freq="D"))
    ]

    # Resample and concate each day's DataFrame
    data_resampled = pd.concat(
        [
            df.resample(rule=rule).agg(aggregation, errors="ignore").dropna()
            for df in df_by_days
        ]
    )

    data_resampled["set"] = data_resampled["set"].astype("int")

    return data_resampled


columns_name = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]

aggregation = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last",
}

rule = "200ms"

data_resampled = resample_and_aggregate_data(
    acc_df, gyr_df, columns_name, rule, aggregation
)

data_resampled.info()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

data_resampled.to_pickle("../../data/interim/01_data_resampled.pkl")
