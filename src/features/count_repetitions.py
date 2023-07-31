import numpy as np
import pandas as pd
from typing import Union, List, Tuple
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_resampled.pkl")
df = df[df["label"] != "rest"]

# Calculating acc_r and gyr_r
acc_r = df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2
gyr_r = df["gyr_x"] ** 2 + df["gyr_y"] ** 2 + df["gyr_z"] ** 2

df["acc_r"] = np.sqrt(acc_r)
df["gyr_r"] = np.sqrt(gyr_r)

# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------

# Creating 5 Dataframes for each exercise
bench_df = df[df["label"] == "bench"]
squat_df = df[df["label"] == "squat"]
row_df = df[df["label"] == "row"]
ohp_df = df[df["label"] == "ohp"]
dead_df = df[df["label"] == "dead"]


# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------

# Looping through each label (exercises) and ploting for each sensor (acc and gyro)
list_df = [bench_df, squat_df, row_df, ohp_df, dead_df]

for plot_df in list_df:
    cat = plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["category"].unique()[0]
    label = plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["label"].unique()[0]
    set = plot_df["set"].unique()[0]
    plot_df[plot_df["set"] == plot_df["set"].unique()[0]].drop(
        "set", axis=1
    ).reset_index(drop=True).plot(
        subplots=True, figsize=(12, 12), title=f"{label} - {cat} - set {set}"
    )

# --------------------------------------------------------------
# Apply LowPassFilter
# --------------------------------------------------------------

bench_set = bench_df[bench_df["set"] == bench_df["set"].unique()[0]]
squat_set = squat_df[squat_df["set"] == squat_df["set"].unique()[0]]
row_set = row_df[row_df["set"] == row_df["set"].unique()[0]]
ohp_set = ohp_df[ohp_df["set"] == ohp_df["set"].unique()[0]]
dead_set = dead_df[dead_df["set"] == dead_df["set"].unique()[0]]

# Before the LowPassFilter is applied, the data is plotted
# to see if the filter is working correctly
dead_set["acc_r"].reset_index(drop=True).plot(label="raw")

# Configure LowPassFilter
fs = 1000 / 200
LowPass = LowPassFilter()
cutoff = 0.5
columns = "acc_r"
order = 10

# After applying the filter, the data is plotted again
temp_data = LowPass.low_pass_filter(
    dead_set.reset_index(drop=True),
    col=columns,
    sampling_frequency=fs,
    cutoff_frequency=cutoff,
    order=order,
)
temp_data[columns + "_lowpass"].plot(label="filtered")

# Find the local maxima (peaks) in the filtered data.
idx_max = argrelextrema(
    data=temp_data[columns + "_lowpass"].values, comparator=np.greater
)
peaks = temp_data[columns + "_lowpass"].iloc[idx_max]

# Peaks
plt.scatter(x=idx_max, y=peaks, c="red", s=100, label="peaks")

plt.legend()
plt.show()

# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------


def count_reps(
    dataset: pd.DataFrame,
    cutoff: Union[float, int],
    order: 10,
    columns: List[str],
    plot: bool = True,
) -> Tuple[pd.Series, pd.Index, int]:
    """
    Count the number of peaks in the low-pass filtered data for specified columns.

    Parameters:
        dataset (pd.DataFrame): The input dataset containing the time-series data.
        cutoff (Union[float, int]): The cutoff frequency for the low-pass filter.
        order (int): The order of the low-pass filter.
        columns (List[str]): A list of column names in the dataset to process.
        plot (bool, optional): If True, plot the raw and filtered data with identified peaks.
                                Default is True.

    Returns:
        Tuple[pd.Series, pd.Index, int]: A tuple containing:
            - peaks (pd.Series): A pandas Series containing the peak values for each column.
            - idx_max (pd.Index): The indices of the identified peaks in the filtered data.
            - num_peaks (int): The total number of peaks found in the data.

    Notes:
        The function applies a low-pass filter to the dataset using the specified cutoff frequency
        and order. It then identifies local maxima (peaks) in the filtered data and returns
        their values along with their indices. If `plot` is True, the function also generates
        a plot with the raw and filtered data, highlighting the identified peaks.
    """
    # `LowPass.low_pass_filter()` is a function that returns the low-pass filtered data.
    data = LowPass.low_pass_filter(
        dataset.reset_index(drop=True),
        col=columns,
        sampling_frequency=fs,  # Note: `fs` should be defined somewhere in the code.
        cutoff_frequency=cutoff,
        order=order,
    )[columns + "_lowpass"]

    # Find the local maxima (peaks) in the data.
    idx_max = argrelextrema(data=data.values, comparator=np.greater)
    peaks = data.iloc[idx_max]

    if plot:
        fig, ax = plt.subplots()
        # Ploting raw
        dataset[columns].reset_index(drop=True).plot(label="raw")
        # After applying the filter
        data.plot(label="filtered")
        # Plot the peaks
        plt.scatter(x=idx_max, y=peaks, c="red", s=100, label=f"peaks ({num_peaks})")
        label = dataset["label"].iloc[0].title()
        category = dataset["category"].iloc[0].title()
        ax.set_ylabel(f"{columns}_lowpass")
        plt.title(f"{label} - {category}: {len(peaks)} reps")

    return peaks, idx_max, len(peaks)


cutoff = 0.6
columns = "acc_r"
order = 10
dataset = squat_set

peaks, idx_max, num_peaks = count_reps(dataset, cutoff, order, columns)


peaks, idx_max, num_peaks = count_reps(bench_set, 0.5, 10, "acc_x")  # bench press
peaks, idx_max, num_peaks = count_reps(squat_set, 0.4, 10, "acc_r")  # squat
peaks, idx_max, num_peaks = count_reps(row_set, 0.75, 10, "gyr_x")  # row
peaks, idx_max, num_peaks = count_reps(ohp_set, 0.65, 10, "acc_r")  # ohp
peaks, idx_max, num_peaks = count_reps(dead_set, 0.5, 10, "acc_r")  # dead

# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------

# Creating a column to apply the number of reps for each label (medium or heavy)
df["reps"] = df["category"].apply(lambda x: 5 if x == "heavy" else 10)

rep_df = df.groupby(["label", "category", "set"])["reps"].max().reset_index()
rep_df["reps_pred"] = 0

# Looping over the df to apply the number of reps for each label (medium or heavy)
for s in df["set"].unique():
    subset = df[df["set"] == s]

    if subset["label"].iloc[0] == "bench":
        cutoff = 0.6
        columns = "acc_x"

    elif subset["label"].iloc[0] == "squat":
        cutoff = 0.4
        columns = "acc_r"

    elif subset["label"].iloc[0] == "row":
        cutoff = 0.65
        columns = "gyr_x"

    elif subset["label"].iloc[0] == "ohp":
        cutoff = 0.5
        columns = "acc_r"

    elif subset["label"].iloc[0] == "dead":
        cutoff = 0.4
        columns = "acc_r"

    peaks, idx_max, num_peaks = count_reps(subset, cutoff, 10, columns, plot=False)

    rep_df.loc[rep_df["set"] == s, "reps_pred"] = num_peaks

# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------

error = mean_absolute_error(y_true=rep_df["reps"], y_pred=rep_df["reps_pred"]).round(2)
print(f"MAE: {error}")

# Plot the diference
rep_df.groupby(["label", "category"])[["reps", "reps_pred"]].mean().plot.bar()
