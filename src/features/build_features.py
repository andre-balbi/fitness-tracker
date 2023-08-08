import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/02_outliers_removes_chauvenet.pkl")

predictor_columns = df.columns[:6].tolist()

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

# Concate the percentage of missing values per column with the number
# of missing values per column in a dataframe
def calculate_missing_values(df):
    """
    Calculate the percentage and total count of missing values for each column in the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with columns "Percentage of missing values (%)" and "Total of missing values",
                      containing information about missing values for each column.
    """
    missing_values_percentage = df.isnull().sum() / len(df) * 100
    total_missing_values = df.isnull().sum()

    df_missing_values = pd.concat(
        [missing_values_percentage, total_missing_values],
        axis=1,
        keys=["Percentage of missing values (%)", "Total of missing values"],
    ).sort_values(by="Percentage of missing values (%)", ascending=False)

    return df_missing_values


calculate_missing_values(df)

# Add a subset at random and plot it
set = 33
sensor = "gyr_y"
subset = df[df["set"] == set][sensor]
subset.plot()

# Interpolate linearly the NaN values
subset_interpolated = subset.interpolate()
# Find the indices of the interpolated values
interpolated_indices = subset_interpolated[
    subset_interpolated.notnull() & subset.isnull()
].index
# Plot the "gyr_y" values
plt.plot(subset.index, subset.values, label="Original Data")
# Plot the interpolated values as a scatter point
plt.scatter(
    interpolated_indices,
    subset_interpolated[interpolated_indices],
    color="red",
    label="Interpolated Values",
    marker="o",
)
plt.xlabel("Index")
plt.ylabel("gyr_y Values")
plt.title('Interpolated "gyr_y" Values')
plt.legend()
plt.show()

# Loop through all the columns and interpolate the NaN values
for col in predictor_columns:
    df[col] = df[col].interpolate()

calculate_missing_values(df)


# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

# Add a subset at random and plot it
set = 7
sensor = "acc_y"
subset = df[df["set"] == set][sensor]
subset.plot()

set = 21
sensor = "acc_y"
subset = df[df["set"] == set][sensor]
subset.plot()

# Time difference between the first and last value of the set 1
duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]
duration.seconds

# Loop through all the columns and calculate the set duration
for s in df["set"].unique():
    start = df[df["set"] == s].index[0]
    stop = df[df["set"] == s].index[-1]
    duration = stop - start
    df.loc[(df["set"] == s), "duration"] = duration.seconds

# Duration  for eac repetition
df.groupby("category")["duration"].describe()


duration_df = df.groupby("category")["duration"].mean()
duration_df.iloc[0] / 5  # Duration for a single heavy repetition (5 reps)
duration_df.iloc[1] / 10  # Duration for a single medium repetition (10 reps)

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()
LowPass = LowPassFilter()

# Defining sampling frequency (sample rate)
fs = 1000 / 200  # Samples per second (1000ms/20ms)
cutoff = 1  # Cutoff frequency of the filter (in Hz)

# Add low pass a single sensor
df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5)

# Comparing with the filter data
set = 7
sensor = "acc_y"

subset = df_lowpass[df_lowpass["set"] == set]
print(subset["label"][0])

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset[sensor].reset_index(drop=True), label="raw_data")
ax[1].plot(
    subset[sensor + "_lowpass"].reset_index(drop=True), label="butterworth_filter"
)
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)


# --------------------------------------------------------------
# Turn into a single function (plot_binary_outliers)
# --------------------------------------------------------------


def apply_and_plot_low_pass_filter(df, set_value, sensor_column, fs, cutoff, order=5):
    """
    Apply a low-pass filter to the specified sensor column in the DataFrame for a given dataset.
    Plot the raw data and the filtered data side by side.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        set_value (int): The dataset value to filter and plot.
        sensor_column (str): Name of the column (sensor) to apply the filter to and plot.
        fs (float): Sampling frequency (in Hz).
        cutoff (float): Cutoff frequency of the filter (in Hz).
        order (int, optional): Order of the filter. Defaults to 5.
    """
    df_lowpass = df.copy()
    LowPass = LowPassFilter()

    # Apply the low-pass filter
    df_lowpass = LowPass.low_pass_filter(
        df_lowpass, sensor_column, fs, cutoff, order=order
    )

    # Filter the subset of data for the specified set_value and sensor_column
    subset = df_lowpass[df_lowpass["set"] == set_value]

    # Print the label of the first row in the subset
    category = subset["label"].iloc[0]

    # Plot the raw data and the filtered data
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
    ax[0].plot(subset[sensor_column].reset_index(drop=True), label="raw_data")
    ax[1].plot(
        subset[sensor_column + "_lowpass"].reset_index(drop=True),
        label="butterworth_filter",
    )
    ax[0].legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True
    )
    ax[1].legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True
    )

    plt.suptitle(
        f"Comparison of Raw and Butterworth-Filtered Data for Set: {set_value}, category: {category} and Sensor: {sensor_column}",
        fontsize=16,
    )

    plt.show()


# Example usage:
# Assuming you have a DataFrame named "df" containing your data
# You can call the function like this:
fs = 1000 / 200
cutoff = 1.2
set_value = 7
sensor_column = "acc_y"

apply_and_plot_low_pass_filter(df, set_value, sensor_column, fs, cutoff, order=5)

# Loop through all the columns and apply the low-pass filter
for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]

df_lowpass.columns

# Add a subset at random after the lowpass filter
set = 7
sensor = "acc_y"
subset = df_lowpass[df_lowpass["set"] == set][sensor]
subset.plot()

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df_lowpass.copy()

pca = PrincipalComponentAnalysis()

# Get the variance of each feature
pca_values = pca.determine_pc_explained_variance(df_pca, predictor_columns)


# Determine the number of principal components using elbow technique
# Calculate the second derivative of the variance explained
second_derivative = np.diff(pca_values, n=2)

# Find the index of the inflection point (where the second derivative changes sign)
inflection_index = np.where(second_derivative < 0)[0][0]

# Plot the PCA values
plt.figure(figsize=(8, 8))
plt.plot(range(1, len(pca_values) + 1), pca_values)

# Plot a mark at the point of inflection
plt.scatter(
    inflection_index + 1,  # add 1 cause the index is 0 based
    pca_values[inflection_index],
    color="red",
    label="Inflection Point",
    marker="x",
    s=100,
    zorder=10,
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Variance Explained")
plt.legend()
plt.show()

# Summarazing all features into 3 principal components
# Apply a PCA given the number of components we have selected.
# We add new pca columns.
df_pca = pca.apply_pca(df_pca, predictor_columns, 3)
df_pca


# Visualize the PCA components
set = 45
subset = df_pca[df_pca["set"] == set]
subset[["pca_1", "pca_2", "pca_3"]].plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_squared = df_pca.copy()

# Square all
acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

df_squared

# Visuzalizing the squared attributes
set = 20
subset = df_squared[df_squared["set"] == set]
subset[["acc_r", "gyr_r"]].plot(subplots=True, figsize=(20, 10))

# Add the new cols
predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_squared.copy()

NumAbs = NumericalAbstraction()

# Window size for the rooling in 1s (abstraction)
ws = int(1000 / 200)  # Delta is 200ms

# Add new features, mean and std for all predictor columns
# We need to separate the data by set before applying the aggragation
df_temporal_list = []
for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], ws, "std")
    df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list)
df_temporal.info()

calculate_missing_values(df_temporal)

# Visualize the temporal abstraction
set = 44

df_temporal[df_temporal["set"] == set][
    ["acc_x", "acc_x_temp_mean_ws_5", "acc_x_temp_std_ws_5"]
].plot()

df_temporal[df_temporal["set"] == set][
    ["gyr_x", "gyr_x_temp_mean_ws_5", "gyr_x_temp_std_ws_5"]
].plot()

# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

df_freq = df_temporal.copy().reset_index()  # Must be a discrete index not time
FreqAbs = FourierTransformation()

ws = int(2800 / 200)  # Avarage of each repetition (2800ms)/ delta time (200ms)
fs = int(1000 / 200)  # Sampling frequency is (1000ms) / delta time (200ms)


# Get frequencies for a single sensor (acc_y) over a certain window.
df_freq = FreqAbs.abstract_frequency(
    df_freq, cols=["acc_y"], window_size=ws, sampling_rate=fs
)

df_freq.columns

# Visualize plots
set = 45
subset = df_freq[df_freq["set"] == set]

subset[["acc_y"]].plot()

subset[["acc_y_max_freq", "acc_y_freq_weighted", "acc_y_pse"]].plot(
    subplots=True, figsize=(20, 10)
)

subset[
    [
        # "acc_y_freq_0.0_Hz_ws_14",
        "acc_y_freq_0.357_Hz_ws_14",
        "acc_y_freq_0.714_Hz_ws_14",
        "acc_y_freq_1.071_Hz_ws_14",
        "acc_y_freq_1.429_Hz_ws_14",
        "acc_y_freq_1.786_Hz_ws_14",
        "acc_y_freq_2.143_Hz_ws_14",
        "acc_y_freq_2.5_Hz_ws_14",
    ]
].plot(figsize=(20, 10))

# Loop to get the frequency (FRF) for each set individually.
df_freq_list = []
for s in df_freq["set"].unique():
    print(f"Applying FRF for set: {s}")
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------
calculate_missing_values(df_freq)  # Missing values before
df_freq = df_freq.dropna()
calculate_missing_values(df_freq)  # missing values after

# Deleting part of the data (jumping a line)
# Reducing correlation between records to avoid overfitting
# This will cause a 50% loss in dataframe
df_freq = df_freq.iloc[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

df_cluster = df_freq.copy()

cluster_columns = ["acc_x", "acc_y", "acc_z"]
k_values = range(2, 10)  # Range of number of clusters
inertias = []  # Sum of squared distances of samples to their closest cluster center

# Lopping over the dataframe to create the clusters
for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)  # Train & make pred
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 8))
plt.plot(k_values, inertias, "-o")
plt.xlabel("Number of clusters")
plt.ylabel("Sum of squared distances")

# number of clusters = 5
kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

# Plot the clusters
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")
plt.legend()
plt.show()

# Plot acc data to compare
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for l in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == l]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=l)
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("../../data/interim/03_data_features.pkl")
