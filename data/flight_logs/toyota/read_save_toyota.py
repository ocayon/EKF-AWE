import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


def align_whinch_to_kite(kite_df, whinch_df):
    # 1. Detect time columns
    for col in kite_df.columns:
        print(col)
    for col in whinch_df.columns:
        print(col)
    time_kite_col = [col for col in kite_df.columns if "Sync_Time" in col][0]
    time_whinch_col = [col for col in whinch_df.columns if "Sync_Time" in col][0]

    # 2. Prepare whinch dataframe for interpolation
    whinch_interp = (
        whinch_df.set_index(time_whinch_col)
        .interpolate(method="linear")
        .reindex(kite_df[time_kite_col], method="nearest", tolerance=0.1)
        .reset_index()
    )

    # 3. Replace interpolated time with kite time
    whinch_interp[time_kite_col] = kite_df[time_kite_col].values

    # 4. Drop redundant time column (if present)
    whinch_interp = whinch_interp.drop(columns=[time_whinch_col], errors="ignore")

    # 5. Concatenate on columns
    aligned_df = pd.concat(
        [
            kite_df.reset_index(drop=True),
            whinch_interp.drop(columns=[time_kite_col], errors="ignore"),
        ],
        axis=1,
    )

    return aligned_df


# Load Excel file (choose sheet if needed)
subset_winch_df = pd.read_excel(
    "data/flight_logs/toyota/4_2023_0322_98_new.xlsx", sheet_name="Winch"
)
subset_kite_df = pd.read_excel(
    "data/flight_logs/toyota/4_2023_0322_98_new.xlsx", sheet_name="Kite"
)
print(subset_winch_df.columns)
print(subset_kite_df.columns)

# Remove nan rows
subset_kite_df = subset_kite_df.dropna()
subset_winch_df = subset_winch_df.dropna()

print(len(subset_kite_df), len(subset_winch_df))

aligned_df = align_whinch_to_kite(subset_winch_df, subset_kite_df)


for col in aligned_df.columns:
    if "Sync_Time" in col:
        time = aligned_df[col]
    elif "Ground_Tension" in col:
        tension = aligned_df[col]
    elif "air_spd" in col:
        air_spd = aligned_df[col]
    elif "Line_Vel" in col:
        whinch_L_dot = aligned_df[col]
flight_data = pd.DataFrame()
for col in aligned_df.columns:
    if "Sync_Time" in col:
        flight_data["time"] = aligned_df[col]
    elif "Ground_Tension" in col:
        flight_data["ground_tether_force"] = aligned_df[col]
    elif "pos_E" in col:
        flight_data["kite_position_x"] = aligned_df[col]
    elif "pos_N" in col:
        flight_data["kite_position_y"] = aligned_df[col]
    elif "pos_D" in col:
        flight_data["kite_position_z"] = -aligned_df[col]
    elif "vel_y" in col:
        flight_data["kite_velocity_x"] = aligned_df[col]
    elif "vel_x" in col:
        flight_data["kite_velocity_y"] = aligned_df[col]
    elif "vel_z" in col:
        flight_data["kite_velocity_z"] = -aligned_df[col]
    elif "Line_Vel" in col:
        flight_data["tether_reelout_speed"] = aligned_df[col]
    elif "Ground_Length" in col:
        flight_data["tether_length"] = aligned_df[col]
    elif "air_spd" in col:
        flight_data["kite_apparent_windspeed"] = aligned_df[col]

# vel_x = flight_data["kite_position_x"].diff() / flight_data["time"].diff()
# vel_y = flight_data["kite_position_y"].diff() / flight_data["time"].diff()
# vel_z = flight_data["kite_position_z"].diff() / flight_data["time"].diff()

# flight_data["kite_velocity_x"] = savgol_filter(vel_x.fillna(0), 11, 3)
# flight_data["kite_velocity_y"] = savgol_filter(vel_y.fillna(0), 11, 3)
# flight_data["kite_velocity_z"] = savgol_filter(vel_z.fillna(0), 11, 3)

# Drop rows that have any NaN value
cleaned_fd = flight_data.dropna(axis=0)
cleaned_fd.to_csv(
    "data/flight_logs/toyota/2023-03-22_aligned_flight_data.csv",
    index=False,
    encoding="utf-8",
)
# 1. Set time as index and create new time grid (every 0.05s)
cleaned_fd = cleaned_fd.set_index("time")
cleaned_fd = cleaned_fd[cleaned_fd.index > 460]  # Ensure no negative time values
time_start = cleaned_fd.index.min()
time_end = cleaned_fd.index.max()
regular_time = pd.Series(data=np.arange(time_start, time_end, 0.05), name="time")

# 2. Interpolate to regular grid
resampled_fd = (
    cleaned_fd.interpolate(method="linear")
    .reindex(regular_time, method="nearest", tolerance=0.05)
    .interpolate(method="linear")
    .reset_index()
)

# 3. Rename index column back to 'time'
resampled_fd.rename(columns={"index": "time"}, inplace=True)

# 4. Save to CSV
resampled_fd.to_csv(
    "data/flight_logs/toyota/2023-03-22_aligned_flight_data.csv",
    index=False,
    encoding="utf-8",
)


fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
axs[0].plot(
    resampled_fd["time"],
    resampled_fd["ground_tether_force"],
    label="Tension",
)
axs[0].plot(
    cleaned_fd.index,
    cleaned_fd["ground_tether_force"],
    label="Tension (Raw)",
    linestyle="--",
)
axs[0].set_ylabel("Tension (N)")
axs[0].legend()
axs[1].plot(
    resampled_fd["time"],
    resampled_fd["kite_apparent_windspeed"],
    label="Air Speed",
)
axs[1].plot(
    cleaned_fd.index,
    cleaned_fd["kite_apparent_windspeed"],
    label="Air Speed (Raw)",
    linestyle="--",
)
axs[1].set_ylabel("Air Speed (m/s)")
axs[1].legend()
axs[2].plot(
    resampled_fd["time"],
    resampled_fd["tether_reelout_speed"],
    label="Whinch L Dot",
)
axs[2].plot(
    cleaned_fd.index,
    cleaned_fd["tether_reelout_speed"],
    label="Whinch L Dot (Raw)",
    linestyle="--",
)
axs[2].set_ylabel("Whinch L Dot (m/s)")
axs[2].set_xlabel("Time (s)")
axs[2].legend()

plt.tight_layout()
plt.show()
