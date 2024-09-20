import yaml
import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout, Input
import xarray as xr
from metrics import calculate_metrics, get_available_metrics
from keras.callbacks import EarlyStopping, TensorBoard, LearningRateScheduler
from keras.losses import MeanSquaredError
import datetime
import gc
from keras import backend as K
from keras import mixed_precision

# Enable mixed precision for memory efficiency
mixed_precision.set_global_policy('mixed_float16')

# Enable GPU memory growth to avoid pre-allocating all GPU memory at once
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Learning rate schedule
def lr_schedule(epoch, lr):
    if epoch < 20:
        return 1e-3
    elif 20 <= epoch < 25:
        return 5e-4
    else:
        return 1e-4

# Create sequences from the data
def create_sequences(data, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(data):
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix:out_end_ix, -1]  # Assuming the last column is the target
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run forecast with a specific config file.')
parser.add_argument('--config', type=str, default='config.yml', help='Path to the config file.')
args = parser.parse_args()

# Load the specified configuration file
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# Create output directory based on the output_prefix
output_dir = config['output_prefix']
os.makedirs(output_dir, exist_ok=True)

# Prepare directory for saving evaluation metrics
evaluation_folder = os.path.join(output_dir, config.get('evaluation_folder', 'evaluation_metrics'))
os.makedirs(evaluation_folder, exist_ok=True)

# Helper function to load and process datasets
def load_and_process_data(file_paths, for_transfer_learning=False):
    data_list = []
    for path in file_paths:
        if os.path.isdir(path):
            csv_files = glob.glob(os.path.join(path, "*.csv"))
        else:
            csv_files = [path]

        for csv_file in csv_files:
            dataset = pd.read_csv(csv_file, header=0, index_col=0)
            dataset.index = pd.to_datetime(dataset.index)
            dataset.replace('Min', np.nan, inplace=True)
            dataset = dataset.asfreq('D')
            dataset = dataset.apply(pd.to_numeric, errors='coerce')
            dataset = dataset.loc[dataset['streamflow'].first_valid_index():]
            dataset = dataset.loc[:dataset['streamflow'].last_valid_index()]
            dataset = dataset.interpolate(method='linear').ffill().bfill()
            if for_transfer_learning:
                data_list.append(dataset)
            else:
                data_list.append((csv_file, dataset))
    return data_list

# Load main datasets
print("Loading main datasets...")
main_datasets = load_and_process_data(config['data_paths'])

# Load transfer learning datasets (if applicable)
if 'transfer_learning_paths' in config:
    print("Loading transfer learning datasets...")
    transfer_datasets = load_and_process_data(config['transfer_learning_paths'], for_transfer_learning=True)
else:
    transfer_datasets = []

# Identify all feature names (excluding 'streamflow') across all main datasets
print("Identifying all feature names...")
all_features = set()
for _, dataset in main_datasets:
    all_features.update([col for col in dataset.columns if col != 'streamflow'])
all_features = sorted(all_features)  # Ensure consistent ordering
number_of_features = len(all_features) + 1  # +1 for 'streamflow' column

# Compute global min and max for each column across all main and transfer learning datasets
print("Computing global min and max for each column...")
min_max_dict = {}
# Include both main and transfer datasets for scaling
for dataset in main_datasets:
    _, ds = dataset
    for col in ds.columns:
        col_min = ds[col].min()
        col_max = ds[col].max()
        if col not in min_max_dict:
            min_max_dict[col] = {'min': col_min, 'max': col_max}
        else:
            if col_min < min_max_dict[col]['min']:
                min_max_dict[col]['min'] = col_min
            if col_max > min_max_dict[col]['max']:
                min_max_dict[col]['max'] = col_max

for ds in transfer_datasets:
    for col in ds.columns:
        col_min = ds[col].min()
        col_max = ds[col].max()
        if col not in min_max_dict:
            min_max_dict[col] = {'min': col_min, 'max': col_max}
        else:
            if col_min < min_max_dict[col]['min']:
                min_max_dict[col]['min'] = col_min
            if col_max > min_max_dict[col]['max']:
                min_max_dict[col]['max'] = col_max

# Convert min_max_dict to a DataFrame and save as CSV
min_max_df = pd.DataFrame(min_max_dict).T  # Transpose for easier CSV format
min_max_csv_path = os.path.join(output_dir, 'min_max.csv')
min_max_df.to_csv(min_max_csv_path)
print(f"Global min and max saved to {min_max_csv_path}")

# Function to scale data using global min and max
def scale_data(df, min_max_dict):
    scaled_df = pd.DataFrame(index=df.index)
    for col in df.columns:
        min_val = min_max_dict[col]['min']
        max_val = min_max_dict[col]['max']
        if max_val > min_val:
            scaled_df[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            scaled_df[col] = 0.0  # Handle case where min == max to avoid division by zero
    return scaled_df

# Data generator for tf.data.Dataset
def data_generator(datasets, n_steps_in, n_steps_out, min_max_dict, features, split='train'):
    global test_data_dict  # Use a global dictionary to store the test data

    for dataset_info in datasets:
        if split == 'test' and dataset_info[0] is None:
            # Skip transfer learning datasets in test split
            continue

        csv_file, dataset = dataset_info
        station_name = os.path.splitext(os.path.basename(csv_file))[0] if csv_file else "transfer_learning"
        print(f"Processing station: {station_name} for split: {split}")

        # Ensure the dataset has all required features
        missing_features = set(features) - set(dataset.columns)
        if missing_features:
            print(f"Warning: Missing features {missing_features} in station {station_name}. Filling with NaN.")
            for mf in missing_features:
                dataset[mf] = np.nan

        # Reorder the dataset columns to match all_features + 'streamflow'
        dataset = dataset[features + ['streamflow']]

        # Scale all features using global min and max
        scaled_features = scale_data(dataset[features], min_max_dict)
        # Scale the target variable (streamflow) using global min and max
        scaled_target = scale_data(dataset[['streamflow']], min_max_dict)

        # Combine scaled features and target
        scaled = np.hstack((scaled_features.values, scaled_target.values))  # 'streamflow' is the last column

        # Create sequences for LSTM
        X, y = create_sequences(scaled, n_steps_in, n_steps_out)

        # Determine split indices
        total_sequences = len(X)
        train_end = int(total_sequences * 0.8)
        val_end = int(total_sequences * 0.9)

        if split == 'train':
            sequences_X = X[:train_end]
            sequences_y = y[:train_end]
        elif split == 'val':
            sequences_X = X[train_end:val_end]
            sequences_y = y[train_end:val_end]
        elif split == 'test':
            sequences_X = X[val_end:]
            sequences_y = y[val_end:]
            # Store the test data and start date for this station
            if len(sequences_X) > 0:
                # Determine the start date based on the dataset's index
                # The first test sequence starts at index: train_end + n_steps_in
                test_start_idx = train_end + n_steps_in
                if test_start_idx >= len(dataset):
                    test_start_idx = len(dataset) - n_steps_in - n_steps_out
                test_start_date = dataset.index[test_start_idx]
                test_data_dict[station_name] = {
                    'X': sequences_X,
                    'y': sequences_y,
                    'start_date': test_start_date
                }
            continue  # Skip yielding for test split
        else:
            raise ValueError("split must be one of 'train', 'val', or 'test'")

        # Yield data for the specified split
        for i in range(len(sequences_X)):
            yield sequences_X[i].astype(np.float32), sequences_y[i].astype(np.float32)  # Ensure float32 dtype for TensorFlow

# Initialize the dictionary to store test data
test_data_dict = {}

# Create separate generators for training and validation
train_generator = lambda: data_generator(
    main_datasets + [(None, ds) for ds in transfer_datasets],  # Include transfer datasets in train split
    config['n_steps_in'],
    config['n_steps_out'],
    min_max_dict,
    all_features,
    split='train'
)

val_generator = lambda: data_generator(
    main_datasets + [(None, ds) for ds in transfer_datasets],  # Include transfer datasets in val split
    config['n_steps_in'],
    config['n_steps_out'],
    min_max_dict,
    all_features,
    split='val'
)

# Create dataset from generator for training
train_dataset = tf.data.Dataset.from_generator(
    train_generator,
    output_signature=(
        tf.TensorSpec(shape=(config['n_steps_in'], number_of_features), dtype=tf.float32),
        tf.TensorSpec(shape=(config['n_steps_out'],), dtype=tf.float32)
    )
)

# Create dataset from generator for validation
val_dataset = tf.data.Dataset.from_generator(
    val_generator,
    output_signature=(
        tf.TensorSpec(shape=(config['n_steps_in'], number_of_features), dtype=tf.float32),
        tf.TensorSpec(shape=(config['n_steps_out'],), dtype=tf.float32)
    )
)

# Shuffle, batch, and prefetch the training dataset
train_dataset = train_dataset.shuffle(buffer_size=10000) \
                             .batch(config['batch_size']) \
                             .prefetch(tf.data.AUTOTUNE)

# Batch and prefetch the validation dataset
val_dataset = val_dataset.batch(config['batch_size']) \
                         .prefetch(tf.data.AUTOTUNE)

# Additionally, generate test data by running the generator with split='test'
# This ensures that test_data_dict is populated
_ = list(data_generator(
    main_datasets,  # Only main datasets are used for test split
    config['n_steps_in'],
    config['n_steps_out'],
    min_max_dict,
    all_features,
    split='test'
))

# Build the LSTM/GRU model with Dropout layers
output_prefix = config['output_prefix']  # e.g., "run_1"
log_dir = os.path.join(
    "logs",
    "fit",
    f"{output_prefix}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
)

# Callbacks
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(lr_schedule)

# Build the LSTM/GRU model
model = Sequential()
model.add(Input(shape=(config['n_steps_in'], number_of_features)))  # Updated input shape

if config['model_type'].upper() == 'GRU':
    model.add(GRU(256, activation='tanh'))  # GRU layer
else:
    model.add(LSTM(256, activation='tanh'))  # LSTM layer

model.add(Dropout(0.4))  # Add dropout after the recurrent layer
model.add(Dense(config['n_steps_out'], dtype='float32'))  # Final dense layer

# Compile the model
opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], clipvalue=1.0)
model.compile(optimizer=opt, loss=MeanSquaredError(), metrics=['accuracy'])
model.summary()

# Train the model using the separate training and validation datasets
history = model.fit(
    train_dataset,
    epochs=config['epochs'],
    validation_data=val_dataset,
    verbose=1,
    callbacks=[early_stopping, tensorboard_callback, lr_scheduler]
)

# Save the model if required
if config.get('export_model', False):
    model_path = os.path.join(output_dir, "combined_model.keras")
    model.save(model_path)
    print(f"Model saved at {model_path}")

# Clear GPU memory
gc.collect()
K.clear_session()

# ========================= EVALUATION ================================
for station_name, test_info in test_data_dict.items():
    print(f"Testing on station: {station_name}")

    test_X = test_info['X']
    test_y = test_info['y']
    test_start_date = test_info['start_date']

    if len(test_X) == 0:
        print(f"No test data available for station: {station_name}")
        continue

    # Create a tf.data.Dataset for testing
    test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    test_dataset = test_dataset.batch(config['batch_size']).prefetch(tf.data.AUTOTUNE)

    # Predict on the test dataset
    y_pred = model.predict(test_dataset)

    # Inverse transform the predictions and actual values using global min and max
    # Assuming 'streamflow' is the last column in min_max_dict
    streamflow_min = min_max_dict['streamflow']['min']
    streamflow_max = min_max_dict['streamflow']['max']
    y_pred_inv = y_pred * (streamflow_max - streamflow_min) + streamflow_min
    test_y_inv = test_y * (streamflow_max - streamflow_min) + streamflow_min

    # Ensure that the lengths of y_pred_inv and test_y_inv match before constructing the DataFrame
    min_length = min(len(y_pred_inv), len(test_y_inv))
    y_pred_inv = y_pred_inv[:min_length]  # Truncate to the minimum length
    test_y_inv = test_y_inv[:min_length]  # Truncate to the minimum length

    # Create the date range for the test data
    # Each test sequence predicts n_steps_out days ahead
    # Therefore, for each sequence, we need to create n_steps_out dates starting from test_start_date
    forecast_horizons = np.arange(1, config['n_steps_out'] + 1)  # e.g., [1, 2, 3] for n_steps_out=3
    test_dates = pd.date_range(start=test_start_date, periods=min_length)

    # Repeat each date for each forecast horizon and shift by the horizon
    repeated_dates = np.repeat(test_dates, config['n_steps_out'])
    shifted_dates = repeated_dates + pd.to_timedelta(np.tile(forecast_horizons, min_length), unit='D')

    # Flatten the observed and forecasted values
    forecasted_values = y_pred_inv.flatten()
    observed_values = test_y_inv.flatten()

    # Construct DataFrame with correctly aligned dates
    df_result = pd.DataFrame({
        'Date': shifted_dates,
        'Forecast_Horizon': np.tile(forecast_horizons, min_length),
        'Observed': observed_values,
        'Forecasted': forecasted_values
    })

    # Save the forecast vs observed results to a CSV file
    result_csv_path = os.path.join(output_dir, f"{station_name}_forecast_vs_observed.csv")
    df_result.to_csv(result_csv_path, index=False)
    print(f"Results saved for {station_name} at {result_csv_path}")

    # ======================= METRICS CALCULATION ===========================
    print(f"Calculating metrics for station: {station_name}")

    # Convert to xarray DataArrays for metric calculations
    y_pred_da = xr.DataArray(y_pred_inv)
    test_y_da = xr.DataArray(test_y_inv)

    # Define the metrics list, excluding Peak-Timing and Missed-Peaks
    metrics_list = get_available_metrics()
    metrics_list = [m for m in metrics_list if m not in ["Peak-Timing", "Missed-Peaks"]]

    # Dictionary to store metrics per forecast day
    metrics_per_day = {}

    # Calculate metrics for each forecast day
    for day in range(config['n_steps_out']):
        # Extract the observations and simulations for the current forecast horizon
        obs = test_y_da[:, day] if test_y_da.ndim > 1 else test_y_da
        sim = y_pred_da[:, day] if y_pred_da.ndim > 1 else y_pred_da

        # Calculate metrics
        day_metrics = calculate_metrics(obs, sim, metrics=metrics_list)
        metrics_per_day[day + 1] = day_metrics  # Store metrics for each forecast day

    # Convert metrics_per_day into a DataFrame and save as CSV
    metrics_df = pd.DataFrame(metrics_per_day).T  # Transpose for better readability
    metrics_csv_path = os.path.join(evaluation_folder, f"{station_name}_daywise_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=True)
    print(f"Metrics saved for {station_name} at {metrics_csv_path}")
