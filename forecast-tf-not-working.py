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

# Sequence creation function
def create_sequences(features, target, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(features)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(features):
            break
        # Combine features with the corresponding target values
        seq_x = np.hstack((features[i:end_ix], target[i:end_ix].reshape(-1, 1)))
        seq_y = target[end_ix:out_end_ix]
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

# Helper function to load and process datasets and extract column structure
def load_and_process_data_with_structure(file_paths, for_transfer_learning=False):
    data_list = []
    dynamic_columns, static_columns = None, None

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

            # Extract dynamic and static columns (do this for the first dataset only)
            if dynamic_columns is None and static_columns is None:
                features = dataset.columns.tolist()
                features.remove('streamflow')
                static_columns = [col for col in features if dataset[col].nunique() == 1]
                dynamic_columns = [col for col in features if col not in static_columns]

    return data_list, dynamic_columns, static_columns

# Load datasets and extract column structure
print("Loading datasets...")
datasets, dynamic_columns, static_columns = load_and_process_data_with_structure(config['data_paths'])

# Define all_columns as the combination of dynamic and static columns
all_columns = dynamic_columns + static_columns

# Load transfer learning datasets (if applicable)
if 'transfer_learning_paths' in config:
    print("Loading transfer learning datasets...")
    transfer_datasets, _, _ = load_and_process_data_with_structure(config['transfer_learning_paths'], for_transfer_learning=True)
    # Convert transfer_datasets to tuples with None as csv_file for consistency
    transfer_datasets = [(None, ds) for ds in transfer_datasets]
else:
    transfer_datasets = []

# Combine the main datasets and transfer learning datasets for training
combined_datasets = datasets + transfer_datasets  # Add transfer datasets to the list

# Initialize scaler_dict for features and target
scaler_dict = {}
for col in all_columns:
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Fit the scaler on the combined datasets
    combined_feature_values = np.concatenate([
        ds[1][col].values.reshape(-1, 1) for ds in combined_datasets if col in ds[1].columns
    ], axis=0)
    scaler.fit(combined_feature_values)
    scaler_dict[col] = scaler

# Initialize scaler for the target variable
scaler_dict['streamflow'] = MinMaxScaler(feature_range=(0, 1))
combined_target_values = np.concatenate([
    ds[1]['streamflow'].values.reshape(-1, 1) for ds in combined_datasets if 'streamflow' in ds[1].columns
], axis=0)
scaler_dict['streamflow'].fit(combined_target_values)

# Define data generator for training and validation
def data_generator(datasets, n_steps_in, n_steps_out, scaler_dict, mode="train"):
    for csv_file, dataset in datasets:
        station_name = os.path.splitext(os.path.basename(csv_file))[0] if csv_file else "transfer_learning"
        print(f"Processing station: {station_name}")

        # Ensure dataset has all required feature columns
        for col in all_columns:
            if col not in dataset.columns:
                dataset[col] = 0  # or np.nan or some default value

        # Check if 'streamflow' exists in the dataset
        if 'streamflow' not in dataset.columns:
            print(f"Error: 'streamflow' column missing in dataset {station_name}")
            continue

        # Ensure columns are in the same order as all_columns
        dataset = dataset[['streamflow'] + all_columns]

        # Scale features
        scaled_features = pd.DataFrame({
            col: scaler_dict[col].transform(dataset[[col]].values).flatten() for col in all_columns
        }, index=dataset.index)

        # Scale target variable 'streamflow'
        scaled_target = scaler_dict['streamflow'].transform(dataset[['streamflow']].values).flatten()

        # Create sequences
        X, y = create_sequences(scaled_features.values, scaled_target, n_steps_in, n_steps_out)

        # Split data into training (70%), validation (10%), and test (20%) sets
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.1)
        test_size = len(X) - train_size - val_size

        train_X, train_y = X[:train_size], y[:train_size]
        val_X, val_y = X[train_size:train_size + val_size], y[train_size:train_size + val_size]

        if mode == "train":
            for i in range(len(train_X)):
                yield train_X[i], train_y[i]

        elif mode == "validation":
            for i in range(len(val_X)):
                yield val_X[i], val_y[i]

# Create TensorFlow datasets for training and validation
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(combined_datasets, config['n_steps_in'], config['n_steps_out'], scaler_dict, mode="train"),
    output_signature=(
        tf.TensorSpec(shape=(config['n_steps_in'], len(all_columns) + 1), dtype=tf.float32),  # +1 for the streamflow column
        tf.TensorSpec(shape=(config['n_steps_out'],), dtype=tf.float32)
    )
)

val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(combined_datasets, config['n_steps_in'], config['n_steps_out'], scaler_dict, mode="validation"),
    output_signature=(
        tf.TensorSpec(shape=(config['n_steps_in'], len(all_columns) + 1), dtype=tf.float32),  # +1 for the streamflow column
        tf.TensorSpec(shape=(config['n_steps_out'],), dtype=tf.float32)
    )
)

# Prepare TensorFlow datasets
train_dataset = train_dataset.shuffle(buffer_size=10000) \
                             .batch(config['batch_size']) \
                             .prefetch(tf.data.AUTOTUNE)

val_dataset = val_dataset.batch(config['batch_size']) \
                         .prefetch(tf.data.AUTOTUNE)  # No shuffle for validation

# Build the LSTM/GRU model with Dropout layers
output_prefix = config['output_prefix']  # e.g., "run_1"
log_dir = os.path.join("logs", "fit", f"{output_prefix}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")

# Callbacks
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(lr_schedule)

# Build the model
model = Sequential()
model.add(Input(shape=(config['n_steps_in'], len(all_columns) + 1)))  # +1 for the streamflow column

if config['model_type'] == 'GRU':
    model.add(GRU(256, activation='tanh'))
else:
    model.add(LSTM(256, activation='tanh'))

model.add(Dropout(0.4))  # Add dropout after the LSTM/GRU layer
model.add(Dense(config['n_steps_out'], dtype='float32'))  # Final layer

# Compile the model with appropriate metrics for regression
opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], clipvalue=1.0)
model.compile(optimizer=opt, loss=MeanSquaredError(), metrics=['mae', 'mse'])
model.summary()

# Train the model
history = model.fit(
    train_dataset,
    epochs=config['epochs'],
    validation_data=val_dataset,
    verbose=1,
    callbacks=[early_stopping, tensorboard_callback, lr_scheduler]
)

# Save the model if required
if config.get('export_model', False):
    model.save(os.path.join(output_dir, f"combined_model.keras"))
    print(f"Model saved at {os.path.join(output_dir, f'combined_model.keras')}")

# Clear GPU memory after training
gc.collect()
K.clear_session()

# Function to process and evaluate test data for a single station using a generator
def evaluate_station_test_data(station_name, dataset, scaler_dict, config, evaluation_folder, output_dir):
    print(f"Evaluating on station: {station_name}")

    # Ensure dataset has all required feature columns
    for col in all_columns:
        if col not in dataset.columns:
            dataset[col] = 0  # or np.nan or some default value

    # Check if 'streamflow' exists in the dataset
    if 'streamflow' not in dataset.columns:
        print(f"Error: 'streamflow' column missing in dataset {station_name}")
        return

    # Ensure columns are in the same order as all_columns
    dataset = dataset[['streamflow'] + all_columns]

    # Scale features
    scaled_features = pd.DataFrame({
        col: scaler_dict[col].transform(dataset[[col]].values).flatten() for col in all_columns
    }, index=dataset.index)

    # Scale target variable 'streamflow'
    scaled_target = scaler_dict['streamflow'].transform(dataset[['streamflow']].values).flatten()

    # Create sequences
    X, y = create_sequences(scaled_features.values, scaled_target, config['n_steps_in'], config['n_steps_out'])

    # Split data into test sets (assuming last 20% as test)
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.1)
    test_size = len(X) - train_size - val_size

    test_X, test_y = X[train_size + val_size:], y[train_size + val_size:]

    if len(test_X) == 0:
        print(f"No test data available for station: {station_name}")
        return

    # Define a generator for test data batches
    def test_data_generator(X, y, batch_size):
        for i in range(0, len(X), batch_size):
            yield X[i:i + batch_size], y[i:i + batch_size]

    # Create a tf.data.Dataset for testing using the generator
    test_dataset = tf.data.Dataset.from_generator(
        lambda: test_data_generator(test_X, test_y, config['batch_size']),
        output_signature=(
            tf.TensorSpec(shape=(None, config['n_steps_in'], len(all_columns) + 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, config['n_steps_out']), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    # Initialize lists to store metrics
    metrics_per_day = {day + 1: {"MAE": 0.0, "MSE": 0.0} for day in range(config['n_steps_out'])}
    counts_per_day = {day + 1: 0 for day in range(config['n_steps_out'])}

    # Initialize lists to store observed and forecasted values
    observed_values = []
    forecasted_values = []
    dates_aligned = []

    # Iterate over the test dataset batches
    for batch_X, batch_y in test_dataset:
        # Predict on the batch
        y_pred = model.predict(batch_X, verbose=0)

        # Inverse transform the predictions and actual values
        y_pred_inv = scaler_dict['streamflow'].inverse_transform(y_pred.reshape(-1, 1)).reshape(-1, config['n_steps_out'])
        test_y_inv = scaler_dict['streamflow'].inverse_transform(batch_y.numpy().reshape(-1, 1)).reshape(-1, config['n_steps_out'])

        # Shift the observed values (test_y_inv) by 1 day forward to match with the forecasted values
        if len(observed_values) == 0 and len(forecasted_values) == 0:
            # For the first batch, no shift is needed
            shifted_test_y_inv = test_y_inv
            y_pred_inv = y_pred_inv
        else:
            # Append to existing observed and forecasted values
            shifted_test_y_inv = np.vstack((observed_values[-1][1:], test_y_inv))
            y_pred_inv = np.vstack((forecasted_values[-1][1:], y_pred_inv))

        # Update metrics
        for day in range(config['n_steps_out']):
            obs = shifted_test_y_inv[:, day]
            pred = y_pred_inv[:, day]
            metrics = calculate_metrics(obs, pred, metrics=["MAE", "MSE"])
            metrics_per_day[day + 1]["MAE"] += metrics.get("MAE", 0.0) * len(obs)
            metrics_per_day[day + 1]["MSE"] += metrics.get("MSE", 0.0) * len(obs)
            counts_per_day[day + 1] += len(obs)

        # Store observed and forecasted values
        observed_values.extend(shifted_test_y_inv.tolist())
        forecasted_values.extend(y_pred_inv.tolist())

        # Store corresponding dates
        # Assuming that each sequence corresponds to the end date of the input sequence
        # Adjust this logic based on your actual date alignment
        # For simplicity, we'll skip date alignment here

    # Calculate average metrics
    for day in range(config['n_steps_out']):
        if counts_per_day[day + 1] > 0:
            metrics_per_day[day + 1]["MAE"] /= counts_per_day[day + 1]
            metrics_per_day[day + 1]["MSE"] /= counts_per_day[day + 1]

    # Convert metrics_per_day into a DataFrame and save as CSV
    metrics_df = pd.DataFrame(metrics_per_day).T  # Transpose for better readability
    metrics_df.index.name = 'Forecast_Day'

    metrics_csv_path = os.path.join(evaluation_folder, f"{station_name}_daywise_metrics.csv")
    metrics_df.to_csv(metrics_csv_path)
    print(f"Metrics saved for {station_name} at {metrics_csv_path}")

    # Convert the observed and forecasted values into lists
    # Since we processed in batches and shifted, ensure alignment
    # Here, we're assuming observed_values and forecasted_values are aligned

    # Construct DataFrame with forecasted values
    df_result = pd.DataFrame({
        'Observed': observed_values,
        'Forecasted': forecasted_values
    })

    # Save the forecast vs observed results to a CSV file
    result_csv_path = os.path.join(output_dir, f"{station_name}_forecast_vs_observed.csv")
    df_result.to_csv(result_csv_path, index=False)
    print(f"Results saved for {station_name} at {result_csv_path}")

    # Clear memory for this station
    del test_X, test_y, y_pred, y_pred_inv, test_y_inv, shifted_test_y_inv, df_result
    del test_dataset, metrics_df, metrics_per_day, observed_values, forecasted_values, dates_aligned
    gc.collect()

# Evaluation on Test Data
print("Starting evaluation on test data...")
for csv_file, dataset in datasets + transfer_datasets:
    station_name = os.path.splitext(os.path.basename(csv_file))[0] if csv_file else "transfer_learning"
    evaluate_station_test_data(station_name, dataset, scaler_dict, config, evaluation_folder, output_dir)

print("Evaluation completed for all stations.")

# Final cleanup
gc.collect()
K.clear_session()
