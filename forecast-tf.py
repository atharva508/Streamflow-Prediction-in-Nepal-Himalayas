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

# Load transfer learning datasets (if applicable)
if 'transfer_learning_paths' in config:
    print("Loading transfer learning datasets...")
    transfer_datasets, _, _ = load_and_process_data_with_structure(config['transfer_learning_paths'], for_transfer_learning=True)
else:
    transfer_datasets = []

# Combine the main datasets and transfer learning datasets for training
combined_datasets = datasets + [(None, ds) for ds in transfer_datasets]  # Add transfer datasets to the list

# Initialize scalers
scaler_features = MinMaxScaler(feature_range=(0, 1))
scaler_target = MinMaxScaler(feature_range=(0, 1))

# Initialize the dictionary to store test data
test_data_dict = {}

# Data generator for tf.data.Dataset
def data_generator(datasets, n_steps_in, n_steps_out, scaler_features, scaler_target):
    global test_data_dict  # Use a global dictionary to store the test data
    
    for csv_file, dataset in datasets:
        station_name = os.path.splitext(os.path.basename(csv_file))[0] if csv_file else "transfer_learning"
        print(f"Processing station: {station_name}")
        
        features = dataset.columns.tolist()
        features.remove('streamflow')

        static_columns = [col for col in features if dataset[col].nunique() == 1]
        dynamic_columns = [col for col in features if col not in static_columns]

        # Scale dynamic columns
        scaled_dynamic_features = scaler_features.fit_transform(dataset[dynamic_columns])
        scaled_dynamic_features_df = pd.DataFrame(scaled_dynamic_features, columns=dynamic_columns, index=dataset.index)

        combined_features_df = pd.concat([scaled_dynamic_features_df, dataset[static_columns]], axis=1)

        # Scale the target variable (streamflow)
        scaled_target = scaler_target.fit_transform(dataset[['streamflow']])
        scaled = np.hstack((combined_features_df.values, scaled_target))

        # Create sequences for LSTM
        X, y = create_sequences(scaled, n_steps_in, n_steps_out)

        # Split data into training (80%) and test (20%) sets
        train_size = int(len(X) * 0.8)
        train_X, train_y = X[:train_size], y[:train_size]
        test_X, test_y = X[train_size:], y[train_size:]

        # Store the test data for this station
        test_data_dict[station_name] = (test_X, test_y)

        # Yield training data for each sequence
        for i in range(len(train_X)):
            yield train_X[i], train_y[i]  # Yield data for training

# Create dataset from generator with combined datasets
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(combined_datasets, config['n_steps_in'], config['n_steps_out'], scaler_features, scaler_target),
    output_signature=(
        tf.TensorSpec(shape=(config['n_steps_in'], len(dynamic_columns) + len(static_columns) + 1), dtype=tf.float32),  # +1 for the streamflow column
        tf.TensorSpec(shape=(config['n_steps_out'],), dtype=tf.float32)
    )
)

# Shuffle, batch, repeat, and prefetch the dataset
train_dataset = train_dataset.shuffle(buffer_size=10000) \
                             .batch(config['batch_size']) \
                             .prefetch(tf.data.AUTOTUNE)  # Remove .repeat()

# Build the LSTM model with Dropout layers
output_prefix = config['output_prefix']  # "run_1" in your example
log_dir = os.path.join("logs", "fit", f"{output_prefix}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")

# Callbacks
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(lr_schedule)

# Build the LSTM model
model = Sequential()
model.add(Input(shape=(config['n_steps_in'], len(dynamic_columns) + len(static_columns) + 1)))  # +1 for the streamflow column

if config['rnn_type'] == 'GRU':
    model.add(GRU(256, activation='tanh'))  # Reduced GRU units to 256
else:
    model.add(LSTM(256, activation='tanh'))  # LSTM layer (default)

model.add(Dropout(0.4))  # Add dropout after the LSTM layer
model.add(Dense(config['n_steps_out'], dtype='float32'))  # Final layer

# Compile the model
opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], clipvalue=1.0)
model.compile(optimizer=opt, loss=MeanSquaredError(), metrics=['accuracy'])
model.summary()

# Manually split the dataset to avoid validation split issues with `tf.data.Dataset`
val_dataset = train_dataset.take(100)  # Take a subset for validation
train_dataset = train_dataset.skip(100)  # Use the remaining data for training

# Train the model without steps_per_epoch
history = model.fit(train_dataset,
                    epochs=config['epochs'],
                    validation_data=val_dataset,
                    verbose=1,
                    callbacks=[early_stopping, tensorboard_callback, lr_scheduler])

# Save the model if required
if config['export_model']:
    model.save(os.path.join(output_dir, f"combined_model.h5"))

# Clear GPU memory between epochs
gc.collect()
K.clear_session()


# Evaluate the model using the stored test data
for station_name, (test_X, test_y) in test_data_dict.items():
    print(f"Testing on station: {station_name}")

    # Test data as tf.data.Dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    test_dataset = test_dataset.batch(config['batch_size']).prefetch(tf.data.AUTOTUNE)
    
    # Predict on the test dataset
    y_pred = model.predict(test_dataset)

    # Inverse transform the predictions and actual values
    y_pred_inv = scaler_target.inverse_transform(y_pred)
    test_y_inv = scaler_target.inverse_transform(test_y)

    # Ensure that the lengths of y_pred_inv and test_y_inv match before constructing the DataFrame
    min_length = min(len(y_pred_inv), len(test_y_inv))
    y_pred_inv = y_pred_inv[:min_length]  # Truncate to the minimum length
    test_y_inv = test_y_inv[:min_length]  # Truncate to the minimum length

    # Get the first date from the dataset's index to create the date range
    start_date = datasets[0][1].index[-min_length]  # This will be the first date of the test set
    test_dates = pd.date_range(start=start_date, periods=min_length)

    # Store the forecasted values as lists
    forecasted_values = [list(pred) for pred in y_pred_inv]

    # Store observed values (only the first day of the test set)
    observed_values = [obs[0] for obs in test_y_inv]  # Since observed is only for day 1

    # Construct DataFrame with forecasted values stored as lists (or JSON)
    df_result = pd.DataFrame({
        'Date': test_dates,
        'Observed': observed_values,
        'Forecasted': forecasted_values  # Store forecasted values as lists
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
        obs = test_y_da[:, day]
        sim = y_pred_da[:, day]
        day_metrics = calculate_metrics(obs, sim, metrics=metrics_list)
        metrics_per_day[day + 1] = day_metrics  # Store metrics for each forecast day

    # Convert metrics_per_day into a DataFrame and save as CSV
    metrics_df = pd.DataFrame(metrics_per_day).T  # Transpose for better readability
    metrics_csv_path = os.path.join(evaluation_folder, f"{station_name}_daywise_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=True)
    print(f"Metrics saved for {station_name} at {metrics_csv_path}")

