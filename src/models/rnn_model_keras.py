"""
Recurrent neural network model for electricity load
"""
import os
import numpy as np
import pandas as pd
import datetime
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def create_window_data(data, window, keep_whole_window=True):
    """Preprocess data to match RNN model specification
    
    Args:
        data: matrix of inputs for the RNN (num_steps, num_features)
        window: the width of the look-back window used for the RNN
        keep_whole_window: flag controlling whether the whole window
            will be preserved in each entry of the output, or only
            the current element at the end of the window.

    Returns: 3-dimensional tensor (np.ndarray) of dimensions
    (num_steps, window [or 1 if keep_whole_window is False], num_features) 
    ready for input to an LSTM.
    """
    num_steps = data.shape[0] - window
    num_features = data.shape[1]
    if keep_whole_window is False:
        keep_nelems = 1
    else:
        keep_nelems = window
    arr = np.zeros((num_steps, keep_nelems, num_features))
    for i in range(window, num_steps):
        cur_window = data[(i-keep_nelems):i,:]
        arr[i-window,:,:] = cur_window
    arr = arr.squeeze()
    return arr

def define_vanilla_lstm(window, num_features, units):
    """Defines a Vanilla LSTM model using Keras

    Args:
        window: width of the look-back window used for the model's
            input data
        num_features: number of features in the input data
        units: dimension of "memory" space

    Returns: keras Sequential model object
    """
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(units, return_sequences=True, input_shape=(window, num_features)))
    model.add(keras.layers.LSTM(units))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')

    return model

def main():
    # For now, just run on one city at a time
    region = 'ercot'
    city = 'houston'

    # Read appropriate CSV into a dataframe
    data_dir = os.path.join(os.getcwd(), 'data', 'interim')
    data_dir = os.path.abspath(data_dir)
    filepath = os.path.join(data_dir, f'{region}_{city}_third_pass.csv')
    df = pd.read_csv(filepath)
    
    # Create datetime index
    df.index = pd.to_datetime(df.date_hour)
    input_keep_cols = ['hour', 'weekday', 'weekend', 'pre_weekend', 'post_weekend', 'holiday', 'dwpc', 'relh', 'sped', 'tmpc', 'load']
    df_select = df.loc[:,input_keep_cols]

    # Scale the data
    float_cols = list(df_select.columns[df_select.dtypes==np.float64])
    float_scaler = StandardScaler(copy=False)
    float_scaler.fit(df_select.loc[:, float_cols])
    df_select.loc[:, float_cols] = float_scaler.transform(df_select.loc[:, float_cols])

    # Split train/test data
    df_train = df_select[:'2019-12-31']
    df_test = df_select['2020-01-01':'2020-03-01']

    # Select data columns for the RNN
    y_train = df_train['load'].to_numpy()
    y_train= np.expand_dims(y_train, axis=1)
    X_train = df_train.drop('load', axis=1).to_numpy()
    y_test = df_test['load'].to_numpy()
    y_test= np.expand_dims(y_test, axis=1)
    X_test = df_test.drop('load', axis=1).to_numpy()

    # Train model and predict
    window = 24
    memory_layer_units = 10
    training_epochs = 30
    X_train_tensor = create_window_data(X_train, window = window)
    y_train_tensor = create_window_data(y_train, window = window, keep_whole_window=False)
    
    model = define_vanilla_lstm(window = window, num_features = X_train.shape[1], units=memory_layer_units)
    model.fit(X_train_tensor, y_train_tensor, epochs=training_epochs)

    X_test_tensor = create_window_data(X_test, window = window)
    y_test_tensor = create_window_data(y_test, window = window, keep_whole_window=False)
    y_pred = model.predict(X_test_tensor)

    # Scatter plot of predictions vs true values
    plt.figure()
    plt.scatter(y_test_tensor, y_pred)
    plt.xlabel('True Load')
    plt.ylabel('Predicted Load')
    plt.show()
    
    # Error statistics
    MSE = mean_squared_error(y_test_tensor, y_pred)
    print(f'Mean Squared Error: {MSE:.2f}')

    return


if __name__ == "__main__":
    main()
