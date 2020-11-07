"""
Recurrent neural network model for electricity load
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


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
    model.add(keras.layers.LSTM(units, input_shape=(window, num_features)))
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
    
    # Select data columns for the RNN
    input_keep_cols = ['hour', 'weekday', 'weekend', 'pre_weekend', 'post_weekend', 'holiday', 'dwpc', 'relh', 'sped', 'tmpc']
    y = df['load'].to_numpy()
    y = np.expand_dims(y, axis=1)
    X = df[input_keep_cols].to_numpy()

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    # Train model and predict
    X_train_tensor = create_window_data(X_train, window = 6)
    y_train_tensor = create_window_data(y_train, window = 6, keep_whole_window=False)
    
    model = define_vanilla_lstm(window = 6, num_features = X_train.shape[1], units=20)
    model.fit(X_train_tensor, y_train_tensor, epochs=30)

    X_test_tensor = create_window_data(X_test, window = 6)
    y_test_tensor = create_window_data(y_test, window = 6, keep_whole_window=False)
    y_pred = model.predict(X_test_tensor)

    # Scatter plot of predictions vs true values
    plt.figure()
    plt.scatter(y_test_tensor, y_pred)
    plt.xlabel('True Load')
    plt.ylabel('Predicted Load')
    plt.show()
    
    # Error statistics
    MSE = mean_squared_error(y_test_tensor, y_pred)
    R2_score = r2_score(y_test_tensor, y_pred)
    
    print(f'Mean Squared Error: {MSE:.2f}')
    print(f'R2 Score: {R2_score:.3f}')

    return


if __name__ == "__main__":
    main()
