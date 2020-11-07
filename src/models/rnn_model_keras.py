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

from base_model_sklearn import get_data

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

def main():
    # Path to data directory contaning CSVs 
    data_dir = os.path.join(os.getcwd(), 'data', 'interim')
    data_dir = os.path.abspath(data_dir)
    
    # For now, just run on one city at a time
    region = 'ercot'
    city = 'houston'
    
    # Get design matrix and labels
    X,y = get_data(data_dir, region, city)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    # Train model and predict
    X_train_tensor = create_window_data(X_train, window = 6)
    y_train_tensor = create_window_data(y_train, window = 6, keep_whole_window=False)
    #TODO: Select columns for use in the NN
    
    # Scatter plot of predictions vs true values
    plt.figure()
    plt.scatter(y_test, y_pred)
    plt.xlabel('True Load')
    plt.ylabel('Predicted Load')
    plt.show()
    
    # Error statistics
    MSE = mean_squared_error(y_test, y_pred)
    R2_score = r2_score(y_test, y_pred)
    
    print(f'Mean Squared Error: {MSE:.2f}')
    print(f'R2 Score: {R2_score:.3f}')

    return


if __name__ == "__main__":
    main()
