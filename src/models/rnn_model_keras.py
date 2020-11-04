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
