#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 19:24:18 2020

@author: joshuageiser
"""

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def get_data(data_dir, region, city):
    
    # second pass file (might need to be cleaned a little in the future)
    filepath = os.path.join(data_dir, f'{region}_{city}_third_pass.csv')
    df = pd.read_csv(filepath)
    
    # Constants corresponding to start/end rows of CSV (assuming not LA datafile)
    start_index = 168 # First line where all features have data
    end_index = 26280 # End at last day/hour of 2019
    
    # Get list of columns that will be used for X dataset
    x_cols = list(df.columns)
    x_cols.remove('date_hour')
    x_cols.remove('load')
    
    # Get X dataset as numpy array
    X = df[x_cols][start_index:end_index]
    X = np.array(X)
    
    # Get y dataset as numpy array
    y_cols = ['load']
    y = df[y_cols][start_index:end_index]
    y = np.array(y)
    
    return X, y

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
    mdl = linear_model.LinearRegression()
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)
    
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
    