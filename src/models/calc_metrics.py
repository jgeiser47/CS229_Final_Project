#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 15:54:04 2020

@author: joshuageiser
"""

import os
import pandas as pd
import numpy as np
import datetime
import json
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def calc_metrics(y_true, y_pred, dates_arr, hyperparams, save_outputs=True):
    '''
    Calculates some metrics and plots for current run of an LSTM and saves them
    to a folder under the models subdirectory

    Parameters
    ----------
    y_true : numpy array of shape (m,1) with true load values
        DESCRIPTION.
    y_pred : numpy array of shape (m,1) with predicted load values
        DESCRIPTION.
    dates_arr : numpy array of shape (m,1) with date_hour strings
        DESCRIPTION.
    hyperparams : dictionary containing hyperparameter specifications
        DESCRIPTION.
    save_outputs : bool, optional
        DESCRIPTION. If True, will save outputs to a generated directory under
                     the root/models subdirectory

    Returns
    -------
    None.

    '''
    
    if save_outputs:
        # Path to data directory contaning CSVs 
        output_dir = os.path.join(os.getcwd(), '..', '..', 'models')
        output_dir = os.path.abspath(output_dir)
        
        # Current date/time
        run_datetime = datetime.datetime.now().strftime('%m%d%y_%H%M%S')
        new_dir_name = f'run_{run_datetime}'
        output_dir = os.path.join(output_dir, new_dir_name)
        os.mkdir(output_dir)
    
    # Error statistics #######################################################
    MSE = mean_squared_error(y_true, y_pred)
    R2_score = r2_score(y_true, y_pred)
    
    print(f'Mean Squared Error: {MSE:.2f}')
    print(f'R2 Score: {R2_score:.3f}')
    
    if save_outputs:
        filename = 'error_stats.txt'
        with open(os.path.join(output_dir,filename), 'w') as f:
            f.write(f'Mean Squared Error: {MSE:.2f}\n')
            f.write(f'R2 Score: {R2_score:.3f}')
    
    # Scatter plot of Predicted vs True Load #################################
    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel('True Load (MW)')
    plt.ylabel('Predicted Load (MW)')
    plt.title('Predicted vs True Load')
    
    if save_outputs:
        filename = 'scatter.png'
        plt.savefig(os.path.join(output_dir,filename))
    
    # Timeseris Plot (automatically only plots first two weeks of data) ######
    dates_arr_plt = to_dt(dates_arr)
    
    plt.figure()
    
    start_ind = 0
    end_ind = 336 # First two weeks
    end_ind = min(end_ind, len(dates_arr_plt))
    
    plt.plot_date(dates_arr_plt[start_ind:end_ind], y_true[start_ind:end_ind], 'b:', label='True')
    plt.plot_date(dates_arr_plt[start_ind:end_ind], y_pred[start_ind:end_ind], 'g:', label='Predicted')
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Load (MW)')
    plt.title('2-Week Predicted vs True Load')
    plt.legend()

    if save_outputs:
        filename = 'timeseries.png'
        plt.savefig(os.path.join(output_dir,filename))    
        
    # Save current run's hyperparameters to JSON file ########################
    if save_outputs:
        filename = 'hyperparameters.json'
        with open(os.path.join(output_dir,filename), 'w') as f:
            json.dump(hyperparams, f, indent=4)
            
    # Save arrays of predicted and true load value to CSV ####################
    if save_outputs:
        dates_arr = np.reshape(dates_arr, (len(dates_arr),1))
        df_out = np.hstack([dates_arr, y_true, y_pred])
        
        header = ['date_hour', 'True Load', 'Predicted Load']
        filename = 'prediction_array.csv'
        output_filepath = os.path.join(output_dir, filename)
        pd.DataFrame(df_out).to_csv(output_filepath, header=header, index=None)
    
    return

def get_datetime(dt_str):
    
    dt_obj = datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
    
    return dt_obj

def to_dt(dt_arr):
    
    N = len(dt_arr)
    
    return_arr = [''] * N
    
    j = 0
    for i in range(N):
        return_arr[j] = get_datetime(dt_arr[i])
        j+=1
    
    return return_arr

def main():
    
    # Path to data directory contaning CSVs 
    data_dir = os.path.join(os.getcwd(), '..', '..', 'data', 'interim')
    data_dir = os.path.abspath(data_dir)
    
    # For now, just run on one city at a time
    region = 'ercot'
    city = 'houston'
    
    # Get design matrix and labels
    from base_model_sklearn import get_data
    X,y,X_test,y_test,dates_range_test = get_data(data_dir, region, city)
    
    # Split into train and test sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
    
    # Train model and predict 
    mdl = linear_model.LinearRegression()
    mdl.fit(X_train, y_train)
    
    ##########################################################################
    
    # Will need to update our LSTM architecture so that we input all of our
    # hyperparameters in a dictionary format
    hyperparams = {'window': 24,
                   'memory_layer_units': 10,
                   'num_hidden_layers': 2,
                   'training_epochs': 30
                   }
    
    # Need 3 array inputs: true load, predicted load, and array of date_hour 
    # strings for the range of prediction
    y_December_True = y_test
    y_December_Pred = mdl.predict(X_test)
    December_dates_arr = dates_range_test
    
    # Calculate metrics
    calc_metrics(y_December_True, y_December_Pred, December_dates_arr, hyperparams, save_outputs=True)
    
    return

if __name__ == "__main__":
    main()
    