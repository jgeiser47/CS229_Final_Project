"""
Recurrent neural network model for electricity load
"""
import os
import numpy as np
import pandas as pd
import datetime
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import json
from sklearn.model_selection import TimeSeriesSplit
from sklearn import metrics


class LSTM_Model:

    def __init__(self, df=None, window=24, layers=1, hidden_inputs=50, last_layer="Dense", scaler="Standard", epochs=30,
                 activation="tanh", eval_splits=5, preserve_weights=False):
        self.window = window
        self.layers = layers
        self.hidden_inputs = hidden_inputs
        self.last_layer = last_layer
        self.scaler = scaler
        self.df = df
        self.train_df = None
        self.compare_df = None
        self.test_df = None
        self.scaler_dict = {}
        self.epochs = epochs
        self.activation = activation
        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.X_compare = None
        self.model = None
        self.eval_splits = eval_splits
        self.preserve_weights = preserve_weights
        self.train_errors = []
        self.val_errors = []
        self.test_errors = []
        self.outputpath = None
        self.train_samples = []

    def scale_columns(self, col_name_list):
        """Scale columns named in col_name_list to mean zero, sd one,
        and return the individual scaler objects in a dictionary.
        Args:
            col_name_list: list of string column names denoting columns
                to be scaled
        Returns: Tuple of two objects. The first is the scaled dataframe.
        The second is a dictionary of scaler objects, the keys are the column names
        of the column scaled by each scaler.
        """
        scaler_dict = {}
        for cur_name in col_name_list:
            if self.scaler == "Standard":
                cur_scaler = StandardScaler()
            else:
                cur_scaler = MinMaxScaler()
            cur_data = self.df.loc[:, cur_name].values
            cur_data = np.expand_dims(cur_data, axis=1)
            cur_scaler.fit(cur_data)
            self.df.loc[:, cur_name] = cur_scaler.transform(cur_data).squeeze()
            scaler_dict[cur_name] = cur_scaler
        return scaler_dict

    def train_scaler(self, cols_to_scale):
        """Train scalers to scale down the columns of the dataframe individually
        for each city.

        Args:
            cols_to_scale: names of columns to scale

        Returns: None, but adds the scaler information to the self.scaler_dict
        """
        for city in self.df.index.get_level_values('city').unique():
            self.scaler_dict[city] = {}
            for col_name in cols_to_scale:
                if self.scaler == "Standard":
                    col_scaler = StandardScaler()
                else:
                    col_scaler = MinMaxScaler()
                col_data = self.df.loc[city, col_name].to_numpy()
                col_data = np.expand_dims(col_data, axis=1)
                col_scaler.fit(col_data)
                self.scaler_dict[city][col_name] = col_scaler

    def apply_scaler(self, df):
        """Apply scaler stored in self.scaler_dict to a DataFrame

        Args:
            df: the dataframe to apply self.scaler_dict to.
        Returns: a scaled version of df
        """
        for city in df.index.get_level_values('city').unique():
            cur_dict = self.scaler_dict[city]
            for col_name, col_scaler in cur_dict.items():
                cur_data = df.loc[(city, slice(None)), col_name].to_numpy()
                cur_data = np.expand_dims(cur_data, axis=1)
                transformed = col_scaler.transform(cur_data)
                df.loc[(city, slice(None)), col_name] = transformed
        return df


    def create_window_data(self, data, keep_whole_window=True):
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
        num_steps = data.shape[0] - self.window + 1
        num_features = data.shape[1]
        if keep_whole_window is False:
            keep_nelems = 1
        else:
            keep_nelems = self.window
        arr = np.zeros((num_steps, keep_nelems, num_features))
        for i in range(self.window, data.shape[0]+1):
            cur_window = data[(i - keep_nelems):i, :]
            arr[i - self.window, :, :] = cur_window
        arr = arr.squeeze()
        return arr

    def define_fit_vanilla_lstm(self):
        """Defines a Vanilla LSTM model using Keras
        Args:
            window: width of the look-back window used for the model's
                input data
            num_features: number of features in the input data
            units: dimension of "memory" space
        Returns: keras Sequential model object
        """
        model = keras.models.Sequential()
        num_features = len(self.df.columns) - 1
        if self.layers == 1:
            # TO DO ADD ACTIVATION
            model.add(keras.layers.LSTM(self.hidden_inputs, activation=self.activation, return_sequences=False,
                                        input_shape=(self.window, num_features)))
        else:
            for i in range(self.layers):
                model.add(keras.layers.LSTM(self.hidden_inputs, activation=self.activation, return_sequences=True,
                                            input_shape=(self.window, num_features)))
            model.add(keras.layers.LSTM(self.hidden_inputs , activation=self.activation))
        if self.last_layer == "Dense":
            model.add(keras.layers.Dense(1))
        model.compile(optimizer='adam', loss='mse')
        self.model = model

    def add_csv_data(self, city, dir, keep_cols, data_pass_name = 'third'):
        """Read CSV data from the 'third pass' format into a multi-indexed dataframe.csv

        Args:
            city: city abbreviation used in the name of the csv file
            dir: string file path to directory holding data files
            keep_cols: list of column names to retain
            data_pass_name: name of number of data pass type to use (e.g. use 'third'
            to get one of the 'third_pass.csv' files.)

        Returns: None, but row appends the dataframe to the model's self.df object.
        """
        # Get the desired file's path
        file_list = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        file_list.remove('.gitkeep')
        file_name = None
        for f in file_list:
            f_parts = f.split('_')
            if f_parts[1] == city and f_parts[2] == data_pass_name:
                file_name = f
                break
        
        # Read the file and create indices
        assert file_name is not None
        df = pd.read_csv(os.path.join(dir, file_name))
        df.index = pd.MultiIndex.from_arrays([[city] * len(df), pd.to_datetime(df.loc[:,'date_hour'])], names=['city', 'time'])
        df = df.loc[:, keep_cols]
        self.df = pd.concat([self.df, df], axis = 0)
        self.df.sort_index(inplace=True)
        print(f'Added {city} to model dataframe')

    def split_data_regimes(self, test_start = '2020-01-01 00:00:00', compare_start = '2020-03-01 00:00:00'):
        """Split train/val, test, and compare data.
        
        Args:
            test_start: string in pandas datetime format giving starting time (to hour)
                of test dataset (e.g. '2020-01-01 00:00:00')
            compare_start: string in pandas datetime format giving starting time (to hour)
                of compare dataset (e.g. '2020-03-01 00:00:00')

        Returns: None, but updates LSTM_Model attributes self.train_df, self.compare_df,
            and self.test_df.
        """
        first_time = self.df.index.get_level_values('time').min()
        last_time = self.df.index.get_level_values('time').max()
        train_times = pd.date_range(start=first_time, end=test_start, freq='H', closed='left')
        test_times = pd.date_range(start=test_start, end=compare_start, freq='H', closed='left')
        compare_times = pd.date_range(start=compare_start, end = last_time, freq='H')
        self.train_df = self.df.loc[(slice(None), train_times), :]
        self.compare_df = self.df.loc[(slice(None), compare_times), :]
        self.test_df = self.df.loc[(slice(None), test_times), :]

    def prep_eval_data(self, df):
        """Prepare a predicted dataset for evaluation.

        Args:
            df: Dataframe of observed and predicted load data. Must have a MultiIndex like
                the one in self.df. Must have two data columns, load and load_pred.
        
        Returns: A tuple of the three elements to pass to calc_metrics: (dates_arr, y_true, y_pred)
        """
        # Remove all rows with NaN in load_pred
        df = df.dropna().copy()

        # Unscale load columns
        for city in df.index.get_level_values('city').unique():
            load_scaler = self.scaler_dict[city]['load']

            y_obs = df.loc[(city, slice(None)), 'load'].to_numpy()
            y_obs = np.expand_dims(y_obs, axis=1)
            y_obs = load_scaler.inverse_transform(y_obs)
            df.loc[(city, slice(None)), 'load'] = y_obs

            y_pred = df.loc[city, 'load_pred'].to_numpy()
            y_pred = np.expand_dims(y_pred, axis=1)
            y_pred = load_scaler.inverse_transform(y_pred)
            df.loc[(city, slice(None)), 'load_pred'] = y_pred
        return df

    def run_experiment(self, cities, path, input_keep_cols, test_on_split=False, folds = 6):
        # Read, combine, and scale data
        for city in sorted(cities):
            self.add_csv_data(city, path, input_keep_cols)
        float_cols = list(self.df.columns[self.df.dtypes == np.float64])
        self.train_scaler(float_cols)
        self.df = self.apply_scaler(self.df)

        # Split data into parts
        self.split_data_regimes()

        # Split train and test data
        if test_on_split:
            test_times = pd.date_range(start='2018-01-01', end='2018-02-01', freq='H')
            self.test_on_splits(self.train_df.loc[(slice(None), test_times), :], folds=folds)
        else:
            self.train_test_split()

    def fit_model(self, df):
        """Fit self.model to the passed data frame df, split by city to prevent problems
        with window crossovers.

        Args:
            df: the dataframe to fit the model with. It will probably be a row subset
                of self.df.
        Returns: None, but the self.model is updated via the fitting process
        """
        # Define model
        if self.model is None or self.preserve_weights is False:
            self.define_fit_vanilla_lstm()

        # Split data by city and create windows
        for city in df.index.get_level_values('city').unique():
            print(f'Fitting on {city} data')
            city_df = df.loc[city, :]

            X = city_df.drop("load", axis=1).to_numpy()
            X_tensor = self.create_window_data(X)

            y = city_df.loc[:,"load"].to_numpy()
            y = np.expand_dims(y, axis=1)
            y_tensor = self.create_window_data(y, keep_whole_window=False)

            self.model.fit(X_tensor, y_tensor, epochs=self.epochs)

    def predict_model(self, df):
        """Predict load at each time step by adding a new "buddy" column
            to the original dataframe which gives predicted instead of observed
            load.

        Args:
            df: dataframe, a subset of self.df, to predict on and add the
            prediction column to

        Returns: df with new predicted column
        """
        df_return = df.loc[:, 'load'].to_frame()
        df_return = df_return.assign(load_pred = np.NaN)
        for city in df.index.get_level_values('city').unique():
            print(f'Predicting on {city} data')
            city_df = df.loc[(city, slice(None)), :]

            X = city_df.drop('load', axis=1).to_numpy()
            X_tensor = self.create_window_data(X)

            y_pred = self.model.predict(X_tensor)

            # Assign values to the correct part of df_return
            window_start = city_df.index.get_level_values('time').min() + pd.Timedelta(f'{self.window - 1} hours')
            window_end = city_df.index.get_level_values('time').max()
            window_times = pd.date_range(start=window_start, end=window_end, freq='H')
            df_return.loc[(city, window_times), 'load_pred'] = y_pred
        return df_return

    def train_test_split(self):
        # Create TimeSeriesSplitter
        tscv = TimeSeriesSplit(n_splits=self.eval_splits)
        X = self.df

        # Find X_train and X_compare
        for train_index, test_index in tscv.split(X):
            X_train = X.iloc[train_index]
            X_compare_2020 = X.iloc[test_index]

        # Find X_val+test
        X_val = X_train.loc[(X_train.index.date > datetime.date(2019, 9, 30))]

        # Find X_Train
        X_train = X_train.loc[~X_train.index.isin(X_val.index)]

        # Find exact X_val and X_test
        tscv = TimeSeriesSplit(n_splits=2)
        for train_index, test_index in tscv.split(X_val):
            X_val_2 = X_val.iloc[train_index]
            X_test_2_month = X_val.iloc[test_index]

        self.X_train = X_train
        self.X_test = X_test_2_month
        self.X_val = X_val_2
        self.X_compare = X_compare_2020

    def test_on_splits(self, df, folds = 6):
        """Tests model fit on increasingly large subsets of the data, producing
        the Andrew Ng "bias-variance diagnostic" graph.
        
        Args:
            df: the dataframe to test over. It must have a MultiIndex like that
            created by add_csv_data.
            folds: number of testing points to calculate and plot

        Returns: None, but saves plots and prints summary statistics as it runs.
        
        If multiple cities are present in df, then the training folds include an
        equal proportion of data from each. Folds are determined by time, not
        by city.
        """
        sample_sizes = []
        start_time_train = df.index.get_level_values('time').min()
        end_time_train = df.index.get_level_values('time').max()
        num_hours = (end_time_train - start_time_train)/pd.Timedelta(hours=1) + 1
        train_ind, test_ind = self.split_for_splits(folds, num_hours)
        next_start_time_train = start_time_train
        for i in range(len(train_ind)):
            # Set training date ranges, both cumulative (for testing), and incremental (for training)
            end_time_train = start_time_train + pd.Timedelta(train_ind[i], 'h')
            cumul_train_range = pd.date_range(start=start_time_train, end=end_time_train, freq='h')
            increm_train_range = pd.date_range(start=next_start_time_train, end=end_time_train, freq='h')
            next_start_time_train = end_time_train + pd.Timedelta(1, 'h')

            # Set validation date range
            start_time_val = end_time_train + pd.Timedelta(1, 'h')
            end_time_val = start_time_val + pd.Timedelta(test_ind[i], 'h')
            val_range = pd.date_range(start=start_time_val, end=end_time_val, freq='h')

            # Fit on incremental training data
            self.fit_model(df.loc[(slice(None), increm_train_range), :])

            # Evaluate results on validation and cumulative training data
            pred_train_df = self.predict_model(self.train_df.loc[(slice(None), cumul_train_range), :])
            sample_sizes.append(len(pred_train_df))
            pred_train_df = self.prep_eval_data(pred_train_df)
            self.raw_metrics(y_true=pred_train_df.loc[:,'load'], y_pred=pred_train_df.loc[:, 'load_pred'], prefix='train', save_outputs=False, swoth=True)
            pred_val_df = self.predict_model(self.train_df.loc[(slice(None), val_range), :])
            pred_val_df = self.prep_eval_data(pred_val_df)
            self.raw_metrics(y_true=pred_val_df.loc[:,'load'], y_pred=pred_val_df.loc[:, 'load_pred'], prefix='val', save_outputs=False, swoth=True)

        # Run full metrics on final version
        self.calc_metrics(y_true=pred_val_df.loc[:,'load'], y_pred=pred_val_df.loc[:, 'load_pred'], dates_arr=val_range,
                            train_y_true=pred_train_df.loc[:,'load'], train_y_pred=pred_train_df.loc[:, 'load_pred'], train_dates_arr=cumul_train_range,
                            swoth=True)

        # Create bias-variance diagnostic plot
        filename_l = ["MSE","RMSE","MAPE"]
        for i in range(len(filename_l)):
            train_error = [self.train_errors[j][i] for j in range(len(self.train_errors))]
            val_error = [self.val_errors[j][i] for j in range(len(self.val_errors))]
            plt.figure()
            plt.plot(sample_sizes, train_error[:-1],label="training error")
            plt.plot(sample_sizes, val_error[:-1], label="validation error")
            plt.legend()
            plt.xlabel("Training Size (Number of Windows)")
            plt.ylabel("Error")
            plt.title(f"{filename_l[i]} vs Training Size")
            filename = f'error_scatter_test_splits_{filename_l[i]}.png'
            plt.savefig(os.path.join(os.getcwd(), "model", filename))
            plt.close()


    def split_for_splits(self, folds, length):
        adj = self.window - 1
        nsplits_len_train = [i * (length-adj) // (folds + 1) + (length-adj) % (folds + 1) - 1 if i* (length-adj) // (folds + 1) <= int(0.8 * (length-adj) // 1) else int(0.8*(length-adj)) for i in np.arange(1, folds+1)]
        nsplits_len_test = [int((i - adj) / 0.8 * 0.2 // 1) + adj for i in nsplits_len_train]
        return nsplits_len_train, nsplits_len_test

    def calc_metrics(self, y_true, y_pred, dates_arr, save_outputs=True,
                     train_y_true=None, train_y_pred=None, train_dates_arr=None, swoth = False):
        '''
        Calculates some metrics and plots for current run of an LSTM and saves them
        to a folder under the root/models s ubdirectory
        Parameters
        ----------
        y_true : numpy array of shape (m,1)
            True true load values
        y_pred : numpy array of shape (m,1)
            Predicted load values
        dates_arr : numpy array of shape (m,1)
            Each entry in numpy array should contain a unique date_hour string in
            the format: 'YYYY-MM-DD HH:MM:SS'
        hyperparams : dictionary
            Contains hyperparameter specifications
        save_outputs : bool, optional
            If True, will save outputs to a generated timestamped directory under
            the root/models subdirectory
        Returns
        -------
        None.
        '''

        output_dir = ''  # Placeholder if save_outputs is False
        if save_outputs:
            path = os.path.join(os.getcwd(), "model")
            if not os.path.exists(path):
                os.mkdir(path)
                print("Creating model directory")
            else:
                print("Directory exists, adding to directory")
            # Path to data directory contaning CSVs
            output_dir = path

            run_datetime = datetime.datetime.now().strftime('%m%d%y_%H%M%S')
            new_dir_name = f'run_{run_datetime}'

            if swoth:
                output_dir = os.path.join(output_dir, 'split_train_val')
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)
                self.outputpath = output_dir
            else:
                pass
            # Current date/time
            output_dir = os.path.join(output_dir, new_dir_name)
            os.mkdir(output_dir)

        # Error statistics #######################################################
        # self.outputpath = output_dir
        self.raw_metrics(y_true, y_pred, output_dir, save_outputs=save_outputs, prefix='val', swoth = swoth)
        if train_y_true is not None and train_y_pred is not None and train_dates_arr is not None:
            self.raw_metrics(train_y_true, train_y_pred, output_dir,save_outputs=save_outputs, prefix='train',swoth = swoth)

        # Scatter plot of Predicted vs True Load #################################

        self.plot_scatter(y_true, y_pred, output_dir, save_outputs=save_outputs, prefix='val')
        if train_y_true is not None and train_y_pred is not None and train_dates_arr is not None:
            self.plot_scatter(train_y_true, train_y_pred, output_dir, save_outputs=save_outputs, prefix='train')

        # Timeseries Plot (automatically only plots first two weeks of data) ######

        self.plot_timeseries(y_true, y_pred, dates_arr, output_dir, save_outputs=save_outputs, prefix='val')
        if train_y_true is not None and train_y_pred is not None and train_dates_arr is not None:
            self.plot_timeseries(train_y_true, train_y_pred, train_dates_arr, output_dir, save_outputs=save_outputs,
                            prefix='train')

        # Save current run's hyperparameters to JSON file ########################
        if save_outputs:
            self.hyperparameter_dict(output_dir)
        # Save arrays of predicted and true load value to CSV ####################
        self.save_CSV(y_true, y_pred, dates_arr, output_dir, save_outputs=save_outputs, prefix='val')
        if train_y_true is not None and train_y_pred is not None and train_dates_arr is not None:
            self.save_CSV(train_y_true, train_y_pred, train_dates_arr, output_dir, save_outputs=save_outputs, prefix='train')

        return

    def raw_metrics(self, y_true, y_pred, output_dir=None, save_outputs=True, prefix='val', swoth = False):
        '''
        Various useful regression metrics, the following link is a helpful reference
        https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
        '''

        print('------------------------------------------------------------------')
        print(f'{prefix} raw metrics')
        print('------------------------------------------------------------------')

        #UnScaled
        MSE = metrics.mean_squared_error(y_true, y_pred)
        RMSE = metrics.mean_squared_error(y_true, y_pred, squared=False)

        Mean_abs_error = metrics.mean_absolute_error(y_true, y_pred)
        Median_abs_error = metrics.median_absolute_error(y_true, y_pred)

        Max_error = metrics.max_error(y_true, y_pred)
        R2_score = metrics.r2_score(y_true, y_pred)

        MAPE = ((np.abs(y_true - y_pred)/y_true).sum())/len(y_true)

        metrics_str = ''
        metrics_str += f'Mean Squared Error: {MSE:.2f}\n'
        metrics_str += f'Root Mean Squared Error: {RMSE:.2f}\n'
        metrics_str += f'Mean Absolute Error: {Mean_abs_error:.2f}\n'
        metrics_str += f'Median Absolute Error: {Median_abs_error:.2f}\n'
        metrics_str += f'Max Error: {Max_error:.2f}\n'
        metrics_str += f'R2 Score: {R2_score:.3f}\n'
        metrics_str += f'Mean Absolute Percentage Error (MAPE): {MAPE:.4f}\n'

        print(metrics_str)

        if save_outputs:
            filename = f'{prefix}_error_stats.txt'
            with open(os.path.join(output_dir, filename), 'w') as f:
                f.write(metrics_str)

        if swoth:
            if prefix == "val":
                self.val_errors.append([MSE, RMSE, MAPE])
            elif prefix == "train":
                self.train_errors.append([MSE, RMSE, MAPE])
            elif prefix == "test":
                self.test_errors.append([MSE, RMSE, MAPE])

        return

    def hyperparameter_dict(self, output_dir):
        hyperparams = {"window": self.window, "layers": self.layers, "hidden_inputs": self.hidden_inputs,
                       "epochs": self.epochs}
        filename = 'hyperparameters.json'
        with open(os.path.join(output_dir, filename), 'w') as f:
            json.dump(hyperparams, f, indent=4)

    def plot_scatter(self, y_true, y_pred, output_dir, save_outputs=True, prefix='val'):
        '''
        Scatter plot of predicted vs true load, ideal is a line of slope 1
        '''

        plt.figure()
        plt.scatter(y_true, y_pred)
        plt.xlabel('True Load (MW)')
        plt.ylabel('Predicted Load (MW)')
        plt.title('Predicted vs True Load')

        if save_outputs:
            filename = f'{prefix}_scatter.png'
            plt.savefig(os.path.join(output_dir, filename))

        return

    def plot_timeseries(self, y_true, y_pred, dates_arr, output_dir, save_outputs=True, prefix='val'):
        '''
        Timeseries plot of predicted/true load vs datetime to see how well the
        prediction values match the true curve
        '''

        dates_arr_plt = self.to_dt(dates_arr)

        plt.figure()

        start_ind = 0
        end_ind = 336  # First two weeks
        end_ind = min(end_ind, len(dates_arr_plt))

        plt.plot_date(dates_arr_plt[start_ind:end_ind], y_true[start_ind:end_ind], 'b:', label='True')
        plt.plot_date(dates_arr_plt[start_ind:end_ind], y_pred[start_ind:end_ind], 'g:', label='Predicted')
        plt.xticks(rotation=45)
        plt.xlabel('Date')
        plt.ylabel('Load (MW)')
        plt.title('2-Week Predicted vs True Load')
        plt.legend()

        if save_outputs:
            filename = f'{prefix}_timeseries.png'
            plt.savefig(os.path.join(output_dir, filename))

        return

    def save_CSV(self, y_true, y_pred, dates_arr, output_dir, save_outputs=True, prefix='val'):
        '''
        Save CSV file of predicted and true load
        '''
        if save_outputs:
            df_out = y_true.to_frame().join(y_pred.to_frame())
            filename = f'{prefix}_pred_array.csv'
            output_filepath = os.path.join(output_dir, filename)
            pd.DataFrame(df_out).to_csv(output_filepath, index=True)

    def get_datetime(self, dt_str):
        '''
        Helper function to get a datetime object from a date_hour string
        '''

        dt_obj = datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')

        return dt_obj

    def to_dt(self, dt_arr):
        '''
        Helper function to convert array of date_hour timestrings to datetime objects
        '''

        # If already a list of datetime objects, just return the list
        if isinstance(dt_arr[0], datetime.datetime):
            return dt_arr

        # Other wise convert from strings to datetime objects
        N = len(dt_arr)

        return_arr = [''] * N

        j = 0
        for i in range(N):
            return_arr[j] = self.get_datetime(dt_arr[i])
            j += 1

        return return_arr


def main():
    # For now, just run on one city at a time
    cities = ['houston', 'boston', 'nyc', 'chicago', 'kck']
    data_dir = os.path.join(os.getcwd(), 'data', 'interim')

    keep_cols = ['hour', 'weekday', 'weekend', 'pre_weekend',
                'post_weekend', 'holiday', 'dwpc', 'relh', 'sped', 'tmpc', 'load',
                'city_flag_la', 'city_flag_houston', 'city_flag_boston', 'city_flag_nyc',
                'city_flag_chicago', 'city_flag_kck', 'week_of_year', 'sin_week_of_year',
                'cos_week_of_year', 'sin_hour', 'cos_hour']

    # Train model and predict
    lstm = LSTM_Model(df=None, window=24, layers=2, hidden_inputs=50, last_layer="Dense", scaler="Standard", epochs=1,
                      activation="relu", preserve_weights=True)

    lstm.run_experiment(cities, data_dir, test_on_split=True, folds=7, input_keep_cols=keep_cols)
    #for city in cities:
    #    print(f"Fitting on data from {city}")
    #    lstm.run_experiment(city, data_dir, test_on_split=True, folds=7, input_keep_cols=keep_cols)

    return


if __name__ == "__main__":
    main()