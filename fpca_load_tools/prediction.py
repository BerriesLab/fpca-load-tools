import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.colors import Normalize
from fpca_load_tools.fpca import ElectricityLoadFPCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler


class ElectricityLoadRegression:
    """ A class to predict daily electricity load curves, based on the FPCs obtained on daily time series grouped by date.
    The class can be instantiated either by providing an ElectricityLoadFPCA object or by passing None, in which case the
    class will create an empty ElectricityLoadFPCA instance.

    Note: The ElectricityLoadFPCA object is stored as a reference (not as copy) in ElectricityLoadRegression.
    Therefore, any modifications to the ElectricityLoadFPCA object outside ElectricityLoadRegression will affect the data
    being processed by the regressor.
    """

    def __init__(self, fpca: ElectricityLoadFPCA = None):

        # Validation block
        if fpca is None:
            self.fpca = ElectricityLoadFPCA()
        else:
            if isinstance(fpca, ElectricityLoadFPCA):
                self.fpca = fpca
            else:
                raise TypeError(f'fpca must be of type {ElectricityLoadFPCA}')

        self.model = {}
        self.feature_scaler = None

    # ----- Train and predict -----
    def train_linear_model(self, n_fpc: int = 3, train_start_date: str = None, train_end_date: str = None, plot=True):
        """ Train a linear regression model for the Functional Principal Components (FPCs) of the daily load curves
        and evaluate its performance. The method includes standardizing the features, fitting the model, validating the
        model, and plotting the results.

        Parameters:
        n_fpc (int): The number of FPCs to use in the model. Default is 3.
        train_start_date (str): The start date for the training period in 'YYYY-MM-DD' format. Default is None.
        train_end_date (str): The end date for the training period in 'YYYY-MM-DD' format. Default is None.

        Steps:
        1. Determine the number of FPCs if not provided.
        2. Prepare the training and validation datasets.
        3. Standardize the training features and targets.
        4. Define and fit the linear regression model on the scaled training data.
        5. Make predictions using the model.
        6. Evaluate the model using Mean Squared Error (MSE) and R-squared (R2) metrics.
        7. Plot the scatter plot of actual vs. predicted scores for each FPC.
        """

        # 1. Determine number of FPC if not provided
        if n_fpc > self.fpca.results.day['fpca'].n_components_:
            raise ValueError('The number of FPCs must be smaller than the number of data points per time series.')

        # 2. Prepare training and validation dataset
        X_train, X_val, y_train, y_val = self._train_val_split(train_start_date, train_end_date)

        # 3. Standardize training and validation features
        self.feature_scaler = StandardScaler()
        self.feature_scaler.fit(X_train)
        X_train_scaled = self.feature_scaler.transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)

        for idx_fpc in range(n_fpc):
            # 4. Initialize and fit the linear regression model (on the scaled training dataset)
            self.model[idx_fpc] = LinearRegression()
            self.model[idx_fpc].fit(X=X_train_scaled, y=y_train[:, idx_fpc])

            # 5. Run prediction on validation dataset
            y_pred = self.model[idx_fpc].predict(X_val_scaled)

            # 6. Evaluate the model
            rmse = root_mean_squared_error(y_val[:, idx_fpc], y_pred)
            r2 = r2_score(y_val[:, idx_fpc], y_pred)
            print(f'FPC{idx_fpc + 1}, MSE: {rmse:.02f}, r2: {r2:.02f}')

            # 7. Plot actual vs predicted scores
            if plot is True:
                self._plot_actual_vs_predicted_score(idx_fpc, y_pred, y_val)

        if plot is True:
            plt.show()

    def predict_daily_electricity_load_curve(self, date: str):
        """ Predicts daily electricity consumption patterns for date 'date'.
        Date must be in ISO8061 format: YYYY-MM-DD. Returns a list of prediction metrics."""

        # 1. Collect features for the day to predict
        xy_actual = self.fpca.ts.ts.loc[date].reset_index().to_numpy()[:, 0:2]
        features = self.fpca.ts.ts.loc[date].mean().to_numpy()[1:]

        # 2. Rescale features
        scaled_features = self.feature_scaler.transform(features.reshape(1, -1))

        # 3. Calculate predicted curve
        functions = []
        for idx_fpc in self.model.keys():
            score = self.model[idx_fpc].intercept_ + np.dot(self.model[idx_fpc].coef_, scaled_features.reshape(-1, 1))
            functions.append(score * self.fpca.results.day['fpca'].components_[idx_fpc])

        # 4. Evaluate the prediction
        cumulative_prediction = self.fpca.results.day['fpca'].mean_
        for idx_fpc in range(len(functions)):
            cumulative_prediction += functions[idx_fpc]

        rmse = root_mean_squared_error(cumulative_prediction, xy_actual[:, 1])
        r2 = r2_score(cumulative_prediction, xy_actual[:, 1])
        actual_energy = np.sum(xy_actual[:, 1])
        predicted_energy = np.sum(cumulative_prediction)
        ee = 100 * (predicted_energy - actual_energy) / actual_energy
        print(f'date: {date}, rmse = {rmse:.02f}, r2 = {r2:.02f}, %EE = {ee:.02f} %')

        self._plot_actual_vs_predicted_load_curve(functions, xy_actual)

        return rmse, r2, ee

    def _train_val_split(self, train_start_date: str = None, train_end_date: str = None):
        """ Determines and splits the dataset into training and validation datasets based on the provided date range.

        Parameters:
        train_start_date (str): The start date for the training period in 'YYYY-MM-DD' format. If None, the earliest date in the dataset is used.
        train_end_date (str): The end date for the training period in 'YYYY-MM-DD' format. If None, the end date is determined as the beginning
        of the year corresponding to the 75th percentile of the dataset's years.

        Returns:
        X_train (numpy.ndarray): The training features.
        X_val (numpy.ndarray): The validation features.
        y_train (numpy.ndarray): The training target values.
        y_val (numpy.ndarray): The validation target values.

        Steps:
        1. Determine the start date if not provided.
        2. Determine the end date if not provided.
        3. Split the dataset into training and validation groups.
        4. Calculate the mean and scores for the training and validation period.
        """
        # 1. Determine start date if not provided.
        if train_start_date is None:
            train_start_date = self.fpca.ts.ts.index.min()

        # 2. Determine end date if not provided: the beginning of the year corresponding to data's 3rd quantile.
        if train_end_date is None:
            train_end_year = int(np.quantile(self.fpca.ts.ts.index.year.values, q=0.75))
            train_end_date = self.fpca.ts.ts[self.fpca.ts.ts.index.year == train_end_year].index.min()

        print(f'Train start date: {train_start_date}\n'
              f'Train end date: {train_end_date}')

        # 3. Split the dataset into training and validation groups.
        # 3a. Determine train and validation groups. Note: [:-1] is necessary as DataFrame.loc includes also the last element, while numpy does not.
        train_groups = self.fpca.ts.ts.loc[train_start_date:train_end_date][:-1].groupby(self.fpca.ts.ts.loc[train_start_date:train_end_date][:-1].index.date)
        val_groups = self.fpca.ts.ts.loc[train_end_date:][:-1].groupby(self.fpca.ts.ts.loc[train_end_date:][:-1].index.date)

        # 3b. Determine index corresponding to start and end date for the scores' matrix
        start_date_row_index = list(self.fpca.ts.ts.groupby(self.fpca.ts.ts.index.date).indices).index(train_start_date.date())
        end_date_row_index = list(self.fpca.ts.ts.groupby(self.fpca.ts.ts.index.date).indices).index(train_end_date.date())

        # 4. Calculate the mean and scores for the training and validation period.
        X_train = train_groups.mean().to_numpy()[:, 1:]
        y_train = self.fpca.results.day['scores'][start_date_row_index:end_date_row_index]
        X_val = val_groups.mean().to_numpy()[:, 1:]
        y_val = self.fpca.results.day['scores'][end_date_row_index:]

        return X_train, X_val, y_train, y_val

    # ----- Plot -----
    def _plot_actual_vs_predicted_load_curve(self, functions, xy_actual):
        """ Plot scatter plot of the actual vs predicted scores. """
        plt.figure()
        plt.plot(xy_actual[:, 1], label='Measured', color='black', alpha=0.8, linewidth=4)
        # Accumulate the functions for plotting
        cmap = matplotlib.colormaps['Oranges']
        norm = Normalize(vmin=0, vmax=len(functions) + 2)
        cumulative_prediction = self.fpca.results.day['fpca'].mean_
        plt.plot(cumulative_prediction, label=f'Mean', linewidth=4, alpha=0.8, c=cmap(norm(1)))
        for idx_fpc in range(len(functions)):
            cumulative_prediction += functions[idx_fpc]
            plt.plot(cumulative_prediction, label=f'FPC{idx_fpc + 1}', linewidth=4, alpha=0.8, c=cmap(norm(idx_fpc + 2)))
        plt.xlabel('Time (h)')
        plt.ylabel('Load')
        plt.title('Actual vs Predicted Load')
        plt.tight_layout()
        plt.legend()
        plt.show()

    @staticmethod
    def _plot_actual_vs_predicted_score(idx_fpc, y_pred, y_val):
        plt.figure()
        plt.scatter(y_val[:, idx_fpc], y_pred, color='blue', edgecolor='k', alpha=0.7)
        plt.plot([y_val[:, idx_fpc].min(), y_val[:, idx_fpc].max()], [y_val[:, idx_fpc].min(), y_val[:, idx_fpc].max()], color='red', lw=2, linestyle='--')
        plt.xlabel('Actual Scores')
        plt.ylabel('Predicted Scores')
        plt.title(f'Actual vs Predicted Scores for FPC{idx_fpc + 1}')
        plt.tight_layout()

    # ----- Load and Save -----
    def save_model(self, file_path=None):
        """ Save model and scaler to file in pickle format. """
        if file_path is None:
            file_path = os.path.join(os.getcwd(), 'model.pkl')
        data_to_save = {'model': self.model,
                        'feature_scaler': self.feature_scaler}
        with open(file_path, 'wb') as file:
            pickle.dump(data_to_save, file)

    def load_model(self, file_path):
        """ Load model and scaler. """
        if file_path is None:
            raise ValueError("A valid file path must be provided")

        try:
            with open(file_path, 'rb') as file:
                loaded_data = pickle.load(file)
                self.model = loaded_data['model']
                self.feature_scaler = loaded_data['feature_scaler']
        except FileNotFoundError:
            raise FileNotFoundError(f"The file at {file_path} was not found.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the file: {e}")
