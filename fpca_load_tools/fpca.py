import matplotlib.cm
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from fpca_load_tools.time_series import ElectricityLoadTimeSeries
from sklearn.decomposition import PCA


class ElectricityLoadFPCAResults:
    """ A class to store the results of FPCA. """

    def __init__(self):
        # An attribute to store the results on time series grouped by date
        self.day = None
        # An attribute to store the results on time series grouped by date and day of the week
        self.day_of_the_week = None
        # An attribute to store the results on time series grouped by date, year and month.
        self.month_of_the_year = None


class ElectricityLoadFPCA:
    """ A class to performa Functional Principal Component Analysis (FPCA) on time series of electricity load.
    This class supports the following types of FPCA on daily load curves:
        - FPCA on daily curves, grouped by date.
        - FPCA on daily curves, grouped by day of the week.
        - FPCA on daily curves, grouped by month of the year.

    The class can be instantiated either by providing an ElectricityLoadTimeSeries object or by passing None,
    in which case the class will create an empty ElectricityLoadTimeSeries instance.

    Note: Results from each analysis type are stored in an instance of the `ElectricityLoadFPCAResults` class,
    which is an attribute of this class. Only one result per analysis type is maintained:
    executing an analysis again will overwrite the previous results. For example, performing 'apply_fpca_to_all_days_grouped_by_date'
    a second time will replace the results of the previous analysis.
    """

    def __init__(self, time_series: ElectricityLoadTimeSeries = None):
        # An attribute to store a reference of time-series.
        if time_series is None:
            self.ts = ElectricityLoadTimeSeries()
        else:
            if isinstance(time_series, ElectricityLoadTimeSeries):
                self.ts = time_series
            else:
                raise TypeError(f'time_series must be of type {ElectricityLoadTimeSeries}')

        # An attribute to store the results
        self.results = ElectricityLoadFPCAResults()

    # ----- FPCA -----
    def apply_fpca_to_all_days_grouped_by_date(self, n_fpc: int = None):
        """ Applies FPCA to all daily load curves grouped by date, and store the results in self.results.day.

        Parameters:
            n_fpc (int, optional): The number of principal components to compute. If not provided, the
            number of components will default to the number of data points per day.

        Process:
            1. Groups the time series data by date.
            2. Builds a data matrix compatible with sklearn.decomposition.pca.
            3. Applies FPCA
            4. Stores the results in `self.results.day`.
        """

        # 1. Group dataframe by date
        groups = self.ts.ts.groupby(self.ts.ts.index.date)

        # 2. Build data matrix for FPCA (one curve per row)
        data, n = self._build_data_matrix_for_fpca(groups)

        # 3. Apply PCA
        if n_fpc is None:
            n_fpc = n
        pca = PCA(n_components=n_fpc)
        pca.fit(data)
        scores = pca.transform(data)

        # 4. Save fpca
        self.results.day = {'fpca': pca, 'scores': scores}

    def apply_fpca_to_all_days_grouped_by_weekday(self, n_fpc: int = None):
        """ Applies FPCA to daily load curves grouped by day of the week, and store the results in self.results.day_of_the_week.

        Parameters:
            n_fpc (int, optional): The number of principal components to compute. If not provided, the
            number of components will default to the number of data points per day.

        Process:
            1. Initializes results: a dictionary having for keys the `day of the week` (0-6).
            2. Iterate through each day of the week.
            3. Filters dataframe on day of the week and group by date.
            4. Builds a data matrix compatible with sklearn.decomposition.pca.
            5. Applies FPCA.
            6. Stores the results in `self.results.day_of_the_week`.
        """
        # 1. Initialize results: a dictionary having for keys the `day of the week` (0-6)
        self.results.day_of_the_week = {}

        # 2. Iterate through each day of the week
        for day in range(7):

            # 3. Filter dataframe on day of the week and group dataframe by date
            filtered_df = self.ts.ts[self.ts.ts.index.dayofweek == day]
            groups = filtered_df.groupby(filtered_df.index.date)

            # 4. Build data matrix for FPCA (one curve per row)
            data, n = self._build_data_matrix_for_fpca(groups)

            # 5. Applies FPCA
            if n_fpc is None:
                n_fpc = n
            pca = PCA(n_components=n)
            pca.fit(data)
            scores = pca.transform(data)

            # 6. Save fpca
            self.results.day_of_the_week[day] = {'fpca': pca, 'scores': scores}

    def apply_fpca_to_all_days_grouped_by_month(self, n_fpc: int = None):
        """ Applies FPCA to daily load curves grouped by month of the years, and store the results in self.results.month_of_the_year.

        Parameters:
            n_fpc (int, optional): The number of principal components to compute. If not provided, the
            number of components will default to the number of data points per day.

        Process:
            1. Initializes results: a dictionary having for keys the `month of the year` (0-11).
            2. Iterate through each month of the year.
            3. Filters dataframe on month of the year and group by date.
            4. Builds a data matrix compatible with sklearn.decomposition.pca.
            5. Applies FPCA.
            6. Stores the results in `self.results.month_of_the_year`.
        """

        # 1 Initialize result
        self.results.month_of_the_year = {}

        # 2. Iterate through each month of the year
        for month in range(12):

            # 3. Filter dataframe on month of the year and group by date
            filtered_df = self.ts.ts[self.ts.ts.index.month == month + 1]
            groups = filtered_df.groupby(filtered_df.index.date)

            # 4. Build data matrix for FPCA (one curve per row)
            data, n = self._build_data_matrix_for_fpca(groups)

            # 5. Run FPCA
            if n_fpc is None:
                n_fpc = n
            pca = PCA(n_components=n)
            pca.fit(data)
            scores = pca.transform(data)

            # 6. Save fpca
            self.results.month_of_the_year[month] = {'fpca': pca, 'scores': scores}

    @staticmethod
    def _build_data_matrix_for_fpca(groups):
        """ A helper function to prepare data for sklearn.decomposition.pca.
        It returns a numpy matrix and the number of data points per time series. """
        n = groups.size().unique()[0]
        data = np.zeros(shape=(len(groups), n))
        for idx, (group_key, group) in enumerate(groups):
            data[idx, :] = group['load'].values
        return data, n

    # ----- Plot -----
    def plot_scores_vs_day_of_the_week(self, n_fpc: int = 3):
        """ Plot Boxplot of scores vs day of the week, one figure per FPC. """
        # Validate variables
        if self.results.day_of_the_week is None:
            raise ValueError('Plot Error: Must execute FPCA on "day of the week" before plotting')

        for idx_fpc in range(n_fpc):

            # Build a list of scores
            data_to_plot = []
            for day in range(7):
                scores = self.results.day_of_the_week[day]['scores']
                data_to_plot.append(scores[:, idx_fpc])

            # Plot
            plt.figure()
            plt.boxplot(data_to_plot, vert=True, patch_artist=True, tick_labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
            plt.xlabel('Day of the week')
            plt.ylabel('Score')
            plt.title(f'FPC n.{idx_fpc + 1}')
            plt.tight_layout()

        plt.show()

    def plot_scores_vs_month_of_the_year(self, n_fpc: int = 3):
        """ Plot Boxplot of scores vs month of the year, one figure per FPC. """

        # Validate variables
        if self.results.month_of_the_year is None:
            raise ValueError('Plot Error: Must execute FPCA on "month of the year" before plotting')

        for idx_fpc in range(n_fpc):

            # Build a list of scores
            data_to_plot = []
            for month in range(12):
                scores = self.results.month_of_the_year[month]['scores']
                data_to_plot.append(scores[:, idx_fpc])

            # Plot
            plt.figure()
            plt.boxplot(data_to_plot, vert=True, patch_artist=True, tick_labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            plt.xlabel('Month of the year')
            plt.ylabel('Score')
            plt.title(f'FPC n.{idx_fpc + 1}')
            plt.tight_layout()

        plt.show()

    def plot_cdf_of_explained_variability(self, n_fpc: int = None) -> plt.Figure:
        """ Plot the Cumulative Distribution Function of the explained variability percentage
        of daily power loads, as a function of the number of Functional Principal Components (FPCs). """

        # Validate variables
        if self.results.day is None:
            raise ValueError('Plot Error: Must execute FPCA before plotting')

        # Determine number of FPC to plot if not provided
        if n_fpc is None:
            n_fpc = len(self.results.day['fpca'].explained_variance_ratio_)

        # Compute the cumulative sum
        cdf = np.cumsum(self.results.day['fpca'].explained_variance_ratio_)[0: n_fpc]

        # Plot
        plt.figure()
        plt.plot(cdf, marker='o')
        plt.xlabel('Number of FPCs')
        plt.ylabel('CDF of explained variability')
        plt.show()

    def plot_fpc(self, n_fpc: int = 5):
        """ Plot the Functional Principal Components, rescaled by their explained variance ratio,
        of daily electricity consumptions. """

        # Validate variables
        if self.results.day is None:
            raise ValueError('Plot Error: Must execute FPCA before plotting')

        # Determine number of FPC to plot if not provided
        if n_fpc is None:
            n_fpc = len(self.results.day.singular_values_)

        # Plot
        plt.figure()
        plt.xlabel('Sample')
        plt.ylabel('FPC')
        cmap = matplotlib.cm.get_cmap(name='coolwarm')
        norm = matplotlib.colors.LogNorm(
            vmin=self.results.day['fpca'].explained_variance_ratio_[0: n_fpc].min(),
            vmax=self.results.day['fpca'].explained_variance_ratio_[0: n_fpc].max()
        )
        for idx, fpc in enumerate(self.results.day['fpca'].components_[0: n_fpc]):
            # Calculate rescaling factor and plot
            r = self.results.day['fpca'].explained_variance_ratio_[idx]
            plt.plot(r * fpc, marker='o', color=cmap(norm(r)), label=f'FPC {idx + 1}')
        plt.legend()
        plt.show()

    def plot_functional_boxplot(self):
        """ Plot 'functional boxplot' of daily electricity consumption curves. """
        # Prepare data
        groups = self.ts.ts.groupby(self.ts.ts.index.date)
        data = self._build_data_matrix_for_fpca(groups)[0]

        # Calculate median, quantiles and outliers
        median = np.median(data, axis=0)
        lower_quantile = np.percentile(data, 25, axis=0)
        upper_quantile = np.percentile(data, 75, axis=0)
        iqr = upper_quantile - lower_quantile
        lower_bound = lower_quantile - 1.5 * iqr
        upper_bound = upper_quantile + 1.5 * iqr
        outliers = (data < lower_bound) | (data > upper_bound)
        outlier_indices = np.any(outliers, axis=1)

        # Prepare Plot
        fig, ax = plt.subplots()
        time_points = np.arange(data.shape[1])
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Power')
        # Plot all curves with low opacity
        for curve in data:
            ax.plot(time_points, curve, color='black', alpha=0.05)
        # Highlight outliers
        for curve in data[outlier_indices]:
            ax.plot(time_points, curve, color='orange', alpha=0.25)
        # Plot the central region
        ax.fill_between(time_points, lower_quantile, upper_quantile, color='red', alpha=0.5, label='IQR')
        # Plot the median
        ax.plot(time_points, median, color='blue', label='Median', linewidth=4, alpha=0.5)
        ax.legend()
        plt.show()
        return fig

    # ----- Save and Load -----
    def save_fpca_results(self, file_path: str = None):
        """ Save results to file in pickle format. """
        if file_path is None:
            file_path = os.path.join(os.getcwd(), 'fpca.pkl')
        # Check if directory exists
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            raise FileNotFoundError(f'The directory {directory} does not exist.')
        with open(file_path, 'wb') as file:
            pickle.dump(self.results, file)

    def load_fpca_results(self, file_path: str = None):
        """ Load results from pickle file. """
        if file_path is None:
            raise ValueError("A valid file path must be provided")

        try:
            with open(file_path, 'rb') as file:
                self.results = pickle.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file at {file_path} was not found.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the file: {e}")
