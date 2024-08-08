import pandas as pd
import pytz
import os
import data.opsd_time_series_2020_10_06 as load_data
import data.opsd_weather_data_2020_09_16 as weather_data
import importlib.resources


class ElectricityLoadTimeSeries:
    """ A class for handling and pre-processing time series in Pandas DataFrame. """

    def __init__(self):
        """The DataFrame has a DateTimeIndex and the Electricity Consumption Load is in the first column.
        Any additional column is intended to be a feature for the Linear Regressor, and must be either of type int or float."""
        self.ts = pd.DataFrame(data=None)

    # ----- Filter database -----
    def filter_complete_data(self):
        """ Filter database and replace it with a complete database, i.e. a database where
         all years, month and days are complete up to a certain tolerance level. """
        self.filter_non_null_entries()
        self.filter_complete_days()
        self.filter_complete_months()
        self.filter_complete_years()

    def filter_non_null_entries(self):
        """Filter non-null entries."""
        self._validate_df()
        self.ts = self.ts.dropna()

    def filter_complete_years(self, tolerance=11 / 12):
        """ Filters DataFrame to include only years that contain a sufficient number of months,
        based on a specified tolerance. A year is considered complete if the number of unique months present
        in the year meets or exceeds a percentage (tolerance) of the expected number of months (which is 12).

        The method performs the following steps:
        1. **Extract Date Components**: Extract year, month, and day from the DataFrame's DatetimeIndex and add these as columns.
        2. **Calculate Actual Number of Months per Year**: Group the ts by year and count the number of unique months present in each year.
        3. **Calculate Expected Number of Months per Year**: Set the expected number of months per year to 12.
        4. **Calculate Completion Percentage**: Compute the ratio of actual months to expected months for each year.
        5. **Filter Based on Tolerance**: Retain only years when the percentage of actual months compared to the
        expected months meets or exceeds the specified tolerance.
        6. **Update DataFrame**: Reset the index, merge the filtered ts to retain only the complete years, and set the
        original DatetimeIndex. Drop columns used for filtering purposes.

        Parameters:
        tolerance (float): The minimum percentage of months required to consider a year complete. Default is 11/12.

        Notes:
        - This method modifies the DataFrame in place.
        - The DataFrame should have a DatetimeIndex to extract date components correctly.
        - The method assumes that a year is considered complete if it contains at least the specified percentage of months.

        Raises:
        ValueError: If the DataFrame index is not a `pandas.DatetimeIndex`.
        """

        self._validate_df()
        # 1. Extract date components.
        self.augment_time_series_with_year_month_day()
        # 2. Calculate actual number of days per month.
        months_in_year = self.ts.groupby('year')['month'].nunique().reset_index(name='actual_years')
        # 3. Calculate expected number of days per month.
        months_in_year['expected_years'] = 12
        # 4. Calculate the percentage of completion
        months_in_year['%_of_completion'] = months_in_year['actual_years'] / months_in_year['expected_years']
        # 5. Filter DatFrame based on tolerance criteria.
        complete_years = months_in_year[months_in_year['%_of_completion'] >= tolerance]
        self.ts.reset_index(inplace=True)
        self.ts = self.ts.merge(complete_years, on=['year'])
        self.ts.set_index('utc_timestamp', inplace=True)
        # 6. Update dataframe by dropping all columns used for filtering purposes only.
        self.ts.drop(columns=['actual_years', 'expected_years', '%_of_completion'], inplace=True)
        self.drop_year_month_day()

    def filter_complete_months(self, tolerance=0.95):
        """ Filter DataFrame to include only months where the number of days is close to the expected number of days,
        within a specified tolerance. A month is considered complete if the number of days present in the month meets or exceeds a percentage (tolerance)
        of the expected number of calendar days. By default, the tolerance level is set to 95%.

        The method performs the following steps:
        1. **Extract Date Components**: Extract year, month, and day from the DataFrame's DatetimeIndex and add these as columns.
        This simplifies the logic ahead, with a minor cost on the memory use.
        2. **Calculate Actual Days per Month**: Group the ts by year and month, and count the number of unique days present in each month.
        3. **Calculate Expected Days per Month**: Determine the expected number of days in each month using the `pd.Timestamp` class to get
        the number of days in each month of the year.
        4. **Calculate Completion Percentage**: Compute the ratio of the actual number of days to the expected number of days for each month.
        5. **Filter Based on Tolerance**: Keep only those months where the percentage of actual days compared to the expected days is greater
        than or equal to the specified tolerance.
        6. **Update DataFrame**: Drop all columns used for filtering purposes only.

        Parameters:
        tolerance (float): The minimum percentage of days required in a month to consider it complete. Default is 0.95 (95%).

        Notes:
        - This method modifies the DataFrame in place.
        - The DataFrame should have a DatetimeIndex and contain columns that allow for the extraction of year, month, and day.

        Raises:
        ValueError: If the DataFrame index is not a `pandas.DatetimeIndex`.
        """
        self._validate_df()
        # 1. Extract date components.
        self.augment_time_series_with_year_month_day()
        # 2. Calculate actual number of days per month.
        days_in_month = self.ts.groupby(['year', 'month'])['day'].nunique().reset_index(name='actual_days')
        # 3. Calculate expected number of days per month.
        days_in_month['expected_days'] = days_in_month.apply(lambda row: pd.Timestamp(year=row['year'], month=row['month'], day=1).days_in_month, axis=1)
        # 4. Calculate the percentage of completion
        days_in_month['%_of_completion'] = days_in_month['actual_days'] / days_in_month['expected_days']
        # 5. Filter DatFrame based on tolerance criteria.
        complete_days = days_in_month[days_in_month['%_of_completion'] >= tolerance]
        self.ts.reset_index(inplace=True)
        self.ts = self.ts.merge(complete_days, on=['year', 'month'])
        self.ts.set_index('utc_timestamp', inplace=True)
        # 6. Update dataframe by dropping all columns used for filtering purposes only.
        self.ts.drop(columns=['actual_days', 'expected_days', '%_of_completion'], inplace=True)
        self.drop_year_month_day()

    def filter_complete_days(self, n_entries: int = None, tolerance=1.0):
        """ Filter DataFrame to include only days that have the expected number of entries within a specified tolerance.

        The method performs the following steps:
        1. **Extract Date Components**: Extract year, month, and day from the DataFrame's DatetimeIndex and add these as columns.
        2. **Calculate Actual Entries per Day**: Group the ts by year, month, and day, and count the number of entries per day.
        3. **Calculate Expected Entries per Day**:
            - If `n_entries` is provided, use it as the expected number of entries per day.
            - If `n_entries` is not provided, use the mode of the actual entries to estimate the expected number of entries.
        4. **Calculate Completion Percentage**: Compute the ratio of actual entries to expected entries for each day.
        5. **Filter Based on Tolerance**: Keep only those days where the percentage of actual entries compared to the expected entries
        is greater than or equal to the specified tolerance.
        6. **Update DataFrame**: Reset the index, merge the filtered ts to retain only the complete days, and reset the original
        DatetimeIndex. Drop intermediate columns used for filtering.

        Parameters:
        n_entries (int, optional): The expected number of entries per day. If not provided, the mode of the actual entries will be used.
        tolerance (float): The minimum percentage of entries required to consider a day complete. Default is 1.0 (100%).

        Notes:
        - The DataFrame should have a DatetimeIndex and contain columns that allow for the extraction of year, month, and day.
        - This method modifies the DataFrame in place.

        Raises:
        ValueError: If the DataFrame index is not a `pandas.DatetimeIndex`.
        """

        self._validate_df()
        # 1. Extract date components.
        self.augment_time_series_with_year_month_day()
        # 2. Calculate actual number of entries per day.
        entries_in_day = self.ts.groupby(['year', 'month', 'day']).size().reset_index(name='actual_entries')
        # 3. Calculate expected number of entries per day.
        # If number of entries is not provided, assume that the number of entries per day is equal to the mode of the number of entries per day
        if n_entries is None:
            mode = entries_in_day['actual_entries'].mode()[0]
            entries_in_day['expected_entries'] = mode
        else:
            mode = n_entries
            entries_in_day['expected_entries'] = n_entries
        # 4. Calculate percentage of completion for each day
        entries_in_day['%_of_completion'] = entries_in_day['actual_entries'] / entries_in_day['expected_entries']
        # 5. Filter dataframe based on tolerance criteria.
        complete_days = entries_in_day[(entries_in_day['%_of_completion'] >= tolerance) & (entries_in_day['actual_entries'] <= mode)]
        self.ts.reset_index(inplace=True)
        self.ts = self.ts.merge(complete_days, how='inner', left_on=['year', 'month', 'day'], right_on=['year', 'month', 'day'])
        self.ts.set_index('utc_timestamp', inplace=True)
        # 6. Update DataFrame by dropping columns used for filtering purposes only
        self.ts.drop(columns=['actual_entries', 'expected_entries', '%_of_completion'], inplace=True)
        self.drop_year_month_day()

    # ----- Intra-class Utilities -----
    def resample_days(self, frequency='1h'):
        """ Resamples inplace the time series data to a specified frequency on a per-day basis.
        The method performs the following operations:
        1. **Validation**: Ensures that the DataFrame or Series has a `DatetimeIndex`.
        2. **Grouping**: Groups the data by date.
        3. **Resampling**: Resamples each day's data to the specified frequency.
        4. **Interpolation**: Fills in missing values using linear interpolation.
        5. **Flattening**: Resets the index to remove the date grouping and reverts to a default integer index.

        Parameters:
        - `frequency` (str): The frequency to which the time series should be resampled.
        This should be a valid Pandas frequency string, such as '1H' for hourly, 'D' for daily, 'T' for minute, etc. Defaults to '1H'.

        Raises:
        - `ValueError`: If the index of `self.ts` is not a `DatetimeIndex`.

        Note: Interpolation only fills gaps between existing data points. If there are new timestamps created at the end or beginning
        of the resampling period with no surrounding data to interpolate from, those values will remain None.
        """
        self._validate_df()
        self.ts = (self.ts.groupby(self.ts.index.date).
                   apply(lambda x: x.resample(frequency).interpolate().ffill().bfill()).
                   reset_index(level=0, drop=True))

    def augment_time_series_with_day_of_the_week(self):
        """ Extract 'day_of_the_week' from 'utc_timestamp' and add it as column to DataFrame. """
        self._validate_df()
        self.ts['day_of_the_week'] = self.ts.index.dayofweek

    def augment_time_series_with_year_month_day(self):
        """ Extract year, month and day from DateTimeIndex and add them as columns"""
        self._validate_df()
        self.ts['year'] = self.ts.index.year
        self.ts['month'] = self.ts.index.month
        self.ts['day'] = self.ts.index.day

    def convert_utc_to_local_timestamp(self, timezone: str = 'Europe/Rome'):
        """ Replace DateTimeIndex with local_timestamp taking into account also for daylight saving time.
        Note: the timezone passed must be any of the Olson Timezone Database, or an error will be raised. """
        self._validate_df()
        local_tz = pytz.timezone(timezone)
        self.ts.index = self.ts.index.tz_convert(local_tz)

    def drop_year_month_day(self):
        """ Drop year, month and day columns from DataFrame. """
        self._validate_df()
        self.ts = self.ts.drop(columns=['year', 'month', 'day'])

    def sort(self):
        """ Sort dataframe in chronological order with respect to DateTimeIndex. """
        self._validate_df()
        self.ts.sort_index()

    # ----- Load and Save -----
    def load_time_series(self, file_path, cols_to_load: dict = None):
        """ Load a time series from a CSV file. If a dictionary cols_to_load specifying columns to load and their corresponding
        names in the DataFrame is provided, the method uses these mappings to rename the columns. This method can be called
        multiple times to load and merge multiple time series from different CSV files as the time series in the CSV files
        are attached to the DataFrame via an outer join on 'utc_timestamp'.
        Note 1: the CSV file to load must include a timestamp column in ISO 8601 format. If not, raises an error.
        Note 2: does not check if multiple load columns are being merged, resulting in a dataframe with load_x and load_y after merging.

        :param file_path: The path to the .csv file to be loaded.
        :param cols_to_load: A dictionary specifying which columns to load and the names to give them in the destination DataFrame.
            Example:
            cols_to_load = {
                'csv_col_name1': 'df_col1',
                'csv_col_name2': 'utc_timestamp',
                'csv_col_name3': 'df_col2',
            }
        """
        # Validate file_path
        if file_path is None:
            raise ValueError('file_path is not a valid file path.')
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f'The file at {file_path} does not exist.')

        # Load a set of columns from csv file
        if cols_to_load is None:
            loaded_df = pd.read_csv(file_path)
        else:
            if not isinstance(cols_to_load, dict):
                raise TypeError('cols_to_load must be a dictionary.')
            loaded_df = pd.read_csv(file_path, usecols=cols_to_load.keys())
            for (csv_col_name, df_col_name) in cols_to_load.items():
                loaded_df.rename(columns={csv_col_name: df_col_name}, inplace=True)

        # Check if loaded df has a column called utc_timestamp, and covert it to datetime format ISO8601
        if 'utc_timestamp' not in loaded_df.columns:
            raise ValueError("utc_timestamp not in DataFrame columns.")
        loaded_df['utc_timestamp'] = pd.to_datetime(loaded_df['utc_timestamp'], format='ISO8601', utc=True)

        # Merge with dataframe
        if not self.ts.empty:
            self.ts = self.ts.merge(loaded_df, how='outer', on='utc_timestamp', suffixes=('_l', '_r'))
        else:
            self.ts = loaded_df

        # Set utc_timestamp as index
        self.ts.set_index('utc_timestamp', inplace=True)

        # Validate time_series
        self._validate_df()

    def save_time_series(self, file_path: str = None):
        """ Save dataframe to CSV file. """
        if file_path is None:
            file_path = os.path.join(os.getcwd(), 'ts.csv')
        # Check if directory exists
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            raise FileNotFoundError(f'The directory {directory} does not exist.')
        self.ts.to_csv(file_path, sep=',')

    # ----- Load Example DataSet -----
    def load_example_entsoe_transparency(self, country: str):
        """ Retrieve local dataset by country and set attribute. The dataset includes load and meteorological ts. """
        # Load electricity load time series
        file_path = importlib.resources.files(load_data).joinpath(f'time_series_60min_singleindex_edited_{country.upper()}.csv')
        cols_to_load = {
            'utc_timestamp': 'utc_timestamp',
            f'{country.upper()}_load_actual_entsoe_transparency': 'load'
        }
        self.load_time_series(file_path, cols_to_load)

        # Load weather time series
        file_path = importlib.resources.files(weather_data).joinpath(f'weather_data_edited_{country.upper()}.csv')
        cols_to_load = {
            'utc_timestamp': 'utc_timestamp',
            f'{country.upper()}_temperature': 'temperature',
            f'{country.upper()}_radiation_direct_horizontal': 'direct_radiation',
            f'{country.upper()}_radiation_diffuse_horizontal': 'diffuse_radiation'
        }
        self.load_time_series(file_path, cols_to_load)

    # ----- Validation -----
    def _validate_df(self):
        """ Check if dataframe is not empty and index is of type DateTimeIndex"""
        if self.ts.empty:
            raise ValueError('Dataframe is empty.')

        if not isinstance(self.ts.index, pd.DatetimeIndex):
            raise TypeError('Index must be DateTimeIndex.')

        if not all(pd.api.types.is_numeric_dtype(self.ts[col]) for col in self.ts.columns):
            raise TypeError('Dataframe columns must be of numeric type.')
