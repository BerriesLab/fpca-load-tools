from fpca_load.time_series import ElectricityLoadTimeSeries
import importlib.resources
import test

# Test loading corrupted files on timestamp
x = ElectricityLoadTimeSeries()
x.load_time_series(file_path=importlib.resources.files(test).joinpath('test_csv_corrupted_timestamp.csv'), cols_to_load={'timestamp': 'utc_timestamp', 'load': 'load'})
x.filter_complete_data()

# Test loading corrupted files on load (should be float)
x.load_time_series(file_path=importlib.resources.files(test).joinpath('test_csv_corrupted_load.csv'), cols_to_load={'timestamp': 'utc_timestamp', 'load': 'load'})

# Test loading multiple csv files with same column names
x.load_time_series(file_path=importlib.resources.files(test).joinpath('test_csv_ok.csv'), cols_to_load={'timestamp': 'utc_timestamp', 'load': 'load'})
x.load_time_series(file_path=importlib.resources.files(test).joinpath('test_csv_ok.csv'), cols_to_load={'timestamp': 'utc_timestamp', 'load': 'load'})

