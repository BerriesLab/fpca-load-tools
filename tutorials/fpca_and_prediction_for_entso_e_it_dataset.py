from fpca_load_tools.time_series import ElectricityLoadTimeSeries
from fpca_load_tools.fpca import ElectricityLoadFPCA
from fpca_load_tools.prediction import ElectricityLoadRegression

# Instantiate an ElectricityLoadTimeSeries object.
ts = ElectricityLoadTimeSeries()
# Load example data: Entso E Transparency, Germany.
ts.load_example_entsoe_transparency(country="de")
# Sort time series to make sure data are in chronological order.
ts.sort()
# Convert timestamp from UTC to local
ts.convert_utc_to_local_timestamp('Europe/Berlin')
# Resample days to a frequency of one hour, in order to account for multiple or missing entries due to the timestamp conversion
ts.resample_days('1h')
# Filter complete time series: this removes incomplete years, months and days.
ts.filter_complete_data()
# Augment time series with year, month, day, and weekday
ts.augment_time_series_with_year_month_day()
ts.augment_time_series_with_day_of_the_week()

# Instantiate an ElectricityLoadFPCA object by passing an ElectricityLoadTimeSeries object.
# Note that the ElectricityLoadTimeSeries is passed as a reference, not as copy.
# Therefore, any change made to the ElectricityLoadTimeSeries object outside the ElectricityLoadFPCA
# will affect the data processed by ElectricityLoadFPCA.
fpca = ElectricityLoadFPCA(time_series=ts)
# Display a functional plot of all daily time series
fpca.plot_functional_boxplot()
# Apply three types of FPCA to daily load curves: (i) grouped by date, (ii) grouped by day of the week, and (iii) grouped by month.
fpca.apply_fpca_to_all_days_grouped_by_date()
fpca.apply_fpca_to_all_days_grouped_by_weekday()
fpca.apply_fpca_to_all_days_grouped_by_month()
# Display the Cumulative Distribution Function of the explained variability ratio as a functon of the number of FPCs
fpca.plot_cdf_of_explained_variability()
# Display the FPCs
fpca.plot_fpc()
# Display the scores' boxplot vs day of the week and month of the year.
fpca.plot_scores_vs_day_of_the_week()
fpca.plot_scores_vs_month_of_the_year()

# Instantiate an ElectricityLoadRegression object by passing an ElectricityLoadFPCA object.
# Note that the ElectricityLoadFPCA is passed as a reference, not as copy.
# Therefore, any changes made to the ElectricityLoadFPCA object outside the ElectricityLoadRegression object
# will affect the data processed by ElectricityLoadRegression.
prediction = ElectricityLoadRegression(fpca=fpca)
# Train a linear model
prediction.train_linear_model(n_fpc=3)
# Predict electricity load curves for selected days and display the prediction.
prediction.predict_daily_electricity_load_curve(date='2019-02-21')
prediction.predict_daily_electricity_load_curve(date='2019-06-16')
prediction.predict_daily_electricity_load_curve(date='2019-07-09')
