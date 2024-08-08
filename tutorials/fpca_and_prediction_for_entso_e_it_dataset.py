from fpca_load.time_series import ElectricityLoadTimeSeries
from fpca_load.fpca import ElectricityLoadFPCA
from fpca_load.prediction import ElectricityLoadRegression

ts = ElectricityLoadTimeSeries()
ts.load_example_entsoe_transparency(country="de")
ts.sort()
ts.convert_utc_to_local_timestamp()
ts.filter_complete_data()
ts.resample_days()
ts.augment_time_series_with_year_month_day()
ts.augment_time_series_with_day_of_the_week()
ts.save_time_series()

fpca = ElectricityLoadFPCA(time_series=ts)
fpca.plot_functional_boxplot()
fpca.apply_fpca_to_all_days_grouped_by_date()
fpca.apply_fpca_to_all_days_grouped_by_weekday()
fpca.apply_fpca_to_all_days_grouped_by_month()
fpca.save_fpca_results()
# fpca.load_fpca_results('/Users/berri/GitHub Repositories/fpca-load-tools/tutorials/fpca.pkl')
fpca.plot_cdf_of_explained_variability()
fpca.plot_fpc()
fpca.plot_scores_vs_day_of_the_week()
fpca.plot_scores_vs_month_of_the_year()

prediction = ElectricityLoadRegression(fpca=fpca)
prediction.train_linear_model(n_fpc=3)
prediction.save_model()
prediction.predict_daily_electricity_load_curve(date='2019-02-21')
prediction.predict_daily_electricity_load_curve(date='2019-06-16')
prediction.predict_daily_electricity_load_curve(date='2019-07-09')
