from XGBoost import xgboost_parameterSearch
from linearRegression import linearRegression
from neuralNetwork import 
import dataProcessing

def testLinearRegression():
    MAE, MSE, RMSE = linearRegression()
    print("The mean absolute error is %s \n The mean squared error is %s \n The root mean squared error is %s" % (MAE, MSE, RMSE))
    return MAE, MSE, RMSE

def testXGBoost():
    result_grid = xgboost_parameterSearch()

    #The following are methods of extracting data from the raytune results grid
    if result_grid.errors:
        print("At least one trial failed.")
    best_result = result_grid.get_best_result()
    best_checkpoint = best_result.checkpoint
    MAE, RMSE = best_result.metrics
    MSE = RMSE**2
    results_df = result_grid.get_dataframe()
    print("Shortest training time:", results_df["time_total_s"].min())
    for result in result_grid:
        if result.error:
            print("The trial had an error:", result.error)
            continue
        print("The trial finished successfully with the metrics:", result.metrics["loss"])
    return MAE, MSE, RMSE

