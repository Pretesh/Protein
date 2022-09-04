import dataProcessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

def linearRegression():
    data, labels = dataProcessing.preProcessing()
    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.20)
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    pred_y = reg.predict(X_test)
    MAE = metrics.mean_absolute_error(test_y, pred_y)
    MSE = metrics.mean_squared_error(test_y, pred_y)
    RMSE = np.sqrt(metrics.mean_squared_error(test_y, pred_y))
    return MAE, MSE, RMSE