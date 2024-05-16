# ADS CASE PROBLEM

## Project Structure

The structure of the project is the following:
- gpt: Folder containing the data web application connected with gpt to talk about the TS.
    - contexts: Folder containig the context of the system.
        - ts_context.txt: Context of the system.
    - web.py: Streamlit web app to create a simple chatbot.
- notebooks folder. In this folder we can find the following files (both notebooks are autoexplicative):
    - eda.ipynb: This is the Exploratory Data Analysis, here we can find a description of the TS.
    - train.ipynb: This is the train analysis, here we can find different train approaches.
- src folder: In this folder we can find the executable file for training and for predicting.
    - main.py: Executable file.
    - data: Folder to save the forecasted file for the problem.
        - fcast.csv: File of forecasting data for the problem.
    - ml: Folder to save the train and predict script.
        - forecast.py: Script for forecasting.
        - train.py: Script for training.
    - models: Folder to save the trained models.
        - model.jl: Joblib object representing the saved model.
    - toolbox: Folder with useful functionalities for notebooks and src folder.
        - data_model.py: File to manipulate TS data.
        - data.py: File for reading a transforming appropriatly problem data.
        - model.py: File for utilities for training and forecasting model and data.

## Virtual environment
- Use `pyproject.toml` to create the environment using poetry (recommended).
- Use `environment.yml` to create the environment using coda.
- Use `requirements.txt` to create the environment using pip.


## Execute the solution.
### Training
Over the folder `src` and with the virtual environment activated launch `python main.py --step train`.

### Forecasting
Over the folder `src`, with the virtual environment activated and after launching the training step type `python main.py --step forecast --path ..\train.csv`

## GPT
When the `fcast.csv` file is generated you will be able to launch (over the folder `gpt`) `streamlit run web.py`. Here you will find a chatbot gpt powered to talk about the TS and provide insights to the user.

## Summary
### How models works
#### ARIMA (AutoRegressive Integrated Moving Average)
ARIMA is a popular statistical method for time series forecasting. It combines three components:

1. **AutoRegressive (AR)**: This component models the relationship between an observation and a number of lagged observations.
2. **Integrated (I)**: This component involves differencing the data to make it stationary (i.e., removing trends and seasonality).
3. **Moving Average (MA)**: This component models the relationship between an observation and a residual error from a moving average model applied to lagged observations.

The general form of an ARIMA model is ARIMA(p, d, q), where:
- p is the number of lag observations in the model (order of AR term).
- d is the number of times that the raw observations are differenced (order of integration).
- q is the size of the moving average window (order of MA term).

#### Holt-Winters (Exponential Smoothing)
Holt-Winters is a time series forecasting method that accounts for seasonality in data. It has three components:

1. **Level (L)**: The baseline value for the series.
2. **Trend (T)**: The trend of the series over time.
3. **Seasonality (S)**: The seasonal component of the series.

There are two main versions of the Holt-Winters method:
- **Additive**: Used when seasonal variations are roughly constant over time.
- **Multiplicative**: Used when seasonal variations change proportionally with the level of the series.

The method uses exponential smoothing to update these components over time.

#### Random Forest
Random Forest is an ensemble learning method primarily used for classification and regression tasks, including time series forecasting. It operates by constructing a multitude of decision trees during training and outputting the mean prediction (regression) of the individual trees. 

For time series:
- Time-lagged features are created from the series.
- Multiple decision trees are built using subsets of these features.
- The final prediction is the average of predictions from all the trees.

Random Forest is powerful because it reduces overfitting and improves predictive accuracy by averaging multiple models.

#### Elastic Net
Elastic Net is a regularized regression technique that combines the penalties of both Lasso (L1) and Ridge (L2) methods. It is used for linear regression models and is particularly useful when dealing with multicollinear data (features that are highly correlated).

The Elastic Net objective function is:
[ElasticNet](https://en.wikipedia.org/wiki/Elastic_net_regularization) where:
- y is the target variable.
- X is the matrix of input features.
- beta is the vector of coefficients.
- lambda_1 controls the L1 penalty (Lasso).
- lambda_2 controls the L2 penalty (Ridge).

In time series:
- Lagged values of the series can be used as features.
- Elastic Net can handle many predictors, even when they are highly correlated.

### What is next?
Other techniques could be used here are:
- STL decomposition.
- Neural networks as LSTM for forecasting.
- Direct approach over the forecasting step.
- Probabilistic forecasting.
- Conformal predictions to predict ranges.
- Fourier Analysis to detect strange frequencies and filter this data points.
- Analysis of the correlated lags in the ML approach to reduce the number of variables.
- Test the effect of preprocessing techniques on the final output.
- More folds for cross-validation.

### Conclusion
After using bayessian optimization for the hyper parameters tunning over the different tested models. The best one in an stable way is the Holt-Winter due to the seasonality of the time series.
- ElasticNet: mape=0.256865, mae=0.593873, rmse=0.703187
- RF: mape=0.223113, mae=0.532756, rmse=0.669952
- HW: mape=0.230358, mae=0.536948, rmse=0.609464
- ARIMA: mape=0.232036, mae=0.542176, rmse=0.637554