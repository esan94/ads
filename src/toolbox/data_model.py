from typing import Dict, List

import pandas as pd
from matplotlib import pyplot as plt
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from sktime.split import temporal_train_test_split
from sktime.utils.plotting import plot_series


class DataModel:
    """
    A class to model and manipulate time series data from a DataFrame.

    Parameters
    ----------
    dataframe : pd.DataFrame
        A DataFrame containing the time series data. It must include
        columns named 'dates' and 'target'.

    Attributes
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing the time series data.

    Methods
    -------
    ttt_split(test_size: int = 12) -> Dict[str, List[pd.Series]]
        Splits the data into training and testing sets for cross-validation.
    
    get_metrics(forecast: pd.Series, true: pd.Series) -> Dict[str, float]
        Computes the MAPE, MAE, and RMSE metrics.

    plot_ts(past: pd.Series, future: pd.Series, forecasted: pd.Series, title: str = "TS Forecasting") -> None
        Plots the past, future, and forecasted time series data.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """
        Initializes the DataModel class with the given DataFrame.

        Parameters
        ----------
        dataframe : pd.DataFrame
            A DataFrame containing the time series data. It must include
            columns named 'dates' and 'target'.

        Raises
        ------
        TypeError
            If the input is not a pandas DataFrame.
        ValueError
            If the DataFrame does not contain the required columns.
        """
        ob_cols = {"dates", "target"}

        if not isinstance(dataframe, pd.DataFrame):
            error_msg = (
                "Parameter `DataModel.__init__.dataframe` must be a pd.DataFrame."
            )
            raise TypeError(error_msg)
        if len(set(dataframe.columns).intersection(ob_cols)) != 2:
            error_msg = (
                "Parameter `DataModel.__init__.dataframe` must contain "
                "a column named 'target' and other named 'dates'."
            )
            raise ValueError(error_msg)

        self.dataframe = dataframe

    def ttt_split(self, test_size: int = 12) -> Dict[str, List[pd.Series]]:
        """
        Splits the data into training and testing sets for cross-validation.

        Parameters
        ----------
        test_size : int, optional
            The size of the test set for cross-validation, by default 12.

        Returns
        -------
        Dict[str, List[pd.Series]]
            A dictionary with two cross-validation splits, each containing
            the training and testing sets.
        """
        train_cv1, test_cv1 = temporal_train_test_split(self.dataframe["target"], test_size=test_size * 2)
        test_cv1, _ = temporal_train_test_split(test_cv1, test_size=test_size)
        train_cv2, test_cv2 = temporal_train_test_split(self.dataframe["target"], test_size=test_size)
        return {
            "cv1": [train_cv1, test_cv1],
            "cv2": [train_cv2, test_cv2]
        }

    @staticmethod
    def get_metrics(forecast: pd.Series, true: pd.Series) -> Dict[str, float]:
        """
        Computes the MAPE, MAE, and RMSE metrics.

        Parameters
        ----------
        forecast : pd.Series
            Forecasted data.
        true : pd.Series
            True data to compare with the forecasted one.

        Returns
        -------
        Dict[str, float]
            Dictionary with "mape", "mae" and "rmse" metrics computed.
        """
        return {
            "mape": mean_absolute_percentage_error(y_true=true, y_pred=forecast),
            "mae": mean_absolute_error(y_true=true, y_pred=forecast),
            "rmse": mean_squared_error(y_true=true, y_pred=forecast, square_root=True)
        }

    @staticmethod
    def plot_ts(
        past: pd.Series, future: pd.Series, forecasted: pd.Series,
        title: str = "TS Forecasting"
    ) -> None:
        """
        Plots the past, future, and forecasted time series data.

        Parameters
        ----------
        past : pd.Series
            The past time series data.
        future : pd.Series
            The future (true) time series data.
        forecasted : pd.Series
            The forecasted time series data.
        title : str, optional
            The title of the plot, by default "TS Forecasting".
        
        Returns
        -------
        None
        """
        labels = ["Past", "Future", "Forecasted"]
        plot_series(
            past, future, forecasted,
            labels=labels,
            colors=["blue", "green", "orange"]
        )
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Values")
        plt.grid(True)
        plt.show()
