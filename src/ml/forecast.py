from joblib import load
from os import getcwd, sep
from os.path import join

from pandas import DataFrame, date_range

from toolbox.data import Data

def forecast(path: str) -> None:
    """
    Reads time series data, loads a pre-trained model, makes a forecast, and saves the forecasted values to a CSV file.

    Parameters
    ----------
    path : str
        The file path to the CSV file containing the time series data.

    Returns
    -------
    None
    """
    data = Data(path=path).read()
    model = load(f"{join(getcwd(), 'models')}{sep}model.jl")
    fcast = model.forecast(data=data["target"])
    dates = date_range(start=data["dates"].max(), periods=13, freq="MS")[1:]
    dates = [date.strftime("%d-%m-%y").replace("-", ".") for date in dates]
    DataFrame(data=fcast.values, index=dates, columns=["y"]).to_csv(f"{join(getcwd(), 'data')}{sep}fcast.csv")
