from joblib import dump
from os import getcwd, path, sep

from toolbox.model import Model

from sktime.forecasting.exp_smoothing import ExponentialSmoothing


def train() -> None:
    """
    Trains an Exponential Smoothing model and saves the trained model to a file.

    The model is configured with no trend, multiplicative seasonality, and a seasonal period of 11.
    The trained model is saved to the 'models' directory with the filename 'model.jl'.

    Returns
    -------
    None
    """
    model = Model(model=ExponentialSmoothing, trend=None, seasonal="add", sp=11)
    dump(model, filename=f"{path.join(getcwd(), 'models')}{sep}model.jl")