from typing import Any, Dict, Optional

import pandas as pd
from sktime.forecasting.compose import make_reduction, TransformedTargetForecaster


class Model:
    """
    A class to handle forecasting models, supporting both scikit-learn and other types of models.

    Parameters
    ----------
    model : object
        The forecasting model to be used.
    is_sklearn : bool, optional
        Flag indicating whether the model is a scikit-learn compatible model, by default False.
    strategy : Optional[str], optional
        The strategy for scikit-learn reduction, either 'recursive' or 'direct', by default "recursive".
    fcast_length : int, optional
        The forecast length, by default 12.
    **kwargs : Dict[str, Any]
        Additional keyword arguments passed to the model.

    Attributes
    ----------
    fcast_horizon : list of int
        The forecast horizon as a list of integers.
    model : object
        The initialized forecasting model.

    Methods
    -------
    forecast(data: pd.Series) -> pd.Series
        Fits the model on the data and returns the forecasted values.
    """

    def __init__(
            self, model,
            is_sklearn: bool = False, strategy: Optional[str] = "recursive", fcast_length: int = 12,
            **kwargs: Dict[str, Any]) -> None:
        """
        Initializes the Model class with the given parameters.

        Parameters
        ----------
        model : object
            The forecasting model to be used.
        is_sklearn : bool, optional
            Flag indicating whether the model is a scikit-learn compatible model, by default False.
        strategy : Optional[str], optional
            The strategy for scikit-learn reduction, either 'recursive' or 'direct', by default "recursive".
        fcast_length : int, optional
            The forecast length, by default 12.
        **kwargs : Dict[str, Any]
            Additional keyword arguments passed to the model.

        Raises
        ------
        ValueError
            If `is_sklearn` is True and `strategy` is not a string.
        """
        self.fcast_horizon = list(range(1, fcast_length + 1))
        if is_sklearn:
            if not isinstance(strategy, str):
                error_msg = (
                    "When parameter `is_sklearn` is True, `strategy` must be set to 'recursive' or 'direct'."
                )
                raise ValueError(error_msg)
            self.model = TransformedTargetForecaster([
                ("model", make_reduction(model(**kwargs), window_length=fcast_length, strategy=strategy))
            ])
        else:
            self.model = model(**kwargs)

    def forecast(self, data: pd.Series) -> pd.Series:
        """
        Fits the model on the data and returns the forecasted values.

        Parameters
        ----------
        data : pd.Series
            The time series data to fit the model on.

        Returns
        -------
        pd.Series
            The forecasted values.
        """
        self.model.fit(data)
        return self.model.predict(fh=self.fcast_horizon)
