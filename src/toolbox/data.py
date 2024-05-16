import pandas as pd


class Data:
    """
    A class to represent and manipulate data from a CSV file.

    Parameters
    ----------
    path : str
        The file path to the CSV file.

    Attributes
    ----------
    path : str
        The file path to the CSV file.

    Methods
    -------
    read() -> pd.DataFrame
        Reads the CSV file, processes the dates, and returns a DataFrame.
    """

    def __init__(self, path: str) -> None:
        """
        Initializes the Data class with the given file path.

        Parameters
        ----------
        path : str
            The file path to the CSV file.
        """
        self.path = path

    def read(self) -> pd.DataFrame:
        """
        Reads the CSV file, processes the dates, and returns a DataFrame.

        The CSV file is expected to have columns "dates" and "target".
        The "dates" column is converted to datetime format with day first.
        Any rows with missing values are dropped.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the processed data.
        """
        data = pd.read_csv(self.path, names=["dates", "target"], header=0)
        data["dates"] = pd.to_datetime(data["dates"], dayfirst=True, format="mixed")
        data = data.dropna()
        return data
