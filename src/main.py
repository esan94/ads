from argparse import ArgumentParser
from os import getcwd, listdir, path

from loguru import logger
from ml.train import train
from ml.forecast import forecast


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--step", type=str, 
        help="Represent the step to execute.", 
        choices=["train", "forecast"]
        )
    parser.add_argument(
        "--path", type=str, 
        help="Path to read data for training or to save data for forecating.",
        )
    args = parser.parse_args()

    if args.step == "train":
        logger.info("Step initiated: Training.")
        train()
    elif args.step == "forecast":
        if not listdir(path.join(getcwd(), "models")):
            error_msg = (
                "Before running the forecast script, you have to train a model with "
                "the parameter `--step train`."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.info("Step initiated: Forecasting.")
        forecast(path=args.path)
    else:
        error_msg = "The available value for `step` are 'train' and 'forecast'."
        raise ValueError(error_msg)

    logger.info("Step ended.")
