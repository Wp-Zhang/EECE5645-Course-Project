from pathlib import Path
import warnings
import pandas as pd

from src.features.feature_engineering import feature_engineer
from src.models.local_trainer import LocalTrainer
from src.utils import setup_logger
import argparse
from box import Box

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--model", type=str, default="LR")
    parser.add_argument("--config", type=str, default="./configs/local.yaml")
    args = parser.parse_args()
    logger = setup_logger("Local")

    # * Load Data
    DATA_PATH = Path(args.data_dir)
    train = pd.read_csv(DATA_PATH / "raw" / "train_data.csv", nrows=100000)
    target = pd.read_csv(DATA_PATH / "raw" / "train_labels.csv")
    test = pd.read_csv(DATA_PATH / "raw" / "test_data.csv", nrows=100000)

    train.drop(columns=["S_2", "D_63", "D_64"], inplace=True)
    test.drop(columns=["S_2", "D_63", "D_64"], inplace=True)

    # * Feature Engineering
    train = feature_engineer(train)
    test = feature_engineer(test)
    train["customer_ID"] = train.index
    test["customer_ID"] = test.index
    train = train.merge(target, on="customer_ID", how="left")

    feats = [
        col
        for col in train.columns
        if col not in ["customer_ID", "target", "S_2", "D_63", "D_64"]
    ]

    # * Fill NaN
    train = train.fillna(-1)
    test = test.fillna(-1)

    # * Train Model
    model_config = Box.from_yaml(filename=args.config).get(args.model)
    logger.info(f"Model Config: {model_config}")

    lr_trainer = LocalTrainer()
    oof_pred = lr_trainer.kfold_train(
        train_df=train, feats=feats, target="target", **model_config.to_dict()
    )

    # * Make Prediction
    test_pred = lr_trainer.predict(test)
    logger.info(test_pred.head())
