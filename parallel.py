from pathlib import Path
import warnings
import pyspark
import pandas as pd
import argparse
from box import Box

warnings.filterwarnings("ignore")

spark = (
    pyspark.sql.SparkSession.builder.appName("AMEX")
    # Use 0.11.4-spark3.3 version for Spark3.3 and 1.0.1 version for Spark3.4
    .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:1.0.1")
    .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

from src.features.parallel_feature_engineering import aggregate_data_spark
from src.models.parallel_trainer import ParallelTrainer
from src.utils import setup_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--model", type=str, default="LR")
    parser.add_argument("--config", type=str, default="./configs/parallel.yaml")
    args = parser.parse_args()
    logger = setup_logger("Parallel")

    # * Load Data
    DATA_PATH = Path(args.data_dir)
    # train = pd.read_csv(DATA_PATH / "raw" / "train_data.csv", nrows=1000)
    target = pd.read_csv(DATA_PATH / "raw" / "train_labels.csv")
    train = spark.read.parquet(DATA_PATH / "raw" / "train.parquet")  # removed noise
    test = spark.read.parquet(DATA_PATH / "raw" / "test.parquet")

    train = train.merge(target, on="customer_ID", how="left")

    # train = aggregate_data_spark(train)

    feats = [
        col
        for col in train.columns
        if col not in ["customer_ID", "target", "S_2", "D_63", "D_64"]
    ]

    train = train.fillna(-1)
    test = test.fillna(-1)

    # * Train Model
    model_config = Box.from_yaml(filename=args.config).get(args.model)
    logger.info(f"Model Config: {model_config}")

    plr_trainer = ParallelTrainer(spark)
    oof_pred = plr_trainer.kfold_train(
        train_df=train,
        feats=feats,
        target="target",
        **model_config.to_dict(),
    )

    test_pred = plr_trainer.predict(test)
    logger.info(test_pred.head())
