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

from src.features.parallel_feature_engineering import feature_engineer_spark
from src.models.parallel_trainer import ParallelTrainer
from src.models.metrics import amex_metric
from src.utils import setup_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--model", type=str, default="LR")
    parser.add_argument("--config", type=str, default="./configs/parallel.yaml")
    args = parser.parse_args()
    logger = setup_logger("Parallel")

    model_config = Box.from_yaml(filename=args.config).get(args.model)

    # * Load Data
    DATA_PATH = Path(args.data_dir)
    # train = pd.read_csv(DATA_PATH / "raw" / "train_data.csv", nrows=1000)
    target = pd.read_csv(DATA_PATH / "train_labels.csv")
    # train = spark.read.csv(str(DATA_PATH / "train.csv")).limit(100)  # removed noise
    # test = spark.read.csv(str(DATA_PATH / "test.csv")).limit(100)
    train = spark.read.parquet(str(DATA_PATH / "train.parquet"))  # removed noise
    test = spark.read.parquet(str(DATA_PATH / "test.parquet"))

    train = train.drop("D_63", "D_64")
    test = test.drop("D_63", "D_64")

    train = feature_engineer_spark(train)
    test = feature_engineer_spark(test)

    # * transform target to spark dataframe
    target = spark.createDataFrame(target)
    train = train.join(target, on="customer_ID", how="left")

    feats = [
        col
        for col in train.columns
        if col not in ["customer_ID", "target", "S_2", "D_63", "D_64"]
    ]

    train = train.fillna(-1)
    test = test.fillna(-1)

    # * Train Model
    logger.info(f"Model Config: {model_config}")

    plr_trainer = ParallelTrainer(spark)
    oof_pred = plr_trainer.kfold_train(
        train_df=train,
        feats=feats,
        target="target",
        **model_config.to_dict(),
    )

    test_pred = plr_trainer.predict(test)
    test_target = pd.read_csv(DATA_PATH / "test_labels.csv")
    test_pred = test_pred.merge(test_target, on="customer_ID", how="left")
    logger.info(amex_metric(test_pred["target"], test_pred["prediction"]))
