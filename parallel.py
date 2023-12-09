from pathlib import Path
import warnings
import pyspark
import pandas as pd


warnings.filterwarnings("ignore")

DATA_PATH = Path("./data")

spark = (
    pyspark.sql.SparkSession.builder.appName("AMEX")
    # Use 0.11.4-spark3.3 version for Spark3.3 and 1.0.1 version for Spark3.4
    .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:1.0.1")
    .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

# * Load Data

#train = pd.read_csv(DATA_PATH / "raw" / "train_data.csv", nrows=1000)
target = pd.read_csv(DATA_PATH / "raw" / "train_labels.csv")
train = spark.read.parquet(DATA_PATH/ "raw" / "train.parquet") #removed noise
test = spark.read.parquet(DATA_PATH/ "raw" / "test.parquet")

train = train.merge(target, on="customer_ID", how="left")

from src.features.parallel_feature_engineering import aggregate_data_spark

#train = aggregate_data_spark(train)

feats = [
    col
    for col in train.columns
    if col not in ["customer_ID", "target", "S_2", "D_63", "D_64"]
]

train = train.fillna(-1)
test = test.fillna(-1)

# * Train Logistic Regression
from src.models.parallel_trainer import ParallelTrainer

plr_trainer = ParallelTrainer(spark)
oof_pred = plr_trainer.kfold_train(
    model_name="LR",
    model_params={"regParam": 0.1},
    train_df=train,
    feats=feats,
    target="target",
    n_splits=5,
    random_state=2023,
    normalize=True,
)

test_pred = plr_trainer.predict(test)

print(test_pred.head())
