from pathlib import Path
import warnings
import pandas as pd

from src.features.feature_engineering import aggregate_data
from src.models.local_trainer import LocalTrainer

warnings.filterwarnings("ignore")

DATA_PATH = Path("./data")


# * Load Data
train = pd.read_csv(DATA_PATH / "raw" / "train_data.csv", nrows=100000)
target = pd.read_csv(DATA_PATH / "raw" / "train_labels.csv")
test = pd.read_csv(DATA_PATH / "raw" / "test_data.csv", nrows=100000)

train.drop(columns=["S_2", "D_63", "D_64"], inplace=True)
test.drop(columns=["S_2", "D_63", "D_64"], inplace=True)
print(train.info())

# * Feature Engineering
train = aggregate_data(train)
test = aggregate_data(test)

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

# * Train Logistic Regression
lr_trainer = LocalTrainer()
oof_pred = lr_trainer.kfold_train(
    model_name="LR",
    model_params={"C": 0.9},
    train_df=train,
    feats=feats,
    target="target",
    n_splits=5,
    random_state=2023,
    normalize=True,
)

# * Make Prediction
test_pred = lr_trainer.predict(test)

print(test_pred.head())
