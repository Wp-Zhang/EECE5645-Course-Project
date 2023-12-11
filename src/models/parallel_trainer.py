from typing import Dict, List, Tuple, Any, Literal
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline, PipelineModel


from synapse.ml.lightgbm import LightGBMClassifier
from .metrics import amex_metric
from ..utils import setup_logger, setup_timer

logger = setup_logger(__name__)
timer = setup_timer(logger)


class ParallelTrainer:
    """Parallel model trainer using Spark and LightGBM.

    Examples
    --------
    >>> import pyspark
    >>> spark = (
    ...     pyspark.sql.SparkSession.builder.appName("MyApp")
    ...     # Use 0.11.4-spark3.3 version for Spark3.3 and 1.0.1 version for Spark3.4
    ...     .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:1.0.1")
    ...     .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
    ...     .getOrCreate()
    ... )
    >>> from src.models.parallel_trainer import ParallelTrainer
    >>> trainer = ParallelTrainer(spark)
    >>> oof_preds = trainer.kfold_train(
    ...     model_name="LR",
    ...     model_params={"regParam": 0.1},
    ...     train_df=train,
    ...     feats=feats,
    ...     target="target",
    ...     n_splits=5,
    ...     random_state=42,
    ...     normalize=True,
    ... )
    >>> test_preds = trainer.predict(test)

    Parameters
    ----------
    spark : SparkSession
        The Spark session.
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        # trained models
        self.models: List[Any] = []
        # feature names
        self.feats: List[str] = []
        # fitted PipelineModels for each fold
        self.fold_piplines: List[PipelineModel] = []
        # model name
        self.model_name: Literal["LR", "Ridge", "Lasso", "RF", "LGB"] = None
        # whether to normalize features
        self.normalize: bool = False

    def __create_model(
        self,
        model_name: Literal["LR", "Ridge", "Lasso", "RF", "LGB"],
        model_params: Dict[str, Any],
        target: str,
    ) -> Any:
        """Create a Spark MLlib model based on the given name and parameters.

        Parameters
        ----------
        model_name : Literal["LR", "Ridge", "Lasso", "RF", "LGB"]
            The name of the model.
        model_params : Dict[str, Any]
            Model-specific parameters.
        target : str
            The name of the target column.

        Returns
        -------
        Any
            The created Spark MLlib model.
        """
        if self.normalize:
            featuresCol = "scaled_features"
        else:
            featuresCol = "features"
        if model_name == "LR":
            return LogisticRegression(
                featuresCol=featuresCol, labelCol=target, **model_params
            )
        elif model_name == "Ridge":
            return LinearRegression(
                featuresCol=featuresCol,
                labelCol=target,
                elasticNetParam=0,
                **model_params,
            )
        elif model_name == "Lasso":
            return LinearRegression(
                featuresCol=featuresCol,
                labelCol=target,
                elasticNetParam=1,
                **model_params,
            )
        elif model_name == "RF":
            return RandomForestClassifier(
                featuresCol=featuresCol, labelCol=target, **model_params
            )
        elif model_name == "LGB":
            return LightGBMClassifier(
                featuresCol=featuresCol,
                labelCol=target,
                **model_params,
            )
        else:
            raise NotImplementedError

    def __train(
        self,
        model_name: Literal["LR", "Ridge", "Lasso", "RF", "LGB"],
        model_params: Dict[str, Any],
        train: DataFrame,
        valid: DataFrame,
        feats: List[str],
        target: str = "target",
        normalize: bool = False,
    ) -> Tuple[Any, np.ndarray]:
        """Train model on train set and evaluate on valid set

        Parameters
        ----------
        model_name : Literal['LR','Ridge','Lasso','RF','LGB']
            model name, one of ['LR','Ridge','Lasso','RF','LGB']
        model_params : Dict[str, Any]
            model parameters
        train : pd.DataFrame
            train set
        valid : pd.DataFrame
            valid set
        feats : List[str]
            feature names
        target : str, optional
            target name, by default "target"
        normalize : bool, optional
            whether to normalize features, by default False

        Returns
        -------
        model : Any
            trained model
        valid_preds : np.ndarray
            valid predictions
        """
        self.model_name = model_name
        # * create fold pipeline
        assembler = VectorAssembler(inputCols=feats, outputCol="features")
        stages = [assembler]
        if normalize:
            scaler = StandardScaler(
                inputCol="features",
                outputCol="scaled_features",
                withStd=True,
                withMean=True,
            )
            stages.append(scaler)
        model = self.__create_model(model_name, model_params, target)
        stages.append(model)

        pipeline = Pipeline(stages=stages)
        pipeline_model = pipeline.fit(train)

        # * get valid predictions
        valid_pred = pipeline_model.transform(valid)
        if model_name in ["LR", "RF"]:
            extract_prob_udf = F.udf(lambda x: float(x[1]), DoubleType())
            valid_pred = valid_pred.withColumn(
                "positive_probability", extract_prob_udf(F.col("probability"))
            )
            valid_pred = valid_pred.select(
                "row_num", target, F.col("positive_probability").alias("prediction")
            ).toPandas()
        else:
            valid_pred = valid_pred.select("row_num", target, "prediction").toPandas()

        self.fold_piplines.append(pipeline_model)

        return pipeline_model, valid_pred

    @timer
    def kfold_train(
        self,
        model_name: Literal["LR", "Ridge", "Lasso", "RF", "LGB"],
        model_params: Dict[str, Any],
        train_df: DataFrame,
        feats: List[str],
        target: str = "target",
        n_splits: int = 5,
        random_state: int = 42,
        normalize: bool = False,
    ) -> pd.DataFrame:
        """Perform k-fold cross-validation training for the specified model.

        Parameters
        ----------
        model_name : Literal["LR", "Ridge", "Lasso", "RF", "LGB"]
            The name of the model to be trained.
        model_params : Dict[str, Any]
            Model-specific parameters.
        train_df : DataFrame
            The training DataFrame.
        feats : List[str]
            List of feature names.
        target : str, optional
            The name of the target column, by default "target".
        n_splits : int, optional
            Number of folds for cross-validation, by default 5.
        random_state : int, optional
            Random seed for reproducibility, by default 42.
        normalize : bool, optional
            Whether to normalize features using StandardScaler, by default False.

        Returns
        -------
        pd.DataFrame
            The out-of-fold predictions with ['customer_ID', 'prediction'] columns.
        """
        self.feats = feats.copy()
        self.normalize = normalize

        # * create a column to keep the original order of the rows
        train_df = train_df.withColumn(
            "row_num", F.monotonically_increasing_id()  # .cast("string")
        )
        row_nums = train_df.select("row_num").toPandas().values.flatten()

        # * create a dataframe to store oof predictions
        oof_preds = train_df.select("row_num", "customer_ID", target).toPandas()
        oof_preds["prediction"] = 0
        oof_preds = oof_preds.set_index("row_num")

        # * stratified k-fold training
        skf = StratifiedKFold(
            n_splits=n_splits, random_state=random_state, shuffle=True
        )
        for fold, (train_idx, valid_idx) in enumerate(
            skf.split(row_nums, oof_preds[target])
        ):
            train_fold = train_df.filter(
                train_df["row_num"].isin(row_nums[train_idx].tolist())
            )
            valid_fold = train_df.filter(
                train_df["row_num"].isin(row_nums[valid_idx].tolist())
            )

            pipeline_model, valid_preds = self.__train(
                model_name,
                model_params,
                train_fold,
                valid_fold,
                feats,
                target,
                normalize,
            )
            valid_prediction = valid_preds["prediction"].values
            valid_target = valid_preds["target"].values
            logger.info(
                f"Fold {fold + 1} Score: {amex_metric(valid_target, valid_prediction):.4f}"
            )
            oof_preds.loc[row_nums[valid_idx], "prediction"] = valid_prediction
            self.fold_piplines.append(pipeline_model)

        logger.info(
            f"Overall Score: {amex_metric(oof_preds[target].values, oof_preds['prediction'].values):.4f}"
        )
        oof_preds = oof_preds.reset_index(drop=True)
        return oof_preds

    @timer
    def predict(self, test_df: DataFrame) -> pd.DataFrame:
        """Make predictions on the test set.

        Parameters
        ----------
        test_df : DataFrame
            The test DataFrame.

        Returns
        -------
        pd.DataFrame
            The predictions on the test set with ['customer_ID', 'prediction'] columns.
        """
        test_df = test_df.withColumn("row_num", F.monotonically_increasing_id())
        test_preds = test_df.select("customer_ID", "row_num").toPandas()
        test_preds = test_preds.set_index("row_num")
        test_preds["prediction"] = 0

        for _, fold_pipeline in enumerate(self.fold_piplines):
            test_fold_pred = fold_pipeline.transform(test_df)

            if self.model_name in ["LR", "RF"]:
                extract_prob_udf = F.udf(lambda x: float(x[1]), DoubleType())
                test_fold_pred = test_fold_pred.withColumn(
                    "probability", extract_prob_udf(F.col("probability"))
                )
                test_fold_pred = test_fold_pred.select(
                    "row_num", F.col("probability").alias("prediction")
                ).toPandas()
            else:
                test_fold_pred = test_fold_pred.select(
                    "row_num", "prediction"
                ).toPandas()
            test_preds.loc[test_fold_pred["row_num"], "prediction"] += (
                test_fold_pred["prediction"].values
            ) / len(self.fold_piplines)

        test_preds = test_preds.reset_index(drop=True)
        return test_preds
