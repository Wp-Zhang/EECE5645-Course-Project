from typing import Dict, List, Tuple, Any, Literal
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator


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
        self.model: CrossValidatorModel = None
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

        # * create pipeline
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

        # * K-fold cross-validation training
        crossval = CrossValidator(
            estimator=pipeline,
            evaluator=BinaryClassificationEvaluator(labelCol=target),
            estimatorParamMaps=ParamGridBuilder().build(),
            numFolds=n_splits,
            seed=random_state,
        )
        cv_model = crossval.fit(train_df)

        self.model = cv_model
        return

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
        test_pred = self.model.transform(test_df)

        if self.model_name in ["LR", "RF"]:
            extract_prob_udf = F.udf(lambda x: float(x[1]), DoubleType())
            test_pred = test_pred.withColumn(
                "probability", extract_prob_udf(F.col("probability"))
            )
            test_pred = test_pred.select(
                "customer_ID", F.col("probability").alias("prediction")
            ).toPandas()
        else:
            test_pred = test_pred.select("customer_ID", "prediction").toPandas()

        return test_pred
