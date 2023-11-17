import numpy as np
import pandas as pd

from typing import Dict, List, Tuple, Any, Literal
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso, Ridge
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import QuantileTransformer

from .metrics import amex_metric


class LocalTrainer:
    """Local model trainer

    Examples
    --------


    """

    def __init__(self):
        self.models: List[Any] = []  # trained models
        self.feats: List[str] = []  # feature names
        self.normalize: bool = False  # whether to normlize features
        self.normalizers: List[QuantileTransformer] = []  # normalizer of each fold

    def __normlize(
        self, train: pd.DataFrame, valid: pd.DataFrame, feats: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Normlize features

        Parameters
        ----------
        train : pd.DataFrame
            train set
        valid : pd.DataFrame
            valid set
        feats : List[str]
            feature names

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            normlized train and valid set
        """
        normalizer = QuantileTransformer(output_distribution="normal")
        normalizer.fit(train[feats])
        normalized_train = normalizer.transform(train[feats])
        normalized_valid = normalizer.transform(valid[feats])
        return normalized_train, normalized_valid, normalizer

    def __train(
        self,
        model_name: Literal["LR", "Ridge", "Lasso", "LGB"],
        model_params: Dict[str, Any],
        train: pd.DataFrame,
        valid: pd.DataFrame,
        feats: List[str],
        target: str = "target",
        normlize: bool = False,
    ) -> Tuple[Any, np.ndarray]:
        """Train model on train set and evaluate on valid set

        Parameters
        ----------
        model_name : Literal['LR','Ridge','Lasso','LGB']
            model name, one of ['LR','Ridge','Lasso','LGB']
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
        normlize : bool, optional
            whether to normlize features, by default False

        Returns
        -------
        model : Any
            trained model
        valid_preds : np.ndarray
            valid predictions
        """
        if normlize:
            train_x, valid_x, normalizer = self.__normlize(train, valid, feats)
            self.normalizers.append(normalizer)
        else:
            train_x, valid_x = train[feats].values, valid[feats].values

        if model_name == "LR":
            model = LogisticRegression(**model_params)
        elif model_name == "Ridge":
            model = Ridge(**model_params)
        elif model_name == "Lasso":
            model = Lasso(**model_params)
        elif model_name == "LGB":
            model = LGBMClassifier(**model_params)
        else:
            raise NotImplementedError

        if model_name == "LGB":
            model.fit(
                train_x,
                train[target].values,
                eval_set=[(valid_x, valid[target].values)],
                early_stopping_rounds=100,
                verbose=False,
            )
        else:
            model.fit(train_x, train[target].values)

        if model_name == "LGB":
            valid_preds = model.predict_proba(valid_x)[:, 1]
        else:
            valid_preds = model.predict(valid_x)

        return model, valid_preds

    def kfold_train(
        self,
        model_name: Literal["LR", "Ridge", "Lasso", "LGB"],
        model_params: Dict[str, Any],
        train: pd.DataFrame,
        feats: List[str],
        target: str = "target",
        n_splits: int = 5,
        random_state: int = 42,
        normlize: bool = False,
    ) -> np.ndarray:
        """K-fold train model

        Parameters
        ----------
        model_name : Literal['LR','Ridge','Lasso','LGB']
            model name, one of ['LR','Ridge','Lasso','LGB']
        model_params : Dict[str, Any]
            model parameters
        train : pd.DataFrame
            train set
        feats : List[str]
            feature names
        target : str, optional
            target name, by default "target"
        n_splits : int, optional
            number of folds, by default 5
        random_state : int, optional
            random state, by default 42
        normlize : bool, optional
            whether to normlize features, by default False

        Returns
        -------
        oof_preds : np.ndarray
            out-of-fold predictions
        """
        self.normalize = normlize
        self.feats = feats.copy()

        skf = StratifiedKFold(
            n_splits=n_splits, random_state=random_state, shuffle=True
        )
        oof_preds = np.zeros(train.shape[0])
        models = []
        for fold, (trn_idx, val_idx) in enumerate(skf.split(train, train[target])):
            print(f"Fold {fold + 1}")
            train_df = train.iloc[trn_idx]
            valid_df = train.iloc[val_idx]
            model, valid_preds = self.__train(
                model_name, model_params, train_df, valid_df, feats, target, normlize
            )
            oof_preds[val_idx] = valid_preds
            models.append(model)
            print(
                f"Fold {fold + 1} Score: {amex_metric(valid_df[target].values, valid_preds):.4f}"
            )
            print()
        print(f"Overall Score: {amex_metric(train[target].values, oof_preds):.4f}")
        return oof_preds

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        """Predict on test set

        Parameters
        ----------
        test : pd.DataFrame
            test set

        Returns
        -------
        np.ndarray
            test predictions
        """
        test_preds = np.zeros(test.shape[0])
        for fold, model in enumerate(self.models):
            if self.normalize:
                test_x = self.normalizers[fold].transform(test[self.feats])
            else:
                test_x = test[self.feats].values

            if isinstance(model, LGBMClassifier):
                test_preds += model.predict_proba(test_x)[:, 1] / len(self.models)
            else:
                test_preds += model.predict(test_x) / len(self.models)
        return test_preds
