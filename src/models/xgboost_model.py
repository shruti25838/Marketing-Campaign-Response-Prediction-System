from .base_model import BaseModel


class XGBoostModel(BaseModel):
    def build(self):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. Install it with `pip install xgboost`."
            ) from exc

        default_params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_jobs": -1,
        }
        default_params.update(self.model_params)
        return XGBClassifier(**default_params)
