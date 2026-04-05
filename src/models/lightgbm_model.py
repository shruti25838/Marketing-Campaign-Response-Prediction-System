from .base_model import BaseModel


class LightGBMModel(BaseModel):
    def build(self):
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError(
                "lightgbm is not installed. Install it with `pip install lightgbm`."
            ) from exc

        default_params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary",
            "metric": "binary_logloss",
            "n_jobs": -1,
            "verbose": -1,
        }
        default_params.update(self.model_params)
        return LGBMClassifier(**default_params)
