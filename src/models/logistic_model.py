from sklearn.linear_model import LogisticRegression

from .base_model import BaseModel


class LogisticModel(BaseModel):
    def build(self):
        default_params = {
            "max_iter": 1000,
            "class_weight": "balanced",
            "solver": "lbfgs",
        }
        default_params.update(self.model_params)
        return LogisticRegression(**default_params)
