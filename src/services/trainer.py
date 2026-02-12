from typing import Tuple

import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.models.base_model import BaseModel


def build_pipeline(model: BaseModel, preprocessor, use_smote: bool = True) -> ImbPipeline:
    """Return an unfitted pipeline (preprocess + optional SMOTE + model)."""
    steps = [("preprocess", preprocessor)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=42)))
    steps.append(("model", model.build()))
    return ImbPipeline(steps=steps)


def train_model(
    model: BaseModel,
    X_train,
    y_train,
    preprocessor,
    use_smote: bool = True,
) -> Tuple[ImbPipeline, np.ndarray]:
    pipeline = build_pipeline(model, preprocessor, use_smote)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_train)
    return pipeline, y_pred
