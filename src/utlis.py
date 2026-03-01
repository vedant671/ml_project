import os
import sys

import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, X_test, y_train, y_test, models, param):
    try:
        if not isinstance(models, dict):
            raise ValueError("'models' must be a dictionary of model_name -> estimator")
        if not isinstance(param, dict):
            raise ValueError("'param' must be a dictionary of model_name -> param_grid")

        report = {}

        for model_name, model in models.items():
            try:
                model_param_grid = param.get(model_name, {})

                if model_param_grid:
                    gs = GridSearchCV(model, model_param_grid, cv=3, scoring='r2')
                    gs.fit(X_train, y_train)
                    best_model = gs.best_estimator_
                else:
                    best_model = model
                    best_model.fit(X_train, y_train)

                y_test_pred = best_model.predict(X_test)
                test_model_score = r2_score(y_test, y_test_pred)
                report[model_name] = test_model_score

            except Exception as model_error:
                raise ValueError(f"Model '{model_name}' failed during training/evaluation: {model_error}")

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)

