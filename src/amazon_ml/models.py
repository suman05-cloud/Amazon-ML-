"""Load versioned weighted tree ensembles without giving each worker a GPU."""
from pathlib import Path

import lightgbm as lgb


class WeightedModel:
    def __init__(self, path, kind, names, scale):
        self.kind, self.names, self.scale = kind, names, scale
        if kind == "lightgbm":
            self.model = lgb.Booster(model_file=str(path))
        elif kind == "xgboost":
            import xgboost as xgb
            self.model = xgb.Booster()
            self.model.load_model(path)
            self.model.set_param({"device": "cpu", "nthread": 1})
        else:
            raise ValueError(f"Unsupported tree model: {kind}")

    def predict(self, matrix, num_threads=1):
        if self.kind == "lightgbm":
            value = self.model.predict(matrix, num_threads=num_threads)
        else:
            import xgboost as xgb
            value = self.model.predict(xgb.DMatrix(matrix, feature_names=self.names, nthread=num_threads))
        return value if self.scale == 1 else value*self.scale


def load_models(folder, manifest):
    files = manifest["model_files"]
    kinds = manifest.get("model_types", ["lightgbm"]*len(files))
    weights = manifest.get("model_weights", [1/len(files)]*len(files))
    if len(files) != len(kinds) or len(files) != len(weights) or abs(sum(weights)-1) > 1e-8 or min(weights) < 0:
        raise ValueError("Invalid ensemble weights")
    return [WeightedModel(Path(folder)/f, kind, manifest["feature_names"], weight*len(files))
            for f, kind, weight in zip(files, kinds, weights)]
