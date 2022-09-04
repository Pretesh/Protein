import dataProcessing
import xgboost as xgb
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
from ray import air, tune
from ray.air import session
from ray.tune.integration.xgboost import (
    TuneReportCheckpointCallback,
    TuneReportCallback,
)


def trainXGBoost(config):
    data, labels = dataProcessing.preProcessing()
    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.20)
    train_set = xgb.DMatrix(train_x, label=train_y)
    test_set = xgb.DMatrix(test_x, label=test_y)
    evals_result = {}
    # could try xgbregressor but train works equally well apparently
    # model = xgb.XGBRegressor()
    # model.fit()
    xgb.train(
         config,
         train_set,
         evals=[(test_set, "eval")],
         evals_result=evals_result,
         verbose_eval=False
         callbacks=[TuneReportCheckpointCallback(filename="model.xgb")])
    # Return prediction accuracy
    accuracy = 1. - evals_result["eval"]["error"][-1]
    session.report({"mean_accuracy": accuracy, "done": True})

def xgboost_parameterSearch():
    # Don't believe MSE is an option for the config actually, will probably show error
    config = {
        "objective": "reg:squarederror",
        "eval_metric": ["mae", "rmse"],
        "max_depth": tune.randint(1, 9),
        "min_child_weight": tune.choice([1, 2, 3, 4, 5, 6]),
        "subsample": tune.uniform(0.5, 1.0),
        "eta": tune.loguniform(1e-4, 1e-1),
        "gamma": tune.uniform(0, 1),
        "colsample_bytree": tune.uniform(0.5, 1.0),

    }
    tuner = tune.Tuner(
        trainXGBoost,
        tune_config=tune.TuneConfig(
            num_samples=10,
        ),
        param_space=config,
    )
    results = tuner.fit()

    return results


# Should get best model saved by tune earlier? Need to change error however
def get_best_model_checkpoint(best_result: "ray.air.Result"):
    best_bst = xgb.Booster()
    with best_result.checkpoint.as_directory() as checkpoint_dir:
        best_bst.load_model(os.path.join(checkpoint_dir, "model.xgb"))
    accuracy = 1.0 - best_result.metrics["test-error"]
    print(f"Best model parameters: {best_result.config}")
    print(f"Best model total accuracy: {accuracy:.4f}")
    return best_bst