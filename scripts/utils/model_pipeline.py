import time
from typing import Any, Callable, Optional
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from skopt import BayesSearchCV 
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

@staticmethod
def timed(func: Callable[[Any], Any]):
    """Simple decorator that print wall time for a given function."""

    def _inner(*args: Optional[Any], **kwargs: Optional[Any]) -> Any:
        print(f"Starting process '{func}'...")
        started = time.time()
        response = func(*args, **kwargs)
        ended = time.time()
        print(f"Finished process '{func}' in {ended - started:.2f}s")
        return response

    return _inner


@timed
def run_model(X, y, modelname, ml_model, params= {}, over = 0.2, under = 0.5, search_selection = 'grid', k_folds = 3, scoring_metric = 'f1', verbosity = 1, njobs = -1):
    """A pipeline that will impute the mode for missing categorical variables, median for missing numerical variables
        One hot encode categorical features and Standardise numerical features. 
        The pipeline will carry out over sampling using SMOTE and random undersampling.
        The pipeline will fit while carrying out cross validation to ensure no data-leakage 
        
        Returns a fit pipeline object.

    Args:
        X (pd.DataFrame): dataframe to be run through the pipeline
        y (np.Series): target variable
        modelname (str): string name for the model, needs to match the prefix for the hyper params 
                        passed through the params variable
        model (model): ML model to be fit
        params (dict, optional): Hyperparameters for tuning. Defaults to {}.
        verbosity (int, optional): Verbosity for the process. Defaults to 1.

    Returns:
        pipeline: fit pipeline containing the trained best model and params
    """

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('one_hot', OneHotEncoder(handle_unknown='ignore')) 
    ])
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('std_scaler', StandardScaler())
    ])
    pipe_clf = Pipeline([
    (
        "FeatureEngineering",
        ColumnTransformer(
            [
            ("num", num_pipeline, list(X.select_dtypes(exclude='object').columns)),
            ("cat", cat_pipeline, list(X.select_dtypes(include='object').columns)),
            ],
        
        )
    ),
    ("SMOTE", SMOTE()),
    ("UNDER", RandomUnderSampler()),
    (modelname, ml_model)
    ])
    param_grid = {
                "SMOTE__sampling_strategy": [over],
                "UNDER__sampling_strategy": [under],
                }
    param_grid.update(params)

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Run a bayesian optimisation over the hyperparameters with the pipeline to tune them
    # Use scoring parameter to maximize the f1 score
    if search_selection == 'bayes':
        gs_pipeline = BayesSearchCV(cv= skf, estimator= pipe_clf, search_spaces=param_grid, verbose=verbosity, scoring= scoring_metric, n_jobs= njobs)
    elif search_selection == 'grid':
        gs_pipeline = GridSearchCV(cv= skf, estimator= pipe_clf, param_grid=param_grid, verbose=verbosity, scoring= scoring_metric, n_jobs= njobs)
    elif search_selection == 'random':
        gs_pipeline = RandomizedSearchCV(estimator = pipe_clf, param_distributions=param_grid, cv = skf, verbose=verbosity, scoring=scoring_metric, n_iter=20, n_jobs=njobs)

    gs_pipeline.fit(X, y)

    return gs_pipeline.best_estimator_
    
    

    

    #model_outputs[modelname]["best_model"] = gs_pipeline.best_estimator_
    #model_outputs[modelname]["best_model_params"] = gs_pipeline.best_params_

def get_scoring_metrics(model, X, y):
    """A function that will provide a range of scoring metrics for a classification model

    Args:
        model (model): A pre-fit model
        X (pd.DataFrame): a df of the correct shape and features for the model to make predictions
        y (np.Series): the true values of the target variable

    Returns:
        dict: dictionary containing Accuracy, Recall, Precision, F1 scores and the predictions
    """

    model_outputs = {}
    # Make predictions on our validation set and see recall score of best model
    model_outputs["y_validation_preds"] = model.predict(X)
    model_outputs["accuracy"] =  accuracy_score(y, model_outputs["y_validation_preds"])
    model_outputs["recall"] =  recall_score(y, model_outputs["y_validation_preds"])
    model_outputs["precision"] =  precision_score(y, model_outputs["y_validation_preds"])
    model_outputs["f1"] = f1_score(y, model_outputs["y_validation_preds"]) 

    # cm = confusion_matrix(y_test, model_outputs[modelname]["y_validation_preds"], labels=model_outputs[modelname]["best_model"].classes_)
    #model_outputs[modelname]["confusion_matrix"] = ConfusionMatrixDisplay(confusion_matrix=cm,
    #                                                            display_labels=model_outputs[modelname]["best_model"].classes_)
    
    return model_outputs