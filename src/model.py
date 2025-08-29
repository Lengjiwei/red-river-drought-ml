# model.py — 
from __future__ import annotations
from xgboost import XGBRegressor
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

%%time
# Choose hyperparameter domain to search over
space = {
        'max_depth':hp.choice('max_depth', np.arange(1, 16, 2, dtype=int)),
        'colsample_bytree':hp.quniform('colsample_bytree', 0.5, 1.0, 0.05),
        'min_child_weight':hp.choice('min_child_weight', np.arange(1, 12, 1, dtype=int)),
        'subsample':        hp.quniform('subsample', 0.3, 1.0, 0.05),
        'learning_rate':    hp.choice('learning_rate',    np.arange(0.05, 1, 0.05)),
        'gamma': hp.quniform('gamma', 0.1, 1, 0.05),
        'objective':'reg:squarederror',
        'eval_metric': 'r2_score',
    }
def score(params):
    #Cross-validation
    d_train = xgboost.DMatrix(X_coords,y) 
    cv_results = xgboost.cv(params, d_train, nfold = 5, num_boost_round=50,
                        early_stopping_rounds = 10, metrics = 'rmse', seed = 0)
    loss = min(cv_results['test-rmse-mean'])
    return loss

def optimize(trials, space):
    best = fmin(score, space, algo=tpe.suggest, max_evals=200,
                trials=trials)#Add seed to fmin function
    return best    

    trials = Trials()
    best_params = optimize(trials, space)
    # Return the best parameters
    best_params = space_eval(space, best_params)
    best_params

xgb_model = XGBRegressor(colsample_bytree=0.95,max_depth=15,gamma=0.1,learning_rate= 0.2, min_child_weight=3,subsample=0.9,random_state=1)

xgb_model.fit(X_train.values, y_train)



