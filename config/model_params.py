from scipy.stats import randint,uniform


LIGHTGBM_PARAMS = {
    'n_estimators': randint(100, 500),
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': randint(5, 50),
    'num_leaves': randint(20, 100),
    'boosting_type': ['gbdt', 'dart', 'goss']
}

RANDOM_SEARCH_PARAMS = {
    'n_iter': 4,
    'cv': 5,
    'verbose': 2,
    'random_state': 42,
    'n_jobs': -1,
    'scoring': 'accuracy'
}