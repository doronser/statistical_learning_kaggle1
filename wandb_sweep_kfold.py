import os
import sys
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
import sklearn
from sklearn import preprocessing
from sklearn.kernel_ridge import KernelRidge
from sklearn.impute import KNNImputer

import pycountry_convert as pc

from utils import preprocess_kfold, get_country_continent_map

user = os.getlogin()
sys.path.append(f"/home/{user}/workspace/")

# Define sweep config
sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'val_mean_plus_std'},
    'parameters':
    {
        'kernel': {'values': ['poly', 'rbf', 'laplacian']},
        'alpha': {'max': 0.5, 'min': 0.01},
        'gamma': {'max': 0.5, 'min': 0.01},
        'degree': {'values': [2, 3, 4]},
        'coef0': {'values': [0.1, 0.5, 1]}
     }
}

# Initialize sweep by passing in config. (Optional) Provide a name of the project.
sweep_id = wandb.sweep(sweep=sweep_configuration, project='statistical_learning_kaggle1')


def country_to_continent(country_name):
    country_alpha2 = pc.country_name_to_country_alpha2(country_name)
    country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
    country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
    return country_continent_name


def main():
    run = wandb.init()
    base_dir = "/media/rrtammyfs/Users/doronser/statistical_learning/midterm_kaggle"
    trainval_set = pd.read_csv(f'{base_dir}/train.csv')

    # preprocess data
    X_trainval, y_trainval, y_kfold = preprocess_kfold(trainval_set, conti_map=get_country_continent_map(trainval_set))

    # normalize data to 0 mean and unit variance
    norm = preprocessing.StandardScaler()
    X_trainval = norm.fit_transform(X_trainval)

    # complete missing values using nearest neighboring samples
    imputer = KNNImputer(n_neighbors=5, weights="uniform")
    X_trainval = imputer.fit_transform(X_trainval)

    y_kfold[1860] = '40-50'  # MANUAL OVERRIDE of age=39 to bin 40-50
    lbl_enc = preprocessing.LabelEncoder()
    lbl_enc.fit(y_kfold.unique())
    y = lbl_enc.transform(y_kfold)

    skf = sklearn.model_selection.StratifiedKFold(n_splits=3)
    split = skf.split(X_trainval, y)
    train_metrics = dict(mse=[], score=[])
    val_metrics = dict(mse=[], score=[])
    for fold_idx, (train_indices, val_indices) in enumerate(split):
        X_train = X_trainval[train_indices, :]
        y_train = y_trainval[train_indices]
        X_val = X_trainval[val_indices, :]
        y_val = y_trainval[val_indices]

        model = KernelRidge(alpha=wandb.config.alpha,
                            kernel=wandb.config.kernel,
                            gamma=wandb.config.gamma,
                            degree=wandb.config.degree,
                            coef0=wandb.config.coef0
                            , kernel_params=None)
        model.fit(X_train, y_train)

        score_train = model.score(X_train, y_train)
        mse_train = sklearn.metrics.mean_squared_error(y_train, model.predict(X_train))
        score_val = model.score(X_val, y_val)
        mse_val = sklearn.metrics.mean_squared_error(y_val, model.predict(X_val))

        train_metrics['score'].append(score_train)
        train_metrics['mse'].append(mse_train)
        val_metrics['score'].append(score_val)
        val_metrics['mse'].append(mse_val)

    # calc train/val score and MSE
    mean_train_score = np.mean(train_metrics['score'])
    mean_train_mse = np.mean(train_metrics['mse'])
    mean_val_score = np.mean(val_metrics['score'])
    mean_val_mse = np.mean(val_metrics['mse'])
    std_val_mse = np.std(val_metrics['mse'])

    wandb.log(dict(train_score=mean_train_score, train_mse=mean_train_mse,
                   val_score=mean_val_score, val_mse=mean_val_mse,
                   val_mean_plus_std=mean_val_mse + std_val_mse))


if __name__ == "__main__":
    # Start sweep job.
    wandb.agent(sweep_id, function=main, count=100)
