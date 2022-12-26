import wandb
import sklearn
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.kernel_ridge import KernelRidge
from utils import *

# Define sweep config
sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'val_mse'},
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


def main():
    run = wandb.init()
    base_dir = "/media/rrtammyfs/Users/doronser/statistical_learning/midterm_kaggle"
    trainval_set = pd.read_csv(f'{base_dir}/train.csv')
    test_set = pd.read_csv(f'{base_dir}/test.csv')

    # create train/val split
    train_set, val_set = train_val_split(trainval_set)

    # needed for preprocessing step
    country_continent_map = get_country_continent_map(trainval_set)

    # preprocess data: convert all features to numeric and drop Nans in label
    X_train, y_train = preprocess(train_set, conti_map=country_continent_map)
    X_val, y_val = preprocess(val_set, conti_map=country_continent_map)
    X_test, _ = preprocess(test_set, train=False, conti_map=country_continent_map)

    ################
    #   Pipeline   #
    ################

    # Normalize data to 0 mean and unit variance
    norm = preprocessing.StandardScaler()

    # complete missing values using nearest neighboring samples
    imputer = KNNImputer(n_neighbors=5, weights="uniform")

    # configure model to take parameters from sweep_config
    model = KernelRidge(alpha=wandb.config.alpha,
                        kernel=wandb.config.kernel,
                        gamma=wandb.config.gamma,
                        degree=wandb.config.degree,
                        coef0=wandb.config.coef0
                        )

    # build the pipeline normalize->impute->KernelRidge
    pipe = Pipeline([
        ('normalize', norm),
        ('impute', imputer),
        ('model', model)
    ])

    pipe.fit(X_train, y_train)
    score_train = pipe.score(X_train, y_train)
    mse_train = sklearn.metrics.mean_squared_error(y_train, pipe.predict(X_train))

    # calc train/val score and MSE
    score_val = pipe.score(X_val, y_val)
    mse_val = sklearn.metrics.mean_squared_error(y_val, pipe.predict(X_val))
    print(f"Train\tScore: {score_train}, MSE: {mse_train}")
    print(f"Val\tScore: {score_val}, MSE: {mse_val}")
    wandb.log(dict(train_score=score_train, train_mse=mse_train,
                   val_score=score_val, val_mse=mse_val))


if __name__ == "__main__":
    # Start sweep job.
    wandb.agent(sweep_id, function=main, count=100)
