import sklearn
from sklearn.pipeline import Pipeline
from sklearn.kernel_ridge import KernelRidge
from sklearn.impute import KNNImputer
from utils import *


if __name__ == "__main__":
    # load data
    trainval_set = pd.read_csv('train.csv')
    test_set = pd.read_csv('test.csv')

    # create train/val split
    train_set, val_set = train_val_split(trainval_set)

    # preprocess data: convert all features to numeric and drop Nans in label
    country_continent_map = get_country_continent_map(trainval_set)
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
    model = KernelRidge(alpha=0.03, kernel='laplacian', gamma=0.03, degree=2, coef0=1, kernel_params=None)

    pipe = Pipeline([
        ('normalize', norm),
        ('impute', imputer),
        ('model', model)
    ])

    pipe.fit(X_train, y_train)
    score_train = pipe.score(X_train, y_train)
    mse_train = sklearn.metrics.mean_squared_error(y_train, pipe.predict(X_train))

    score_val = pipe.score(X_val, y_val)
    mse_val = sklearn.metrics.mean_squared_error(y_val, pipe.predict(X_val).round(decimals=1))
    print(f"Train\tScore: {score_train}, MSE: {mse_train}")
    print(f"Val\tScore: {score_val}, MSE: {mse_val}")

    # Train from scratch on entire dataset
    X_full = pd.concat([X_train, X_val])
    y_full = pd.concat([y_train, y_val])

    norm = preprocessing.StandardScaler()
    imputer = KNNImputer(n_neighbors=5, weights="uniform")
    model = KernelRidge(alpha=0.03, kernel='laplacian', gamma=0.03, degree=2, coef0=1, kernel_params=None)
    pipe = Pipeline([('normalize', norm), ('impute', imputer), ('model', model)])
    pipe.fit(X_full, y_full)
    score = pipe.score(X_full, y_full)
    mse = sklearn.metrics.mean_squared_error(y_full, pipe.predict(X_full))
    print(f"Full Train\tScore: {score}, MSE: {mse}")
    submission = pd.DataFrame({'ID': test_set['ID'], 'Life Expectancy': pipe.predict(X_test)})
    submission.to_csv('ShakeDoron_final_submission.csv', index=False)
