# Statistical Learning Kaggle 1 - Life Expectancy 
This repo includes a solution to the linear regression coding assignment for "Statistical Inference and Data Mining" masters level course in BGU.

The task is to predict Life Expectancy using Linear Regression models only.

Our solution ranked 8th in the final leaderboard (70 participants overall).

The Kaggle competition is available [here](https://www.kaggle.com/competitions/kaggle1lifeexpectancy).

## Solution Overview
The final pipeline of our solution is:
- Data Cleaning: convert all features to numeric values, drop samples with missing label, normalize data to zero mean and unit variance, impute Nan values using KNNImputer
- Feature Engineering: added continent_cde feature
- Model: Kernel Ridge Regression with a Laplacian kernel and $\alpha=0.03, \gamma=0.03$

## Hyperparameter Sweep
The hyper-parameters of the Kernel Ridge Regression model were chosen using Weights & Biases Sweep:
![img](imgs/wandb_sweep.png)

An interactive visualization of the sweep is available [here](https://wandb.ai/bio-vision-lab/statistical_learning_kaggle1/reports/Statistical-Learning-Kaggle-Midterm--VmlldzozMjE0MjYx?accessToken=dan7v3xdkone9odej7sq4i8ab50l81lkjwzsa8jye69rgqonizhr2lc5yx9kbmm4).

## Usage
- run main.py to train the final model
- run wandb_sweep.py to perform hyperparameter search over a pre-defined validation set.
- run wandb_sweep.py to perform hyperparameter search over a k-folds generated train/val split.

## Note
This repo does not include our EDA and experiments. It only shows the pipeline we used to generate our final model.
