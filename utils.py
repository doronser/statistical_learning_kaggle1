import numpy as np
import pandas as pd
import pycountry_convert as pc
from sklearn import preprocessing
from tqdm import tqdm


def preprocess(df, train=True, conti_map=None):
    # change string column to boolean
    df['is_developed'] = df['Status'] == 'Developed'
    df['is_developed'] = df['is_developed'].astype(int)

    # convert categorical column to numeric
    lbl_enc = preprocessing.LabelEncoder()
    lbl_enc.fit(df.Country.unique())
    df['Country_code'] = lbl_enc.transform(df.Country)

    if conti_map is not None:
        # create continent feature
        df['continent'] = df['Country'].apply(lambda x: conti_map[x])

        # convert categorical column to numeric
        lbl_enc = preprocessing.LabelEncoder()
        lbl_enc.fit(df.continent.unique())
        df['continent_code'] = lbl_enc.transform(df.continent)
        df['continent_code'].value_counts()
        df.drop(columns=['continent'], inplace=True)

    if train:
        # drop unlabeled data non-numeric columns
        df.dropna(subset='Life expectancy ', inplace=True)
        X = df.drop(columns=['Life expectancy ', 'life_expectancy_decade', 'Country', 'Status'])
        y = df['Life expectancy ']
    else:
        # drop non-numeric columns, produce y for API consistency
        X = df.drop(columns=['ID', 'Country', 'Status'])
        y = None

    return X, y


def preprocess_kfold(df, train=True, conti_map=None):
    # change string column to boolean
    df['is_developed'] = df['Status'] == 'Developed'
    df['is_developed'] = df['is_developed'].astype(int)

    # convert categorical column to numeric
    lbl_enc = preprocessing.LabelEncoder()
    lbl_enc.fit(df.Country.unique())
    df['Country_code'] = lbl_enc.transform(df.Country)

    if conti_map is not None:
        df['continent'] = df['Country'].apply(lambda x: conti_map[x])

        # convert categorical column to numeric
        lbl_enc = preprocessing.LabelEncoder()
        lbl_enc.fit(df.continent.unique())
        df['continent_code'] = lbl_enc.transform(df.continent)
        df['continent_code'].value_counts()
        df.drop(columns=['continent'], inplace=True)

    if train:
        # drop unlabeled data non-numeric columns
        df.dropna(subset='Life expectancy ', inplace=True)
        kfold_y = df.life_expectancy_decade
        X = df.drop(columns=['Life expectancy ', 'life_expectancy_decade', 'Country', 'Status'])
        y = df['Life expectancy '].reset_index(drop=True)
    else:
        # drop non-numeric columns, produce y for API consistency
        X = df.drop(columns=['ID', 'Country', 'Status'])
        y = None
        kfold_y = None

    return X, y, kfold_y


def train_val_split(trainval_set):
    bin_names = ["40-50", "50-60", "60-70", "70-80", "80-90"]
    trainval_set['life_expectancy_decade'] = pd.cut(trainval_set['Life expectancy '],
                                                    bins=range(40, 100, 10), labels=bin_names)
    # build a dictionary to map between age-decade and number of samples it should include in the validation set
    val_size = trainval_set.shape[0] * 0.1
    hist, bins = np.histogram(trainval_set['Life expectancy '].values, bins=range(40, 100, 10))  # generate histograms
    hist_normed = hist / hist.sum()  # normalize values
    num_samples_per_bin = (hist_normed * val_size).astype(int)  # get number of samples per bin
    samples_per_bin_dict = {k: v for k, v in zip(bin_names, num_samples_per_bin)}  # construct dict
    val_list = []
    for bin_name, num_samples in samples_per_bin_dict.items():
        samples = trainval_set.query(f'life_expectancy_decade=="{bin_name}"').sample(num_samples, random_state=42)
        val_list.append(samples)
    validation_set = pd.concat(val_list)
    training_set = trainval_set.drop(index=validation_set.index).reset_index(drop=True)
    return training_set, validation_set


def country_to_continent(country_name):
    country_alpha2 = pc.country_name_to_country_alpha2(country_name)
    country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
    country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
    return country_continent_name


def get_country_continent_map(df):
    countries = list(df.Country.unique())
    country_continent_map = dict()
    country_continent_map["Iran (Islamic Republic of)"] = 'Asia'
    country_continent_map["Republic of Korea"] = 'Asia'
    country_continent_map["The former Yugoslav republic of Macedonia"] = 'Europe'
    country_continent_map["Timor-Leste"] = 'Asia'
    country_continent_map["Micronesia (Federated States of)"] = 'Oceania'
    country_continent_map["Venezuela (Bolivarian Republic of)"] = 'South America'
    country_continent_map["Bolivia (Plurinational State of)"] = 'South America'
    for country in tqdm(countries):
        try:
            country_continent_map[country] = country_to_continent(country)
        except KeyError:
            pass
    return country_continent_map


