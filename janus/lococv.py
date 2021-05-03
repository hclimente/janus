import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


def construct_profiles(df_dataset, features):

    """ Creates profile for given methodology

    Args:
        df_dataset(pd.DataFrame): the dataset
        features(list): list of features names
        method(string): the method

    Returns:
        pd.DataFrame: the phenotypic profiles for each drug

    """

    # extract unique wells
    wells = set(df_dataset.well)

    # a matrix of drugs vs. feature indices
    df_profiles = pd.DataFrame(index=wells, columns=features)

    for well in wells:

        # collect cells from current well
        df_drug = df_dataset[df_dataset['well'] == well]
        profile = list(df_drug[features].mean(axis=0))
        df_profiles.loc[well][features] = profile

    return df_profiles


def lococv(df_profiles, df_metadata, model=KNeighborsClassifier(n_neighbors=1)):

    """ Leave-one-compoud-out cross-validation

        Args:
            df_profile(pd.DataFrame): 
            df_metadata(pd.DataFrame):
        Returns:
            (np.ndarray): confusion matrix
    """

    df_profiles = df_profiles.div(np.sum(df_profiles, axis=1), axis=0)
    moas = df_metadata.moa.unique()
    confusion_matrix = np.zeros((moas.shape[0], moas.shape[0]))

    for well, df_holdout in df_profiles.iterrows():
        # hold out well
        holdout_well = df_holdout.name

        # training set (all other profiles)
        df_train = df_profiles[~(df_profiles.index == holdout_well)]

        labels = df_train.join(df_metadata).moa
        
        model.fit(df_train, labels)

        pred_moa = model.predict(df_holdout.values.reshape(1, -1))
        true_moa = df_metadata.loc[holdout_well].moa

        # record accuracy
        confusion_matrix[np.where(moas == true_moa)[0][0],
                         np.where(moas == pred_moa)[0][0]] += 1

    return confusion_matrix
