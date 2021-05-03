import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


class LocoCV:

    def __init__(self, dataset, net):

        self.dataset = dataset
        self.net = net

    def construct_profiles(self):

        """ Creates profile for given methodology

        Args:

        Returns:
            pd.DataFrame: the phenotypic profiles for each drug

        """

        # compute embeddings
        df_embeddings = pd.DataFrame()
        features = range(self.net.fc[-1].out_features)

        for crop, meta in self.dataset.dataset_1 + self.dataset.dataset_2:
            embedding = self.net.embedding(crop[None])
            df_emb = pd.DataFrame(embedding.detach().numpy())
            df_emb['well'] = meta['well']
            df_emb['moa'] = meta['moa']
            df_embeddings = pd.concat([df_embeddings, df_emb])

        # extract unique wells
        wells = set(df_embeddings.well)

        # a matrix of drugs vs. feature indices
        df_profiles = pd.DataFrame(index=wells, columns=features)

        for well in wells:

            # collect cells from current well
            df_drug = df_embeddings[df_embeddings['well'] == well]
            profile = list(df_drug[features].mean(axis=0))
            df_profiles.loc[well][features] = profile

        return df_profiles

    def lococv(self, df_profiles, model=KNeighborsClassifier(n_neighbors=1)):

        """ Leave-one-compound-out cross-validation

            Args:
                df_profiles(pd.DataFrame):
                model:
            Returns:
                (np.ndarray): confusion matrix
        """

        metadata = self.dataset.metadata

        df_profiles = df_profiles.div(np.sum(df_profiles, axis=1), axis=0)
        moas = metadata.moa.unique()
        confusion_matrix = np.zeros((moas.shape[0], moas.shape[0]))

        for well, df_holdout in df_profiles.iterrows():
            # hold out well
            holdout_well = df_holdout.name

            # training set (all other profiles)
            df_train = df_profiles[~(df_profiles.index == holdout_well)]

            labels = df_train.join(metadata).moa

            model.fit(df_train, labels)

            pred_moa = model.predict(df_holdout.values.reshape(1, -1))
            true_moa = metadata.loc[holdout_well].moa

            # record accuracy
            confusion_matrix[np.where(moas == true_moa)[0][0],
                             np.where(moas == pred_moa)[0][0]] += 1

        return confusion_matrix
