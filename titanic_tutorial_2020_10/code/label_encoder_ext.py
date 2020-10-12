import numpy as np
import sklearn.preprocessing as preprocessing

class LabelEncoderExt(preprocessing.LabelEncoder):

    UNK = "UNK"

    def __init__(self):

        super().__init__()

    def fit(self, y):

        if not isinstance(y, np.ndarray):
            y = np.array(y)
        assert (len(y.shape) == 1), "Require 1D array"
        y = np.concatenate((y, np.array([self.UNK])))
        super().fit(y)

    def transform(self, y):

        y[~np.isin(y, self.classes_, assume_unique=True)] = self.UNK
        return super().transform(y)

    def fit_transform(self, y):

        self.fit(y)
        return self.transform(y)
