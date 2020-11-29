"""
An extension to sklearn.preprocessing.LabelEncoder
that safely handles an unknown word represented by ${UNK}
in the encoded classes.

Attributes:
    LabelEncoderExt (class):

TODO:
    Add ${top_n} encoding.
"""

import numpy as np
import sklearn.preprocessing

log = logging.getLogger(name=__name__)
log.setLevel(logging.INFO)
logging.captureWarnings(True)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
log.addHandler(stream_handler)

UNK = "UNK"
RANK = "rank"
NUMBER = "number"
FREQUENCY = "frequency"


class LabelEncoderExt(sklearn.preprocessing.LabelEncoder):
    """
    """

    def __init__(self, top_n=None, count_thresh=None, freq_thresh=None):
        """
        """
        super(LabelEncoderExt, self).__init__()
        if top_n is not None:
            self.criterion = RANK
            try:
                self.criterion_val = int(top_n)
            except ValueError as e:
                log.error(e)
        elif count_thresh is not None:
            self.criterion = NUMBER
            try:
                self.criterion_val = int(count_thresh)
            except ValueError as e:
                log.error(e)
        elif freq_thresh is not None:
            self.criterion = FREQUENCY
            try:
                self.criterion_val = float(freq_thresh)
            except ValueError as e:
                log.error(e)
        else:
            self.criterion = ""
            self.criterion_val = None


    def fit(self, x):
        """
        """
        try:
            x = np.array(x)
        except ValueError as e:
            log.error(e)
        assert (len(x.shape)==1), "Require 1D array"

        unique_elem, elem_locs, elem_counts = (
            np.unique(
                x,
                return_inverse=True,
                return_counts=True,
            )
        )
        y = np.full(x.shape, np.nan)

        if self.criterion == RANK:
            a = np.argpartition(elem_counts, self.criterion_val)
            for t in a:
                y[elem_locs[t]] = unique_elem[t]
        elif self.criterion == NUMBER:
            for i, t in np.ndenumerate(elem_counts):
                if t >= self.criterion_val:
                    y[elem_locs[i]] = unique_elem[i]
        elif self.criterion == FREQUENCY:
            for i, t in np.ndenumerate(elem_counts):
                if t/x.shape[0] >= self.criterion_val:
                    y[elem_locs[i]] = unique_elem[i]
        else:
            y = x
        y[np.isna(y)] = UNK

        y = np.concatenate((y, np.array([UNK])))
        super(LabelEncoderExt, self).fit(x)

        return 0


    def transform(self, x):
        """
        """
        x[~np.isin(x, self.classes_)] = UNK

        return super(LabelEncoderExt, self).transform(x)


    def fit_transform(self, x):
        """
        """
        self.fit(x)

        return self.transform(x)


class OrdinalEncoderExt(sklearn.preprocessing.OrdinalEncoder):
    """
    """

    def __init__(self,
                 top_n=None, 
                 count_thresh=None, 
                 freq_thresh=None, 
                 categories="auto", 
                 **kwargs,
    ):
        """
        """
        super(OrdinalEncoderExt, self).__init__(
            categories=categories,
            **kwargs
        )
        if top_n is not None:
            self.criterion = RANK
            try:
                self.criterion_val = int(top_n)
            except ValueError as e:
                log.error(e)
        elif count_thresh is not None:
            self.criterion = NUMBER
            try:
                self.criterion_val = int(count_thresh)
            except ValueError as e:
                log.error(e)
        elif freq_thresh is not None:
            self.criterion = FREQUENCY
            try:
                self.criterion_val = float(freq_thresh)
            except ValueError as e:
                log.error(e)
        else:
            self.criterion = ""
            self.criterion_val = None


    def fit(self, X):
        """
        """
        try:
            X = np.array(X)
        except ValueError as e:
            log.error(e)
        assert (len(X.shape)==2), "Require 2D array"
        
        X = X.astype(str)
        
        Y = np.full(X.shape, "", dtype="U20")
        for j in range(X.shape[1]):
            unique_elem, elem_locs, elem_counts = (
                np.unique(
                    X[:, j],
                    return_inverse=True,
                    return_counts=True,
                )
            )

            if self.criterion == RANK:
                a = np.argpartition(elem_counts, self.criterion_val)
                for t in a:
                    Y[elem_locs[t], j] = unique_elem[t]
            elif self.criterion == NUMBER:
                for i, t in np.ndenumerate(elem_counts):
                    if t >= self.criterion_val:
                        Y[elem_locs[i], j] = unique_elem[i]
            elif self.criterion == FREQUENCY:
                for i, t in np.ndenumerate(elem_counts):
                    if t/x.shape[0] >= self.criterion_val:
                        Y[elem_locs[i], j] = unique_elem[i]
            else:
                Y[:, j] = X[:, j]
            Y[np.where(Y[:, j]==''), j] = UNK

        tmp = np.full(X.shape[1], UNK).reshape((1, -1))
        Y = np.append(Y, tmp, axis=0)

        super(OrdinalEncoderExt, self).fit(Y)
        
        return 0


    def transform(self, X):
        """
        """
        X = X.astype(str)
        for i in range(X.shape[1]):
            X[~np.isin(X[:, i], self.categories_[i]), i] = UNK
            
        return super(OrdinalEncoderExt, self).transform(X)


    def fit_transform(self, X):
        """
        """
        self.fit(X)

        return self.transform(X)
