import abc
import numpy as np
from collections import namedtuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import _safe_indexing


Lineage = namedtuple("Lineage", ["name", "transformer", "parents"])


class FeatOp(BaseEstimator, TransformerMixin):

    _estimator_type = "FeatureOperation"

    def __init__(self, src_cols, target_cols_=None, func_=None):
        self.src_cols = src_cols
        # operate on target columns and its result
        # [Lineage...]
        self.target_cols_ = target_cols_ or []
        # operator function
        self.func_ = func_
    
    @property
    def target_cols(self):
        return [lineage for lineage in self.target_cols_ if lineage.transformer == self]


class SamplerMixin(BaseEstimator, metaclass=abc.ABCMeta):
    """Mixin class for samplers with abstract method
    """
    _estimator_type = "sampler"

    def fit(self, X, y):
        # ensure y is one-hot label
        X, y, _ = self._check_X_y(X, y)
        self.sampling_info_ = {}
        for i in range(len(y[0])):
            indices = np.where(y[:, i] == 1)
            self.sampling_info_[i] = (len(indices), indices)
        return self

    def fit_resample(self, X, y):
        self.fit(X, y)
        indices = self._fit_resample(X, y)
        return (
            _safe_indexing(X, indices),
            _safe_indexing(X, indices)
        )

    @abc.abstractmethod
    def _fit_resample(self, X, y):
        pass
