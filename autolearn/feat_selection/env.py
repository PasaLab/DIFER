import os
import copy
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend

# import weka.core.converters as converters
# import weka.core.jvm as jvm
# from weka.classifiers import Classifier, Evaluation
# from weka.core.classes import Random
# from weka.filters import Filter

from autolearn.data import Dataset
from autolearn.utils import timeit, log as logger
from autolearn.metrics import r2_score
from autolearn.feat_selection.nfo.search_space import *

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics.scorer import make_scorer
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor


project_path = Path(os.path.abspath(os.path.dirname(__file__)))


def return_0():
    return 0


def return_false():
    return False


def return_num():
    return OpType.NUM


class Environment:
    def __init__(self, dataset_path, max_order=4, cv=5, parallel=True, use_cat_space=False):
        self.dataset_path = dataset_path
        self.dataset = Dataset(dataset_path)
        self.use_cat_space = use_cat_space
        if 'cat' in set(self.dataset.meta.values()) and use_cat_space:
            self.op_info = all_op_info
        else:
            self.op_info = num_op_info
        self._op_map, self._ops = None, None
        self._op_arity, self._op_order = None, None
        self._type_dict = None
        self.max_order = max_order
        self.max_length = max(self.op_arity.values()) ** (self.max_order + 1) - 1
        self.cv = cv
        self.parallel = parallel

    @property
    def op_map(self):
        if self._op_map is None:
            self._op_map = {op[0]: op for op in self.op_info}
        return self._op_map
    
    @property
    def ops(self):
        if self._ops is None:
            self._ops = [op[0] for op in self.op_info]
        return self._ops

    @property
    def type_dict(self):
        if self._type_dict is None and self.use_cat_space:
            type_map = {'num': OpType.NUM, 'cat': OpType.CAT}
            self._type_dict = defaultdict(return_num)
            self._type_dict.update({
                key: type_map[value] for key, value in self.dataset.meta.items() if value in type_map
            })
            self._type_dict.update({
                op[0]: op[4] for op in self.op_info
            })
        elif self._type_dict is None:
            self._type_dict = defaultdict(return_num)
        return self._type_dict

    @property
    def op_arity(self):
        if self._op_arity is None:
            self._op_arity = defaultdict(return_0)
            for op in self.op_info:
                self._op_arity[op[0]] = op[1]
        return self._op_arity

    @property
    def op_order(self):
        if self._op_order is None:
            self._op_order = defaultdict(return_false)
            for op in self.op_info:
                self.op_order[op[0]] = op[2]
        return self._op_order

    @property
    def features(self):
        return list(map(str, self.dataset.features))

    def _weka_evaluate_r(self, data):
        data.class_is_last()
        model = Classifier(classname='weka.classifiers.trees.RandomForest')
        evl = Evaluation(data)
        evl.crossvalidate_model(model, data, self.cv, Random(0))
        s = 1 - evl.relative_absolute_error / 100
        return s

    def _weka_evaluate_c(self, data):
        weka_filter = Filter(
            classname="weka.filters.unsupervised.attribute.NumericToNominal",
            options=["-R", "last"]
        )
        weka_filter.inputformat(data)
        data = weka_filter.filter(data)
        data.class_is_last()
        model = Classifier(classname='weka.classifiers.trees.RandomForest')
        evl = Evaluation(data)
        evl.crossvalidate_model(model, data, self.cv, Random(0))
        fscore = evl.weighted_f_measure
        s = fscore
        return s

    def add_feature(self, x, feat, name='new'):
        # x.insert(0, name, feat)
        x[name] = feat
        return x

    def add_features(self, x, feats):
        for i, feat in enumerate(feats):
            self.add_feature(x, feat, name=f"new_{i}")
        return x

    def weka_evaluate(self, feats):
        if not jvm.started:
            jvm.start()
        x = copy.deepcopy(self.dataset.instances)
        y = copy.deepcopy(self.dataset.labels)
        x = self.add_features(x, feats)
        x['y'] = y
        d = x.shape[0]
        x = str(x.values.tolist())

        data = np.reshape(eval(x), [d, -1], order='C')
        data = data.astype(np.float64)
        data = converters.ndarray_to_instances(
            data, relation='tmp'
        )
        if self.dataset.task == 'regression':
            score = self._weka_evaluate_r(data)
        elif self.dataset.task == 'classification':
            score = self._weka_evaluate_c(data)
        else:
            score = -1
        return score

    def get_feat(self, feat):
        feat = int(feat)
        return np.asarray(self.dataset.instances[feat].values, dtype=np.float)

    def get_op(self, op):
        return self.op_map[op][3]

    def construct_feature(self, feat):
        return feat.generate(self)

    def _evaluate(self, feats):
        return self.weka_evaluate(feats)

    def _seq_eval(self, features):
        scores = []
        for feature in tqdm(features):
            scores.append(self._evaluate([feature]))
        return scores

    def _parallel_eval(self, features):
        with parallel_backend("multiprocessing", n_jobs=4):
            scores = Parallel()(
                delayed(self._evaluate)([feature]) for feature in features
            )
        return scores

    # def fake_eval(self, features):
    #     return 0.0

    @timeit
    def eval_features(self, features):
        res = -np.ones(len(features), dtype=np.float)
        useful_features = []
        translated_features = []
        for i, feature in enumerate(features):
            try:
                feat_i = self.construct_feature(feature)
                useful_features.append(i)
                translated_features.append(feat_i)
            except Exception as e:
                logger(f"Error in construct {feature} {e}", level='error')
        if self.parallel:
            scores = self._parallel_eval(translated_features)
        else:
            scores = self._seq_eval(translated_features)
        res[useful_features] = scores
        return res

    def eval_set(self, features):
        useful_feats = []
        for feat in features:
            try:
                feat = self.construct_feature(feat)
                useful_feats.append(feat)
            except Exception as e:
                logger(f"Error in construct {feat} {e}", level='error')
        score = self._evaluate(useful_feats)
        return score

    @property
    def vocabulary(self):
        return self.features + self.ops


class SklearnEnv(Environment):
    def __init__(self, *args, n_estimators=10, scoring="f1_micro", model="RF", **kwargs):
        super(SklearnEnv, self).__init__(*args, **kwargs)
        self.n_estimators = n_estimators
        self.scoring = scoring
        self.model = model

    def sklearn_evaluate(self, feats):
        x = copy.deepcopy(self.dataset.instances)
        y = copy.deepcopy(self.dataset.labels)
        x = self.add_features(x, feats)
        x.replace([np.inf, -np.inf], np.nan, inplace=True)
        x.fillna(-1, inplace=True)

        if self.dataset.task == 'regression':
            if self.model == 'LR':
                model = Lasso()
            elif self.model == 'SVM':
                model = LinearSVR()
            elif self.model == "LGB":
                model = LGBMRegressor()
            elif self.model == "XGB":
                model = XGBRegressor()
            else:
                model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=0)
            score = cross_val_score(model, x, y, scoring=make_scorer(r2_score), cv=int(self.cv)).mean()
        elif self.dataset.task == 'classification':
            if self.model == 'LR':
                model = LogisticRegression()
            elif self.model == 'SVM':
                model = LinearSVC()
            elif self.model == 'LGB':
                model = LGBMClassifier()
            elif self.model == "XGB":
                model = XGBClassifier()
            else:
                model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=0)
            score = cross_val_score(model, x, y, scoring=self.scoring, cv=int(self.cv)).mean()
        else:
            score = -1
        return score

    def _evaluate(self, feats):
        return self.sklearn_evaluate(feats)


