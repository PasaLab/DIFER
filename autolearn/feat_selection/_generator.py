from autolearn.feat_engineering.pipeline import Pipeline, Step


class AllOpGenerator:

    def __init__(self, ops):
        self.ops = ops
        self.pipeline = None

    def generator(self, X):
        src_cols = list(X.columns)
        steps = [Step(op, src_cols, None, None) for op in self.ops]
        transformer = Pipeline(None, steps)
        res = transformer.fit_transform(X[src_cols])
        self.pipeline = transformer
        return res

    def re_generator(self, X):
        return self.pipeline.transform(X)
