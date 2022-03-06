import json
import os
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


class Dataset:

    TASK_MAP = {
        'R': 'regression',
        'C': 'classification'
    }

    def __init__(self, name):
        project_path = os.path.abspath(os.path.dirname(__file__))
        data_path = f"{project_path[:project_path.find('autolearn')]}data"
        path = Path(data_path)
        self.dataset_name = name
        self.data = pd.read_csv(
            path / f"{name}.csv",
            header=None
        )
        with open(path / f"{name}.json", 'r') as f:
            self.meta = json.load(f)
        self._x = None
        self._y = None
        self._label_encoder = LabelEncoder()

    @property
    def instances(self):
        if self._x is None:
            self._x = self.data.iloc[:, :-1]
        return self._x

    @property
    def labels(self):
        if self._y is None:
            self._y = self.data.iloc[:, -1]
            if self.task == Dataset.TASK_MAP['C']:
                self._y = self._label_encoder.fit_transform(self._y)
        return self._y

    @property
    def task(self):
        return Dataset.TASK_MAP[self.meta['task']]

    @property
    def time_budget(self):
        return self.meta.get('time_budget', 24 * 3600)

    @property
    def features(self):
        return self.instances.columns


if __name__ == '__main__':
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import cross_val_score

    dataset = Dataset('spectf')
    x = dataset.data.iloc[:, :-1]
    y = dataset.data.iloc[:, -1]
    y = LabelEncoder().fit_transform(y)
    s = cross_val_score(
        RandomForestClassifier(n_estimators=10, random_state=0),
        x, y,
        scoring='f1_micro',
        cv=5
    ).mean()
    print(dataset.task)
    print(s)
