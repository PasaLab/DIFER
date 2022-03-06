import json
import numpy as np
from collections import defaultdict, ChainMap

from autolearn.utils import log


class Config:
    """
    统一管理全局超参数, 如模型序列, 数据处理方式, batch size等
    """
    def __init__(self, filename="", config=None):
        self.filename = filename
        if config is None:
            self.config = defaultdict(lambda: None)
            with open(filename, 'r') as f:
                self.config = json.load(f)
            log(f"Use config {filename}:\n{self.config}")
        else:
            self.config = config
        self.model_list = None

    def __getitem__(self, item):
        return self.config[item]

    def __setitem__(self, key, value):
        self.config[key] = value

    def __delitem__(self, key):
        del self.config[key]

    def __getattr__(self, item):
        return self.config.get(item, None)

    def __str__(self):
        return str(self.config)

    def child_config(self, name):
        config = {}
        if name in self.config:
            config = self.config[name]
        config = ChainMap(config, self.config)
        return Config(config=config)

