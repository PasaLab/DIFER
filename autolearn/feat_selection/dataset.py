from torch.utils.data import Dataset


class PlainDataset(Dataset):
    
    def __init__(self, x, y):
        super(PlainDataset, self).__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.x[item], self.y[item]


class SequenceDataset(Dataset):

    def __init__(self, x, y):
        super(SequenceDataset, self).__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.x[item], self.y[item].reshape(-1)


class MixDataset(Dataset):
    def __init__(self, x, y1):
        super(MixDataset, self).__init__()
        self.x1, self.y1 = x, y1
        self.x2, self.y2 = x, x

    def __len__(self):
        return len(self.y1)

    def __getitem__(self, item):
        return self.x1[item], self.y1[item], self.x2[item], self.y2[item]
