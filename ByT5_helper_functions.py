class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, y_encodings=None):
        self.encodings = encodings

    def __getitem__(self, idx):
        item= {key: val[idx] for key, val in self.encodings.items()}
        #item['labels'] = {key: torch.tensor(val[idx]) for key, val in self.y_encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['labels'])