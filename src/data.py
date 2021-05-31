import torch
from torch.utils.data import Dataset

from .config import Config


class QDataset(Dataset):

    def __init__(self, csv_file, tokenizer, labels_table={}):
        self.labels_table = labels_table
        self.tokenizer = tokenizer
        self.labels = []
        self.questions = []
        self.tokens = []
        csv = open(csv_file)
        lines = csv.readlines()
        self.max_len = 0
        for line in lines:
            assert " " in line
            label = line[:line.index(" ")]
            question = line[line.index(" ") + 1:]
            if label not in self.labels_table:
                self.labels_table[label] = len(self.labels_table)
            self.labels.append(self.labels_table[label])
            self.questions.append(question)
            tokens = tokenizer(question)
            self.tokens.append(tokens)
            if len(tokens['input_ids']) > self.max_len:
                self.max_len = len(tokens['input_ids'])
        self.nclasses = len(self.labels_table)

    def toTensor(self, tokens):
        while len(tokens['input_ids']) < self.max_len:
            tokens['input_ids'].append(0)
            tokens['attention_mask'].append(0)
            tokens['token_type_ids'].append(0)
        tokens['input_ids'] = torch.tensor(tokens['input_ids'], dtype=torch.int, device=Config.device)
        tokens['attention_mask'] = torch.tensor(tokens['attention_mask'], dtype=torch.int, device=Config.device)
        tokens['token_type_ids'] = torch.tensor(tokens['token_type_ids'], dtype=torch.int, device=Config.device)
        return tokens

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.toTensor(self.tokens[idx]), torch.tensor(self.labels[idx], dtype=torch.long).to(device=Config.device)
