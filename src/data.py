import torch
from torch.utils.data import Dataset

from .config import Config


class QDataset(Dataset):

    def __init__(self, csv_file, tokenizer, labels_table={}):
        self.labels_table = labels_table
        self.labels_count = {}
        self.tokenizer = tokenizer
        self.labels = []
        self.train_labels = []
        self.questions = []
        self.tokens = []
        csv = open(csv_file)
        lines = csv.readlines()
        for line in lines:
            assert " " in line
            label = line[:line.index(" ")]
            question = line[line.index(" ") + 1:]
            if label not in self.labels_table:
                self.labels_table[label] = len(self.labels_table)
            if label not in self.labels_count:
                self.labels_count[label] = 0
            self.labels_count[label] += 1
            self.labels.append(self.labels_table[label])
            if self.labels_count[label] <= Config.max_labels_per_classes:
                self.train_labels.append(self.labels_table[label])
            else:
                self.train_labels.append(-1)
            self.questions.append(question)
            tokens = tokenizer(question)
            self.tokens.append(tokens)
        self.nclasses = len(self.labels_table)

    def pad(self, tokens):
        while len(tokens['input_ids']) < Config.max_seq_len:
            tokens['input_ids'].append(0)
            tokens['attention_mask'].append(0)
            tokens['token_type_ids'].append(0)
        return tokens

    def toTensor(self, tokens):
        tokens = self.pad(tokens)
        tokens['input_ids'] = torch.tensor(tokens['input_ids'], dtype=torch.int, device=Config.device)
        tokens['attention_mask'] = torch.tensor(tokens['attention_mask'], dtype=torch.int, device=Config.device)
        tokens['token_type_ids'] = torch.tensor(tokens['token_type_ids'], dtype=torch.int, device=Config.device)
        return tokens

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return (
            self.toTensor(self.tokens[idx]),
            torch.tensor(self.train_labels[idx], dtype=torch.long).to(device=Config.device),
            torch.tensor(self.labels[idx], dtype=torch.long).to(device=Config.device)
        )
