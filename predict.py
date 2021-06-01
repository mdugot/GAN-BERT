import argparse

import torch
from transformers import BertTokenizer

from src.data import QDataset
from src.model import GanBert
from src.config import Config

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
parser = argparse.ArgumentParser(description="Classify a question with a trained model")
parser.add_argument("model", help="path to the model checkpoint", type=str)
parser.add_argument("question", help="question to classify", type=str)
args = parser.parse_args()

trainset = QDataset("./traininig_dataset.txt", tokenizer)

model = GanBert(trainset.nclasses)
model.to(Config.device)
model.load_state_dict(torch.load(args.model, map_location=Config.device))
model.eval()
tokens = trainset.toTensor(tokenizer(args.question))
tokens['input_ids'] = tokens['input_ids'].unsqueeze(0)
tokens['attention_mask'] = tokens['attention_mask'].unsqueeze(0)
tokens['token_type_ids'] = tokens['token_type_ids'].unsqueeze(0)
probs = model.discriminator(model.bert(**tokens)['last_hidden_state'][:,0])
probs = probs[0,:-1]
prediction = probs.argmax().item()
print(f"\nPrediction : {list(trainset.labels_table.keys())[prediction]}")
