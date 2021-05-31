from transformers import BertModel
import torch
from .config import Config


class Generator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden_layer = torch.nn.Linear(Config.noise, Config.generator_hidden_layer)
        self.activation = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(Config.dropout)
        self.output_layer = torch.nn.Linear(Config.generator_hidden_layer, Config.bert_features)

    def forward(self, batch_size):
        noise = torch.normal(torch.zeros([batch_size, 100]), torch.ones([batch_size, 100])).to(Config.device)
        return self.output_layer(self.dropout(self.activation(self.hidden_layer(noise))))


class Discriminator(torch.nn.Module):

    def __init__(self, nclasses):
        super().__init__()
        self.hidden_layer = torch.nn.Linear(Config.bert_features, Config.discriminator_hidden_layer)
        self.activation = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(Config.dropout)
        self.softmax = torch.nn.Softmax()
        self.output_layer = torch.nn.Linear(Config.discriminator_hidden_layer, nclasses + 1)

    def forward(self, features):
        return self.softmax(self.output_layer(self.dropout(self.activation(self.hidden_layer(features)))))


class GanBert(torch.nn.Module):

    def __init__(self, nclasses):
        super().__init__()
        self.discriminator = Discriminator(nclasses)
        self.bert = BertModel.from_pretrained("bert-base-uncased")

    def forward(self, inputs, adv):
        features = self.bert(**inputs)['last_hidden_state']
        all_features = torch.cat([features[:,0], adv])
        return self.discriminator(torch.cat([features[:,0], adv])), all_features


class DiscriminatorLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nll = torch.nn.NLLLoss()
        self.bce = torch.nn.BCELoss()

    def forward(self, probs, labels):
        assert len(probs) // 2 == len(labels)
        split = len(probs) // 2
        nll_loss = self.nll(torch.log(probs[:split]), labels)
        bce_loss = self.bce(probs[:,-1], torch.cat([torch.zeros([split]), torch.ones([split])]).to(device=Config.device))
        return bce_loss + nll_loss


class GeneratorLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = torch.nn.BCELoss()

    def forward(self, probs, features):
        split = len(probs) // 2
        bce_loss = self.bce(probs[split:,-1], torch.zeros([split]).to(device=Config.device))
        f_loss = torch.pow(features[:split].mean(0) - features[split:].mean(0), 2).mean()
        return bce_loss + f_loss
