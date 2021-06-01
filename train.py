import numpy as np
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer

from src.data import QDataset
from src.model import GanBert, Generator, DiscriminatorLoss, GeneratorLoss
from src.config import Config

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
trainset = QDataset("./traininig_dataset.txt", tokenizer)
valset = QDataset("./validation_dataset.txt", tokenizer, trainset.labels_table)
trainloader = DataLoader(trainset, batch_size=Config.batch_size, shuffle=True, drop_last=True)
valloader = DataLoader(valset, batch_size=Config.batch_size, shuffle=True, drop_last=True)

model = GanBert(trainset.nclasses)
model.to(Config.device)
generator = Generator()
generator.to(Config.device)

gen_loss_fn = GeneratorLoss()
dis_loss_fn = DiscriminatorLoss()

gen_optimizer = torch.optim.Adam(generator.parameters(), lr=Config.gen_learning_rate)
dis_optimizer = torch.optim.Adam(model.parameters(), lr=Config.dis_learning_rate)


for e in range(Config.epoch):
    print(f"Epoch {e}")
    model.train()
    running_acc = []
    running_adv_acc = []
    for batch, train_labels, labels in tqdm(trainloader):
        gen_optimizer.zero_grad()
        dis_optimizer.zero_grad()
        adv = generator(len(labels))
        probs, features = model(batch, adv)
        dis_loss = dis_loss_fn(probs, train_labels)
        gen_loss = gen_loss_fn(probs, features)
        dis_loss.backward(retain_graph=True)
        gen_loss.backward()
        dis_optimizer.step()
        gen_optimizer.step()
        acc = (probs[:len(labels),:-1].argmax(1) == labels).sum().item() / len(labels)
        adv_acc_fn = (probs[len(labels):].argmax(1) == trainset.nclasses).sum().item() / len(labels)
        adv_acc_fp = (probs[:len(labels)].argmax(1) != trainset.nclasses).sum().item() / len(labels)
        running_acc.append(acc)
        running_adv_acc.append(adv_acc_fp)
        running_adv_acc.append(adv_acc_fn)
    print(f"Training accuracy = {np.mean(running_acc)}")
    print(f"Training adversary accuracy = {np.mean(running_adv_acc)}")
    model.eval()
    running_acc = []
    running_adv_acc = []
    with torch.no_grad():
        for batch, train_labels, labels in tqdm(valloader):
            adv = generator(len(labels))
            probs, features = model(batch, adv)
            acc = (probs[:len(labels),:-1].argmax(1) == labels).sum().item() / len(labels)
            adv_acc_fn = (probs[len(labels):].argmax(1) == trainset.nclasses).sum().item() / len(labels)
            adv_acc_fp = (probs[:len(labels)].argmax(1) != trainset.nclasses).sum().item() / len(labels)
            running_acc.append(acc)
            running_adv_acc.append(adv_acc_fp)
            running_adv_acc.append(adv_acc_fn)
    print(f"Validation accuracy = {np.mean(running_acc)}")
    print(f"Validation adversary accuracy = {np.mean(running_adv_acc)}")
print("Save model")
torch.save(model.state_dict(), Config.save_to)

