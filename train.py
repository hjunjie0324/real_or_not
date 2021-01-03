import torch
import torch.nn as nn
import numpy as np 
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils import preprocess
from model.bertClassifier import BertClassifier

class RealoOrNotDataset(Dataset):
    def __init__(self,encodings):
        self.encodings = encodings
    def __getitem__(self,idx):
        return {key:torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)


def train(encodings,dataloader,optimizer,model,device, num_epoches):
    for epoch in num_epoches:
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target = batch['target'].to(device)

            outputs = model(input_ids, attention_mask, target)
            loss = outputs[0]
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(),"bertClassifier.pt")


def main():
    dataset_name = "train.csv"
    encodings = preprocess.preprocess(dataset_name)
    train_dataset = RealoOrNotDataset(encodings)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BertClassifier()
    model.to(device)
    parameters = model.parameters
    optimizer = optim.Adam(parameters, lr=1e-5)
    dataloader = DataLoader(train_dataset,batch_size=4,shuffle=True)
    num_epoches = 3
    train(encodings, dataloader, optimizer, model, device, num_epoches)
