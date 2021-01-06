import torch
import torch.nn as nn
import numpy as np 
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils import preprocess
from model.bertClassifier import BertClassifier

class RealOrNotDataset(Dataset):
    def __init__(self,encodings):
        self.encodings = encodings
    def __getitem__(self,idx):
        return {key:torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)


def train(encodings,dataloader,optimizer,model,device, num_epoches):
    model.train()
    for epoch in range(num_epoches):
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target = batch['target'].to(device)

            output = model(input_ids, attention_mask, target)
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
    model.eval()
    torch.save(model.state_dict(),"bertClassifier.pt")


if __name__ == "__main__":
    dataset_name = "train.csv"
    encodings = preprocess.preprocess(dataset_name)
    train_dataset = RealOrNotDataset(encodings)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BertClassifier()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    dataloader = DataLoader(train_dataset,batch_size=16,shuffle=True)
    num_epoches = 4
    train(encodings, dataloader, optimizer, model, device, num_epoches)

