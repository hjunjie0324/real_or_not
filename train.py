import torch
import torch.nn as nn
import numpy as np 
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils import preprocess
from model.bertClassifier import BertClassifier
from transformers import AdamW, get_linear_schedule_with_warmup

class RealOrNotDataset(Dataset):
    def __init__(self,encodings):
        self.encodings = encodings
    def __getitem__(self,idx):
        return {key:torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)


def train(encodings,dataloader,optimizer, scheduler, model,device, num_epoches):
    model.train()
    curr_loss = 0
    all_loss = []
    iteration = 1
    plot_every = 10
    for epoch in range(num_epoches):
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target = batch['target'].to(device)

            output = model(input_ids, attention_mask)
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(output, target)
            loss.backward()
            #clip gradient
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            scheduler.step()

            curr_loss = curr_loss + loss.item()
            if iteration % plot_every == 0:
                all_loss.append(curr_loss/plot_every)
                curr_loss = 0
            iteration += 1
    model.eval()
    torch.save(model.state_dict(),"bertClassifier.pt")

    with open('loss.txt', 'w') as filehandle:
        for loss in all_loss:
            filehandle.write('%s\n' % loss)


if __name__ == "__main__":
    dataset_name = "train.csv"
    encodings = preprocess.preprocess(dataset_name)
    train_dataset = RealOrNotDataset(encodings)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BertClassifier()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=3e-5)
    dataloader = DataLoader(train_dataset,batch_size=16,shuffle=True)
    num_epoches = 4
    total_steps = len(dataloader)*num_epoches
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps = total_steps)

    train(encodings, dataloader, optimizer, scheduler, model, device, num_epoches)

