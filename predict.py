from utils import preprocess
import torch
import torch.nn as nn
import numpy as np 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from model.bertClassifier import BertClassifier
from train import RealOrNotDataset
import pandas as pd

def predict(test_dataset,midel,device):
    n = len(test_dataset)
    prediction_list = []
    for i in range(n):
        input_ids = test_dataset[i]['input_ids'].to(device)
        attention_mask = test_dataset[i]['attention_mask'].to(device)
        with torch.no_grad():
            _, output = model(input_ids, attention_mask)
            pred = torch.argmax(output)
            prediction_list.append(pred)
    return prediction_list 


if __name__ == "__main__":
    dataset_name = "test.csv"
    encodings = preprocess.preprocess(dataset_name)
    test_dataset = RealOrNotDataset(encodings)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BertClassifier()
    model.load_state_dict(torch.load('bertClassifier.pt'))
    model.to(device)
    model.eval()
    prediction = predict(test_dataset,model,device)

    sample_submission = pd.read_csv("sample_submission.csv")
    sample_submission["target"] = prediction
    sample_submission.to_csv("submission.csv",index=False)