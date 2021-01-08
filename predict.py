from utils import preprocess
import torch
import torch.nn as nn
import numpy as np 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from model.bertClassifier import BertClassifier
from train import RealOrNotDataset
import pandas as pd
import torch.nn.functional as F

def predict(test_dataset, threshold):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BertClassifier()
    model.to(device)
    model.train()
    model.load_state_dict(torch.load('bertClassifier.pt'))
    model.eval()

    n = len(test_dataset)
    prediction_list = []
    for i in range(n):
        input_ids = test_dataset[i]['input_ids'].to(device)
        attention_mask = test_dataset[i]['attention_mask'].to(device)
        with torch.no_grad():
            output = model(torch.unsqueeze(input_ids,0), torch.unsqueeze(attention_mask,0))
            prob = F.softmax(output)
            prob = prob.cpu().numpy()[0][1]
            #pred = torch.argmax(output)
            if prob > threshold:
                pred = 1
            else:
                pred = 0
            prediction_list.append(pred)
    return prediction_list 


if __name__ == "__main__":
    dataset_name = "test.csv"
    encodings = preprocess.preprocess_for_test(dataset_name)
    test_dataset = RealOrNotDataset(encodings)

    threshold = 0.9
    prediction = predict(test_dataset, threshold)

    sample_submission = pd.read_csv("sample_submission.csv")
    sample_submission["target"] = prediction
    sample_submission.to_csv("submission.csv",index=False)